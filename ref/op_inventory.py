"""SD1.5 的「算子 × 形状」清单：在 CPU 上（fp32）跑一遍 CLIP 文本编码器、UNet 一次前向、VAE 解码器，
截下每一次 torch 函数级调用，按（算子、形状、关键参数）归并，算 FLOPs 与字节，并对照 calBLAS 的模板标「现成 / 改参数 / 新写」。

为什么要这张表：
- 决定先写哪些 kernel（按 FLOPs 与次数排）、哪些形状要特化（形状是编译期常量，每个形状一份 kernel）；
- 它就是报告里「NPU 贡献表」的底稿：每一类运算最后落在 NPU 还是 CPU，都从这张表逐行勾。

怎么截的：TorchFunctionMode 只截「最外一层」的函数调用（mode 的处理函数里再调 torch 不会被二次截到），
所以 F.group_norm 只记一次，不会连带记它内部的 aten 调用。纯视图类（view / permute / chunk …）单独计数、不进清单，
但会落数据的排布变化（contiguous、cat）照记——在 NPU 上它们是真搬运。

用法：<venv>/bin/python -I ref/op_inventory.py <SD1.5 模型目录> <输出目录>
输出：<输出目录>/{clip,unet,vae}.json（逐个签名的原始记录）与 <输出目录>/README.md（汇总表）。
"""

from __future__ import annotations

import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch
from torch.overrides import TorchFunctionMode

# 只改看法、不落数据的调用：不进清单（另计次数，看个大概）
VIEW_OPS = {
    "view", "reshape", "permute", "transpose", "flatten", "unflatten", "unsqueeze", "squeeze", "expand", "chunk", "split",
    "size", "dim", "shape", "__getitem__", "to", "type", "float", "half", "bfloat16", "__len__", "numel", "is_floating_point",
    "dtype", "device", "requires_grad", "__get__", "__set__", "is_contiguous", "stride", "storage_offset", "data_ptr",
    "element_size", "__format__", "__bool__", "__int__", "__index__", "item", "tolist", "__iter__", "unbind", "narrow",
    "is_complex", "get_device", "_has_compatible_shallow_copy_type", "__hash__", "__eq__", "__ne__", "__contains__",
}


def _name(func) -> str:
    n = getattr(func, "__name__", None) or str(func)
    return n


def _shape(x):
    if isinstance(x, torch.Tensor):
        return list(x.shape)
    if isinstance(x, (list, tuple)) and x and all(isinstance(t, torch.Tensor) for t in x):
        return [list(t.shape) for t in x]
    return None


def _tensors(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            yield from _tensors(o)
    elif isinstance(obj, dict):
        for o in obj.values():
            yield from _tensors(o)


def _arg(args, kwargs, i, key, default=None):
    if key in kwargs:
        return kwargs[key]
    return args[i] if len(args) > i else default


def _pair(v):
    return list(v) if isinstance(v, (list, tuple)) else [v, v]


def describe(name: str, args, kwargs, out):
    """（签名里的关键参数, FLOPs）。签名决定 kernel 特化：同一签名 = 同一份 kernel。"""
    key: dict = {}
    flops = 0
    if name == "conv2d":
        x, w = args[0], args[1]
        stride, pad = _pair(_arg(args, kwargs, 3, "stride", 1)), _pair(_arg(args, kwargs, 4, "padding", 0))
        groups = _arg(args, kwargs, 6, "groups", 1)
        key = {"k": list(w.shape[2:]), "stride": stride, "pad": pad, "groups": groups, "bias": _arg(args, kwargs, 2, "bias") is not None}
        n, cout, ho, wo = out.shape
        flops = 2 * n * cout * ho * wo * (w.shape[1]) * w.shape[2] * w.shape[3]
    elif name == "linear":
        x, w = args[0], args[1]
        key = {"bias": _arg(args, kwargs, 2, "bias") is not None}
        flops = 2 * (x.numel() // x.shape[-1]) * w.shape[0] * w.shape[1]
    elif name == "scaled_dot_product_attention":
        q, k = args[0], args[1]
        b, h, l, d = q.shape
        s = k.shape[-2]
        key = {"mask": _arg(args, kwargs, 3, "attn_mask") is not None, "causal": bool(_arg(args, kwargs, 5, "is_causal", False)),
               "kind": "self" if l == s else "cross"}
        flops = 4 * b * h * l * s * d
    elif name == "group_norm":
        key = {"groups": _arg(args, kwargs, 1, "num_groups"), "eps": _arg(args, kwargs, 4, "eps", 1e-5)}
    elif name == "layer_norm":
        key = {"normalized_shape": list(_arg(args, kwargs, 1, "normalized_shape")), "eps": _arg(args, kwargs, 4, "eps", 1e-5)}
    elif name == "gelu":
        key = {"approximate": kwargs.get("approximate", args[1] if len(args) > 1 else "none")}
    elif name == "interpolate":
        key = {"scale_factor": _arg(args, kwargs, 2, "scale_factor"), "mode": _arg(args, kwargs, 3, "mode", "nearest"),
               "size": _arg(args, kwargs, 1, "size")}
    elif name in ("cat", "concat", "stack"):
        key = {"dim": _arg(args, kwargs, 1, "dim", 0)}
    elif name == "softmax":
        key = {"dim": _arg(args, kwargs, 1, "dim")}
    elif name in ("baddbmm", "bmm", "matmul", "__matmul__"):
        a = [t for t in _tensors(args)]
        if name == "baddbmm":
            a = a[1:]
        if len(a) >= 2 and a[0].dim() >= 2:
            flops = 2 * a[0].numel() * a[1].shape[-1]
    return key, flops


def mapping(comp: str, name: str, key: dict, shapes) -> tuple[str, str]:
    """对照 calBLAS 的模板：（状态, 说明）。状态 ∈ 现成 / 改参数 / 新写 / 主机 / 不搬。"""
    if name == "conv2d":
        k, s = key["k"], key["stride"]
        x = shapes[0]
        cin, cout = x[1], shapes[1][0]
        if k == [3, 3] and s in ([1, 1], [2, 2]) or k == [1, 1] and s == [1, 1]:
            pad_note = []
            if cin % 32:
                pad_note.append(f"Cin {cin} 补到 32 的倍数")
            if cout % 16:
                pad_note.append(f"Cout {cout} 补齐")
            if key["k"] == [3, 3] and s == [2, 2] and key["pad"] == [0, 0]:
                pad_note.append("diffusers 下采样是非对称补边（右下补 1），与模板的四周补边要核对")
            return ("改参数" if pad_note else "现成", "conv2d!（NHWC 带边框）" + ("；" + "；".join(pad_note) if pad_note else ""))
        return ("新写", f"conv2d! 只收核 1 / 3、步长 1 / 2，这里核 {k}、步长 {s}")
    if name == "group_norm":
        return ("现成", "group_norm!（NCHW 连续段；和 NHWC 的卷积之间要转排布 ⇒ NHWC 原生版是 A1 候选）；后面紧跟的 SiLU 可并进 act")
    if name == "silu":
        return ("现成", "并进 group_norm! 的 act，或 unary!(silu)")
    if name == "linear":
        return ("现成", "linear!(rows = T, w = hf, bias…)")
    if name == "layer_norm":
        return ("现成", "layer_norm!（每 token 一行）")
    if name == "scaled_dot_product_attention":
        d = shapes[0][-1]
        dn = f"；头维 {d} 补零到 16 的倍数（scale 给真头维）" if d % 16 else ""
        if key["causal"] or key["mask"]:
            return ("改参数", "attention!(mask = full)：因果 + pad 的掩码由主机给" + dn)
        if key["kind"] == "cross":
            return ("改参数", f"attention!(keys = 128, mask = keys)：{shapes[1][-2]} 个上下文 token 补到 128" + dn)
        return ("改参数" if dn else "现成", "spatial_attn! 或 attention!(mask = keys)" + dn)
    if name == "gelu":
        if key.get("approximate") == "tanh":
            return ("现成", "unary!(gelu)（tanh 形式）")
        return ("改参数", "erf-GELU 的片段现成（frag.rs 的 act!(gelu_erf)，A&S 7.1.26，f32）：GEGLU 把投影拆成门、值两个 linear!，"
                         "门那个带 act = gelu_erf，再 binary!(mul)；不用新写 kernel")
    if name == "interpolate":
        if key.get("mode") == "nearest" and key.get("scale_factor") in (2, 2.0, [2.0, 2.0], (2.0, 2.0)):
            return ("现成", "upsample2!（一次 ≤ 256 通道，多的分几次调）")
        return ("新写", f"interpolate {key}")
    if name in ("cat", "concat"):
        return ("不搬", "通道窗口：生产者直接写进大图的通道窗口（vision.rs 的约定），不起 kernel")
    if name in ("div", "__truediv__", "__rtruediv__") and comp in ("unet", "vae"):
        return ("不搬", "diffusers 残差块的 output_scale_factor（SD1.5 里 = 1）：除以 1 直接去掉；别的除数并进相邻 kernel 的系数")
    if name in ("add", "__add__", "__iadd__", "__radd__", "sub", "__sub__", "__rsub__", "mul", "__mul__", "__rmul__", "__imul__",
                "div", "__truediv__", "__rtruediv__"):
        return ("现成", "binary!（或并进卷积的残差 / bias）")
    if name == "embedding":
        return ("现成", "embedding!")
    if name in ("sigmoid",):
        return ("改参数", "CLIP 的 quick_gelu = x · σ(1.702x)：新写一个 unary，或 binary!(mul_sigmoid) 前接一次缩放" if comp == "clip" else "unary!(sigmoid)")
    if name == "arange" and comp == "clip":
        return ("不搬", "位置号 0..76 是常量：打包时写进参数块，embedding! 按它取行")
    if name in ("exp", "sin", "cos", "arange"):
        return ("新写", "时间步嵌入（sin / cos）：设备上一个小 kernel（VU 有 sin / cos 原语）；时间步可能是小数（ComfyUI 由 sigma 插出来），"
                         "所以不能只靠打包时算好的整数表")
    if name in ("contiguous", "clone"):
        return ("不搬", "排布：整步程序里按目标排布直接写，不单独搬")
    if name == "dropout":
        return ("不搬", "推理时恒等（eval 模式下 dropout 不改数），整步程序里直接去掉")
    if name in ("argmax", "tensor"):
        return ("不搬", "CLIP 池化输出取 EOS 位置用的；SD1.5 只用 last_hidden_state，整步程序里不算" if comp == "clip"
                else "常量构造（时间步张量之类），由主机写进输入")
    if name in ("tanh",):
        return ("改参数", "TAESD 开头的 tanh(x / 3) · 3：VU 有 tanh 原语，写成一个小的逐元素 kernel 或并进第一个卷积之前的一步")
    if name in ("relu", "relu_"):
        return ("新写", "conv2d! 的 act 只有 silu、none 两种：给卷积模板加一个 relu 分支（抄模板要记 VENDORED）；Block 末尾的 relu(conv + skip) 可用 binary!(relu)")
    if name in ("clamp", "clamp_", "clip"):
        return ("现成", "逐元素 min / max（并进输出转 uint8 的那一步）")
    return ("待定", "")


class Recorder(TorchFunctionMode):
    def __init__(self, comp: str, module_stack: list[str]):
        super().__init__()
        self.comp = comp
        self.stack = module_stack
        self.records: "OrderedDict[str, dict]" = OrderedDict()
        self.view_calls = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = _name(func)
        if name in VIEW_OPS:
            self.view_calls += 1
            return out
        ins = [t for t in _tensors(args)] + [t for t in _tensors(kwargs)]
        if not ins and not isinstance(out, torch.Tensor):
            return out
        shapes = [list(t.shape) for t in ins]
        outs = [t for t in _tensors(out)]
        try:
            key, flops = describe(name, args, kwargs, out)
        except Exception as e:  # 描述失败不该让参照跑不下去，但要看得见
            key, flops = {"describe_error": repr(e)}, 0
        sig = json.dumps([name, shapes, key], sort_keys=True, default=str)
        r = self.records.get(sig)
        if r is None:
            status, note = mapping(self.comp, name, key, shapes)
            r = self.records[sig] = {
                "op": name, "in_shapes": shapes, "out_shapes": [list(t.shape) for t in outs], "key": key, "count": 0,
                "flops_each": flops, "in_elems_each": sum(t.numel() for t in ins), "out_elems_each": sum(t.numel() for t in outs),
                "where": [], "status": status, "note": note,
            }
        r["count"] += 1
        where = self.stack[-1] if self.stack else "(top)"
        if len(r["where"]) < 3 and where not in r["where"]:
            r["where"].append(where)
        return out


def attach_stack(model: torch.nn.Module, stack: list[str]):
    """用模块的前后钩子维护「当前在哪个子模块里」，给每条记录标出处。
    ⚠ 钩子必须返回 None：forward hook 返回非 None 会顶替模块的输出（踩过：返回了 pop 出来的名字，CLIP 拿到一个字符串当隐状态）。"""

    def pre(mod, inp, n):
        stack.append(n)

    def post(mod, inp, out):
        stack.pop()

    hs = []
    for n, m in model.named_modules():
        if not n:
            continue
        hs.append(m.register_forward_pre_hook(lambda mod, inp, n=n: pre(mod, inp, n)))
        hs.append(m.register_forward_hook(post))
    return hs


def run(comp: str, model: torch.nn.Module, fn):
    stack: list[str] = []
    hs = attach_stack(model, stack)
    rec = Recorder(comp, stack)
    t = time.time()
    with torch.inference_mode(), rec:
        fn()
    secs = time.time() - t
    for h in hs:
        h.remove()
    return rec, secs


def summarize(comp: str, rec: Recorder, secs: float, note: str) -> str:
    rows = list(rec.records.values())
    total = sum(r["flops_each"] * r["count"] for r in rows) or 1
    by_op: "OrderedDict[str, dict]" = OrderedDict()
    for r in rows:
        a = by_op.setdefault(r["op"], {"sigs": 0, "calls": 0, "flops": 0, "elems": 0, "status": set()})
        a["sigs"] += 1
        a["calls"] += r["count"]
        a["flops"] += r["flops_each"] * r["count"]
        a["elems"] += (r["in_elems_each"] + r["out_elems_each"]) * r["count"]
        a["status"].add(r["status"])
    lines = [f"### {comp}", "", note, "",
             f"CPU fp32 跑一遍 {secs:.1f} s；{sum(r['count'] for r in rows)} 次计算类调用、{len(rows)} 种签名（= 要特化的 kernel 数的上限）；"
             f"另有 {rec.view_calls} 次纯视图调用不计；总计 {total / 1e9:.1f} GFLOPs。", "",
             "| 算子 | 签名数 | 调用数 | GFLOPs | 占比 | 读写元素（亿，未融合） | 对照 calBLAS |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for op, a in sorted(by_op.items(), key=lambda kv: (-kv[1]["flops"], -kv[1]["calls"])):
        lines.append(f"| `{op}` | {a['sigs']} | {a['calls']} | {a['flops'] / 1e9:.1f} | {100 * a['flops'] / total:.1f}% | "
                     f"{a['elems'] / 1e8:.2f} | {' / '.join(sorted(a['status']))} |")
    need = [r for r in rows if r["status"] in ("新写", "改参数", "待定")]
    if need:
        lines += ["", "要新写 / 改参数 / 待定的签名：", "", "| 算子 | 输入形状 | 关键参数 | 次数 | 状态 | 说明 | 出现在 |", "| --- | --- | --- | ---: | --- | --- | --- |"]
        for r in sorted(need, key=lambda r: (r["status"], r["op"])):
            lines.append(f"| `{r['op']}` | `{r['in_shapes']}` | `{json.dumps(r['key'], ensure_ascii=False, default=str)}` | {r['count']} | "
                         f"{r['status']} | {r['note']} | {', '.join(r['where'])} |")
    return "\n".join(lines) + "\n"


def main():
    model_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    torch.set_num_threads(min(16, torch.get_num_threads()))
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    sections = []
    meta = {"torch": torch.__version__, "model_dir": str(model_dir), "dtype": "float32", "latent": [1, 4, 64, 64], "image": [512, 512]}

    tok = CLIPTokenizer.from_pretrained(model_dir / "tokenizer")
    clip = CLIPTextModel.from_pretrained(model_dir / "text_encoder").eval()
    ids = tok(["a photo of an astronaut riding a horse on mars"], padding="max_length", max_length=tok.model_max_length,
              truncation=True, return_tensors="pt").input_ids
    rec, secs = run("clip", clip, lambda: clip(ids))
    sections.append(summarize("CLIP 文本编码器", rec, secs, "输入：一条 prompt，补到 77 个 token（SD 的做法）；CFG 时空 prompt 再跑一次。"))
    (out_dir / "clip.json").write_text(json.dumps(list(rec.records.values()), ensure_ascii=False, indent=1, default=str))
    ctx = clip(ids).last_hidden_state
    del clip

    unet = UNet2DConditionModel.from_pretrained(model_dir / "unet").eval()
    lat = torch.randn(1, 4, 64, 64)
    rec, secs = run("unet", unet, lambda: unet(lat, torch.tensor([999]), encoder_hidden_states=ctx))
    sections.append(summarize("UNet（一次前向，批 1）", rec, secs,
                              "输入：latent 1×4×64×64（512² 的图）、时间步 999、文本嵌入 1×77×768。CFG 每步要跑两路（批 2 或连调两次），FLOPs 翻倍。"))
    (out_dir / "unet.json").write_text(json.dumps(list(rec.records.values()), ensure_ascii=False, indent=1, default=str))
    del unet

    vae = AutoencoderKL.from_pretrained(model_dir / "vae").eval()
    rec, secs = run("vae", vae, lambda: vae.decode(lat / vae.config.scaling_factor))
    sections.append(summarize("VAE 解码器", rec, secs, "输入：latent 1×4×64×64 ⇒ 图 1×3×512×512。每张图只跑一次。"))
    (out_dir / "vae.json").write_text(json.dumps(list(rec.records.values()), ensure_ascii=False, indent=1, default=str))
    del vae

    # TAESD：B1 先用它把「NPU 解码 ⇒ ComfyUI 出图」打通（只有 3×3 卷积、ReLU、最近邻上采样与一个 tanh）
    taesd_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else model_dir.parent / "taesd"
    if taesd_dir.exists():
        from diffusers import AutoencoderTiny

        tiny = AutoencoderTiny.from_pretrained(taesd_dir).eval()
        rec, secs = run("taesd", tiny, lambda: tiny.decode(lat))
        sections.append(summarize("TAESD 解码器", rec, secs,
                                  f"输入：同一个 latent 1×4×64×64 ⇒ 图 1×3×512×512（`{taesd_dir}`）。B1 先用它，正式 VAE 随后。"))
        (out_dir / "taesd.json").write_text(json.dumps(list(rec.records.values()), ensure_ascii=False, indent=1, default=str))
        meta["taesd_dir"] = str(taesd_dir)

    head =["# SD1.5 算子清单（512×512）", "",
            "由 `ref/op_inventory.py` 生成（CPU、fp32、diffusers 的参照实现），不要手改；重出：见 README「CPU 参照」。", "",
            f"环境：torch {meta['torch']}；模型 `{model_dir}`（revision 见该目录的 download-verification.json）。", "",
            "「对照 calBLAS」一栏：现成 = 模板直接能用；改参数 = 模板能用但要补零、换掩码之类；新写 = 没有模板；不搬 = 在整步程序里靠排布消掉、不起 kernel；"
            "主机 = 放 CPU（要记进 NPU 贡献表）。FLOPs 只算卷积、线性层、注意力、矩阵乘；逐元素与归一化按「读写元素」看访存。", ""]
    (out_dir / "README.md").write_text("\n".join(head) + "\n" + "\n".join(sections))
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print("\n".join(sections))


if __name__ == "__main__":
    main()
