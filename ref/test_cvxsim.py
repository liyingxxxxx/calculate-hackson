"""模拟器（host/sd15npu/cvxsim.py）自己的测试：它说「接线检查全过」必须有分量 ⇒ 拿真编出来的 .cvx 故意改坏，每种坏法都要被抓到。

用法：`$CALCULET_HOME/venv/bin/python -I ref/test_cvxsim.py`（先编过 taesd、smoke 的设备遍；tools/verify_all.sh 会先编）。
用随机输入、随机权重（不读真权重，几秒跑完）：这里只查接线，数值对不对由 `tools/taesd.py check` 管。

| 改法 | 期望 |
| --- | --- |
| 原样 | 通过，输出全有限 |
| 删掉一条卷积之后的 `barrier.core.sync` | 「少一条 sync_cores()」 |
| 某个 kernel 的偏移挪 32 字节 | 「不在 64 B 上」 |
| 把一个卷积的输出指到它自己的输入 | 「不能原地」 |
| 去掉入口的清零 | 输出里有 NaN（读到了没写过的内存） |
| 上采样只让 0 号核调（smoke 改之前的写法） | 「按 num_cores() 分行」 |
| 屏障放进 `if core_rank() == 0` | 「挂死」 |
| 偏移越出声明 | 「越出」 |
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "host"))

import numpy as np  # noqa: E402

from sd15npu import cvxsim, taesd  # noqa: E402
from sd15npu.cvxsim import SimError  # noqa: E402

NPU = ROOT / "target" / "dopamine-unknown-none-cvx" / "npu"


def _taesd_inputs(L):
    rng = np.random.default_rng(0)
    lat = rng.standard_normal((4, 64, 64)).astype(np.float32)
    w = bytearray(L["consts"]["W_BYTES"])
    for c in L["convs"]:
        t = (rng.standard_normal(c["w_len"] // 2) * 0.05).astype(np.float32)
        w[c["w_off"]:c["w_off"] + c["w_len"]] = taesd.bf16.bits(t).tobytes()
        w[c["b_off"]:c["b_off"] + c["b_len"]] = (rng.standard_normal(c["b_len"] // 4) * 0.1).astype("<f4").tobytes()
    return {"in_x": taesd.pack_latent(L, lat), "w": bytes(w)}


def _run(text, kernels_rs, consts, inputs):
    return cvxsim.Sim(text, kernels_rs, consts, inputs).run()


def _expect(name, text, kernels_rs, consts, inputs, needle):
    try:
        sim = _run(text, kernels_rs, consts, inputs)
    except SimError as e:
        assert needle in str(e), f"{name}：报的错不对：{e}"
        print(f"✓ {name}：抓到了（{str(e)[:90]}…）")
        return
    if needle == "NaN":
        out = sim.mem["out_y"].view("<u2")
        nan = np.isnan(taesd.bf16.from_bits(out)).sum()
        # out_y 的边框本来就没人写（主机不读），只看内部
        L = taesd.layout()["consts"]
        n, b, co = 8 * L["LAT"], L["B"], L["COUT"]
        inner = taesd.bf16.from_bits(out[:L["OUT_BYTES"] // 2].reshape(n + 2 * b, n + 2 * b, co)[b:b + n, b:b + n, :3])
        assert np.isnan(inner).any(), f"{name}：该有 NaN 却没有"
        print(f"✓ {name}：输出内部有 {int(np.isnan(inner).sum())} 个 NaN（读到了没清零的边框）")
        return
    raise AssertionError(f"{name}：该报「{needle}」却通过了")


def main():
    L = taesd.layout()
    consts = L["consts"]
    text = (NPU / "taesd.cvx").read_text()
    krs = (ROOT / "crates/taesd/src/kernels.rs").read_text()
    inputs = _taesd_inputs(L)

    sim = _run(text, krs, consts, inputs)
    inner = taesd.unpack_image(L, sim.mem["out_y"].tobytes())
    assert np.isfinite(inner).all(), "原样的程序输出里有 NaN / Inf"
    print(f"✓ 原样：{sim.report()}")

    # 1. 删掉第一个卷积（td_conv_in）之后的屏障
    m = re.search(r"call[^\n]*@td_conv_in\n\s*barrier\.core\.sync\n", text)
    bad = text[:m.start()] + m.group(0).split("\n")[0] + "\n" + text[m.end():]
    _expect("删屏障", bad, krs, consts, inputs, "少一条 sync_cores()")

    # 2. 某个偏移挪 32 字节：64² 的 a 图（MAPS[0][0]）第一次出现的那个常量
    base = text.index(".func step()")
    m = re.compile(rf"s\.constant\s+\w+ = ({L['maps'][0][0]}):index").search(text, base)
    a, b = m.start(1), m.end(1)
    bad = text[:a] + str(int(m.group(1)) + 32) + text[b:]
    _expect("偏移不对齐", bad, krs, consts, inputs, "不在 64 B 上")

    # 3. 原地：td_conv_in 的输出寄存器换成它的输入寄存器
    m = re.search(r"call\s+(\w+), (\w+), (\w+), (\w+), @td_conv_in", text)
    bad = text[:m.start()] + f"call         {m.group(1)}, {m.group(2)}, {m.group(3)}, {m.group(1)}, @td_conv_in" + text[m.end():]
    _expect("原地", bad, krs, consts, inputs, "不能原地")

    # 4. 去掉清零：两条 td_zero 的 call 删掉
    bad = re.sub(r"\n\s*call[^\n]*@td_zero", "", text)
    _expect("不清零", bad, krs, consts, inputs, "NaN")

    # 5. 屏障放进 if core_rank() == 0：把第一条 `if rX` 区里的 yield 换成屏障 + yield
    body = text.index(".func step()")
    i = text.index("            yield\n        .end", body)
    bad = text[:i] + "            barrier.core.sync\n" + text[i:]
    _expect("屏障只在一颗核上", bad, krs, consts, inputs, "挂死")

    # 6. 越界：最后一个卷积的输出指到 @out_y 之后
    m = re.search(r"mem\.addr\s+(\w+) = @out_y\n", text[body:])
    reg = m.group(1)
    j = body + m.end()
    bad = text[:j] + f"        s.constant   r99990 = {consts['OUT_BYTES']}:index\n        s.ptradd     r99991 = {reg}, r99990\n" + text[j:]
    bad = re.sub(rf"(call\s+[^\n]*), {reg}, @td_conv_out", r"\1, r99991, @td_conv_out", bad)
    _expect("越界", bad, krs, consts, inputs, "越出")

    # 7. smoke：现在两核都调上采样 ⇒ 过；改回「只让 0 号核调」⇒ 抓到
    st = (NPU / "smoke.cvx").read_text()
    srs = (ROOT / "crates/smoke/src/lib.rs").read_text()
    sin = {"in_x": bytes(591872), "w": bytes(77824)}
    s_ok = _run(st, srs, {}, sin)
    print(f"✓ smoke 原样：{s_ok.report()}")
    m = re.search(r"\n(\s*)(call\s+[^\n]*@smoke_up2_64x64_c64)\n", st)
    ind = m.group(1)
    core0 = (f"\n{ind}sys.query    r99980 = core_rank\n{ind}s.constant   r99981 = 0:index\n{ind}s.cmp        r99982 = r99980, r99981, eq\n"
             f"{ind}if           r99982\n{ind}    {m.group(2)}\n{ind}    yield\n{ind}.end\n{ind}    yield\n{ind}.end\n")
    bad = st[:m.start()] + core0 + st[m.end():]
    _expect("上采样只在 0 号核", bad, srs, {}, sin, "按 num_cores() 分行")
    print("模拟器的自测全过")


if __name__ == "__main__":
    main()
