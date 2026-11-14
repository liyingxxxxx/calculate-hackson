"""生成汇报 PPT：SD1.5 在 Calculet NPU 上的后端模型适配（数字取自仓库 test/results/npu3-20261114/ 与 test/PERF.md）。

运行（python-pptx 不是交付件的依赖，装在单独的 venv 里，不碰项目环境）：
  python3 -m venv /tmp/pptx-venv && /tmp/pptx-venv/bin/pip install python-pptx
  /tmp/pptx-venv/bin/python docs/report/make_ppt.py   ⇒  docs/report/sd15-npu-汇报.pptx
数字改了（新的真卡结果）就改这里再重出；PPT 里的图表是原生图表，也可以直接在 PowerPoint 里改。
"""
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

OUT = Path(__file__).with_name("sd15-npu-汇报.pptx")

# ── 配色与字体（同 HTML 报告：芯片封装的冷灰 + 示波器青，CPU 对照用铜色）──
INK = RGBColor(0x15, 0x20, 0x2B)
MUTED = RGBColor(0x58, 0x66, 0x72)
LINE = RGBColor(0xD9, 0xE0, 0xE4)
BG_SOFT = RGBColor(0xF3, 0xF6, 0xF7)
ACCENT = RGBColor(0x0D, 0x6C, 0x8C)
ACCENT_2 = RGBColor(0x3B, 0x92, 0xB0)
ACCENT_3 = RGBColor(0x7C, 0xB8, 0xCB)
ACCENT_4 = RGBColor(0xB9, 0xD7, 0xE0)
ACCENT_SOFT = RGBColor(0xE2, 0xF0, 0xF4)
CPU = RGBColor(0x9A, 0x63, 0x2A)
DEPLOY = RGBColor(0x8E, 0x7C, 0x63)
HOST = RGBColor(0xCB, 0xC3, 0xB5)
GOOD = RGBColor(0x2D, 0x7A, 0x4C)
WARN = RGBColor(0xA2, 0x6A, 0x14)
BAD = RGBColor(0xAD, 0x3E, 0x3A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Microsoft YaHei"
MONO = "Consolas"

W, H = 13.333, 7.5
ML = 0.6
CW = W - 2 * ML

prs = Presentation()
prs.slide_width = Inches(W)
prs.slide_height = Inches(H)
BLANK = prs.slide_layouts[6]
TOTAL = 15


def _font(run, size, color=INK, bold=False, name=FONT):
    f = run.font
    f.size = Pt(size)
    f.bold = bold
    f.color.rgb = color
    f.name = name
    rpr = run._r.get_or_add_rPr()
    ea = rpr.find(qn("a:ea"))
    if ea is None:
        ea = etree.SubElement(rpr, qn("a:ea"))
    ea.set("typeface", FONT)


def text(slide, x, y, w, h, paras, size=14, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=4, line=1.15):
    """paras：字符串，或段落列表；每段是字符串或 [(文字, {size, color, bold, mono})] 片段列表。"""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.02)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    tf.vertical_anchor = anchor
    if isinstance(paras, str):
        paras = [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(spacing)
        p.line_spacing = line
        parts = [(para, {})] if isinstance(para, str) else para
        for t, o in parts:
            r = p.add_run()
            r.text = t
            _font(r, o.get("size", size), o.get("color", color), o.get("bold", bold), MONO if o.get("mono") else FONT)
    return tb


def box(slide, x, y, w, h, fill=None, line=None, shape=MSO_SHAPE.RECTANGLE, radius=None, lw=0.75):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(lw)
    s.shadow.inherit = False
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    s.text_frame.margin_left = s.text_frame.margin_right = Inches(0.08)
    return s


def box_text(s, paras, size=13, color=INK, bold=False, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE):
    tf = s.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    if isinstance(paras, str):
        paras = [paras]
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        parts = [(para, {})] if isinstance(para, str) else para
        for t, o in parts:
            r = p.add_run()
            r.text = t
            _font(r, o.get("size", size), o.get("color", color), o.get("bold", bold), MONO if o.get("mono") else FONT)


def arrow(slide, x, y, w, h=0.32, color=MUTED):
    s = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = color
    s.line.fill.background()
    s.shadow.inherit = False
    return s


def header(slide, n, eyebrow, title, note=None):
    text(slide, ML, 0.42, CW, 0.3, eyebrow, size=11, color=ACCENT, bold=True)
    text(slide, ML, 0.68, CW, 0.7, title, size=26, bold=True)
    foot = "sd15-npu · 2026-11-14 · npu3 真卡（/dev/calculet0）" + (f" · {note}" if note else "")
    text(slide, ML, 7.02, 10.5, 0.3, foot, size=9, color=MUTED)
    text(slide, W - ML - 1.0, 7.02, 1.0, 0.3, f"{n} / {TOTAL}", size=9, color=MUTED, align=PP_ALIGN.RIGHT)


def table(slide, x, y, widths, rows, header_fill=BG_SOFT, size=11.5, row_h=0.42, highlight=(), aligns=None, bold_first_col=True):
    """rows[0] 是表头；单元格是字符串或片段列表。只画横线（单元格下边框），不画竖线。"""
    nr, nc = len(rows), len(widths)
    gf = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(sum(widths)), Inches(row_h * nr))
    tbl = gf.table
    tblpr = gf._element.graphic.graphicData.tbl.tblPr
    tblpr.set("firstRow", "1")
    tblpr.set("bandRow", "0")
    sid = tblpr.find(qn("a:tableStyleId"))
    if sid is None:
        sid = etree.SubElement(tblpr, qn("a:tableStyleId"))
    sid.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"  # No Style, No Grid
    for j, wv in enumerate(widths):
        tbl.columns[j].width = Inches(wv)
    for i in range(nr):
        tbl.rows[i].height = Inches(row_h)
        for j in range(nc):
            c = tbl.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.08)
            c.margin_top = c.margin_bottom = Inches(0.04)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            fill = header_fill if i == 0 else (ACCENT_SOFT if i in highlight else WHITE)
            c.fill.solid()
            c.fill.fore_color.rgb = fill
            tf = c.text_frame
            tf.word_wrap = True
            val = rows[i][j]
            parts = [(val, {})] if isinstance(val, str) else val
            p = tf.paragraphs[0]
            p.alignment = (aligns[j] if aligns else PP_ALIGN.LEFT)
            for t, o in parts:
                if t == "\n":
                    p = tf.add_paragraph()
                    p.alignment = (aligns[j] if aligns else PP_ALIGN.LEFT)
                    continue
                r = p.add_run()
                r.text = t
                hdr = i == 0
                _font(r, o.get("size", size - (0.5 if hdr else 0)), o.get("color", MUTED if hdr else INK),
                      o.get("bold", hdr or (bold_first_col and j == 0)), MONO if o.get("mono") else FONT)
            # 下边框：浅灰细线
            tcpr = c._tc.get_or_add_tcPr()
            for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
                ln = etree.SubElement(tcpr, qn(tag))
                if tag == "a:lnB":
                    ln.set("w", str(Emu(Pt(0.75))))
                    sf = etree.SubElement(ln, qn("a:solidFill"))
                    clr = etree.SubElement(sf, qn("a:srgbClr"))
                    clr.set("val", "D9E0E4")
                else:
                    ln.set("w", "0")
                    etree.SubElement(ln, qn("a:noFill"))
            # tcPr 里边框要在填充之前：把填充挪到最后
            for f in tcpr.findall(qn("a:solidFill")):
                tcpr.remove(f)
                tcpr.append(f)
    return gf


def pill(slide, x, y, label, color, w=None):
    w = w or (0.22 + 0.17 * len(label))
    s = box(slide, x, y, w, 0.3, fill=None, line=color, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.5, lw=1)
    box_text(s, label, size=10.5, color=color, bold=True)
    return s


def metric(slide, x, y, w, value, label, sub=None, color=ACCENT):
    text(slide, x, y, w, 0.6, value, size=30, bold=True, color=color)
    text(slide, x, y + 0.62, w, 0.35, label, size=13, bold=True)
    if sub:
        text(slide, x, y + 0.95, w, 0.6, sub, size=10.5, color=MUTED)


def bar_chart(slide, x, y, w, h, cats, vals, colors, maximum, unit="s", fmt='0.00" s"', major=None, label_size=11):
    cd = CategoryChartData()
    cd.categories = cats
    cd.add_series("耗时", vals)
    gf = slide.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(x), Inches(y), Inches(w), Inches(h), cd)
    ch = gf.chart
    ch.has_legend = False
    ch.font.size = Pt(label_size)
    ch.font.name = FONT
    ch.font.color.rgb = INK
    plot = ch.plots[0]
    plot.gap_width = 45
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = fmt
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(label_size)
    dl.font.bold = True
    ser = plot.series[0]
    for i, c in enumerate(colors):
        pt = ser.points[i]
        pt.format.fill.solid()
        pt.format.fill.fore_color.rgb = c
        pt.format.line.fill.background()
    ca = ch.category_axis
    ca.reverse_order = True
    ca.format.line.color.rgb = LINE
    ca.tick_labels.font.size = Pt(label_size)
    ca.has_major_gridlines = False
    va = ch.value_axis
    va.maximum_scale = maximum
    va.minimum_scale = 0
    if major:
        va.major_unit = major
    va.has_major_gridlines = True
    va.major_gridlines.format.line.color.rgb = LINE
    va.format.line.fill.background()
    va.tick_labels.font.size = Pt(label_size - 1)
    va.tick_labels.font.color.rgb = MUTED
    va.tick_labels.number_format = '0" ' + unit + '"'
    va.tick_labels.number_format_is_linked = False
    return gf


# ════════════════════════ 1 封面 ════════════════════════
s = prs.slides.add_slide(BLANK)
box(s, 0, 0, W, H, fill=WHITE)
box(s, 0, 0, 0.18, H, fill=ACCENT)
text(s, 0.9, 1.2, 11, 0.4, "Calculet NPU CalVX 黑客松 · 路线 B（ComfyUI 原生工作流）", size=14, color=ACCENT, bold=True)
text(s, 0.9, 1.75, 11.5, 1.6, ["Stable Diffusion 1.5", "在 Calculet NPU 上的后端模型适配"], size=40, bold=True, line=1.05, spacing=2)
text(s, 0.9, 3.55, 11, 0.4, "阶段汇报 · 2026-11-14 · npu3 真卡实测 · 只用赛方公开工具链（docc / dort 0.4.1）", size=15, color=MUTED)
ys = 4.75
box(s, 0.9, ys - 0.2, W - 1.8, 0.012, fill=LINE)
metric(s, 0.9, ys, 3.7, "0.98 s", "VAE 解码一张 512×512", "比同机 CPU 快 1.75 倍，已接入 ComfyUI")
metric(s, 4.9, ys, 3.7, "4.6×", "UNet 一步提速", "12.35 s → 2.67 s，五轮优化，精度不变")
metric(s, 8.9, ys, 3.6, "≈ 50 dB", "全部真卡结果对参照", "多次运行逐字节一致，可从仓库复现")

# ════════════════════════ 2 一页结论 ════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, 2, "一页结论", "解码器已在 NPU 上跑赢 CPU；UNet 已上 NPU 并算对，还在提速")
rows3 = [
    ("已完成", GOOD, "TAESD、正式 VAE 解码器", "真卡跑通、比同机 CPU 快 1.6 / 1.75 倍，已接入 ComfyUI 原生工作流，出图与 ComfyUI 自带 VAEDecode 一致（55.6 dB）。"),
    ("进行中", WARN, "UNet（去噪网络）", "全部算子已上 NPU、真卡结果对参照约 50 dB；单步 12.35 s → 2.67 s，但仍比同机 CPU（1.41 s）慢约 1.9 倍，尚未接入 ComfyUI 采样器。"),
    ("未开始", BAD, "CLIP 文本编码、去噪循环进设备", "工作流里 CLIP 和采样循环目前由 ComfyUI 在 CPU 上执行。"),
]
y = 1.75
for lab, col, mod, desc in rows3:
    box(s, ML, y, CW, 1.35, fill=WHITE, line=LINE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    pill(s, ML + 0.3, y + 0.25, lab, col, w=0.95)
    text(s, ML + 1.5, y + 0.2, 4.0, 0.45, mod, size=17, bold=True)
    text(s, ML + 1.5, y + 0.66, 10.3, 0.65, desc, size=13, color=MUTED)
    y += 1.55
text(s, ML, 6.45, CW, 0.4, [[("现在可演示：", {"bold": True}), ("ComfyUI 原生工作流「提示词 → 出图」，其中 VAE 解码在 NPU 上（0.99 s），CLIP 与 20 步 UNet 在 CPU 上（约 49 s）。", {"color": MUTED})]], size=13)

# ════════════════════════ 3 适配范围与路线 ════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, 3, "适配范围", "SD1.5 文生图的四个模型，按 ComfyUI 原生工作流接入")
stages = [
    ("提示词", "文本", None, None),
    ("CLIP", "文本编码", "CPU", BAD),
    ("UNet × 20 步", "去噪（CFG 两路）", "NPU 已算对\n未接采样器", WARN),
    ("VAE / TAESD", "latent → 图像", "NPU\n已接入", GOOD),
    ("图像", "512 × 512", None, None),
]
bx, bw, gap = ML, 2.05, 0.45
for i, (name, sub, st, col) in enumerate(stages):
    x = bx + i * (bw + gap)
    edge = col if col else LINE
    sh = box(s, x, 2.0, bw, 1.25, fill=WHITE if col else BG_SOFT, line=edge, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08, lw=1.5 if col else 0.75)
    box_text(sh, [[(name, {"bold": True, "size": 16})], [(sub, {"size": 11.5, "color": MUTED})]])
    if st:
        text(s, x, 3.38, bw, 0.6, st.split("\n"), size=11.5, color=col, bold=True, align=PP_ALIGN.CENTER, spacing=0)
    if i < len(stages) - 1:
        arrow(s, x + bw + 0.08, 2.48, gap - 0.16, 0.28, LINE)
text(s, ML, 4.4, CW, 0.4, "适配约束与口径", size=15, bold=True)
cons = [
    [("工具链：", {"bold": True}), ("只用赛方公开栈 rustc 1.98.1-calculet、docc 0.4.1、dort 0.4.1，基础库经赛方 cargo 源依赖 calBLAS / calDNN；不用机器上的老版本，不引用未公开资料。", {})],
    [("数值：", {"bold": True}), ("激活 bf16、矩阵乘在 TU 里 f32 累加；每个模型都对两份独立参照（按设备口径舍 bf16 的 torch 参照、diffusers fp32）。", {})],
    [("形态：", {"bold": True}), ("每个模型编成整步设备程序（calbin），由 dort 在真卡上执行；代码超过每核 1 MB 上限的按段切开、主机串起来。", {})],
    [("档位：", {"bold": True}), ("先做 512 × 512（latent 64 × 64）；ComfyUI 节点三种后端：npu（真卡）/ npu-sim（CPU 上模拟，不算 NPU 贡献）/ cpu（兜底，计数）。", {})],
]
text(s, ML, 4.85, CW, 2.0, cons, size=12.5, color=INK, spacing=6)

# ════════════════════════ 4 架构与组件 ════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, 4, "系统架构", "四层组件：ComfyUI 节点 → 主机执行器 → 设备程序 → NPU 算子")
layers = [
    ("ComfyUI 层", "comfy/sd15_npu", "自定义节点 SD15 NPU · TAESD Decode / VAE Decode；后端 npu / npu-sim / cpu；工作流 JSON"),
    ("主机执行层", "host/sd15npu", "权重打包（bf16、按头补零、掩码）、输入输出打包、段间数据按偏移转手、dort 封装（缺省拒绝上板）"),
    ("设备程序层", "crates/taesd · vae · unet", "整步程序：TAESD 1 段、VAE 3 段、UNet 6 段；UNet 由生成器从 diffusers 结构自动产出"),
    ("算子层", "crates/kernels + calBLAS", "自研十余类算子（注意力、矩阵乘、NHWC 归一化……）+ calBLAS 模板（卷积、上采样、线性层）"),
    ("工具链与硬件", "docc / dort 0.4.1", "CalVX ⇒ calbin ⇒ 真卡（npu3，2 芯粒；程序单芯粒两核）"),
]
ly = 1.7
for i, (name, path, desc) in enumerate(layers):
    fill = ACCENT if i == 4 else (ACCENT_SOFT if i % 2 == 0 else WHITE)
    tc = WHITE if i == 4 else INK
    sh = box(s, ML, ly, 8.3, 0.88, fill=fill, line=LINE if i != 4 else None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
    text(s, ML + 0.2, ly + 0.1, 2.1, 0.35, name, size=14, bold=True, color=tc)
    text(s, ML + 0.2, ly + 0.47, 2.1, 0.35, path, size=10, color=WHITE if i == 4 else ACCENT, bold=False)
    text(s, ML + 2.4, ly + 0.12, 5.75, 0.7, desc, size=11.5, color=WHITE if i == 4 else MUTED, anchor=MSO_ANCHOR.MIDDLE)
    ly += 1.0
dx = ML + 8.65
box(s, dx, 1.7, CW - 8.65, 4.88, fill=BG_SOFT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.04)
text(s, dx + 0.25, 1.85, 3.2, 0.35, "研发与验证工具", size=14, bold=True)
tools = [
    ("unet_gen.py", "生成算子特化与算子序列、按代码量自动切段"),
    ("prof.py", "真卡逐算子计时，定位瓶颈"),
    ("cvxsim.py", "读编译产物在 numpy 里重放，查接线"),
    ("ref/", "独立 CPU 参照（bf16 口径 / fp32）"),
    ("board_check.py", "上板一条命令，卡不满足就停"),
    ("verify_all.sh", "14 项不上板自检，提交前必跑"),
]
ty = 2.3
for nm, d in tools:
    text(s, dx + 0.25, ty, 3.2, 0.3, nm, size=12, bold=True, color=ACCENT)
    text(s, dx + 0.25, ty + 0.28, 3.2, 0.4, d, size=10.5, color=MUTED)
    ty += 0.7

# ════════════════════════ 5 指标总表 ════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, 5, "核心指标", "各模型在真卡上的正确性与速度", note="NPU 为 dort 整次墙钟（含部署）；CPU 为同机 diffusers fp32 实测")
C = PP_ALIGN.CENTER
R = PP_ALIGN.RIGHT
L = PP_ALIGN.LEFT
rows = [
    ["模型", "设备程序", "正确性（真卡，对独立参照）", "NPU 耗时", "同机 CPU", "NPU / CPU", "ComfyUI"],
    ["TAESD 解码", "1 段", "52.7–65.8 dB（4 组 latent）", "0.113 s", "0.185 s", [("快 1.6×", {"color": GOOD, "bold": True})], [("已接入", {"color": GOOD, "bold": True})]],
    ["正式 VAE 解码", "3 段", "58.3 dB；出图与自带 VAEDecode 比 55.6 dB", "0.98 s", "1.71 s", [("快 1.75×", {"color": GOOD, "bold": True})], [("已接入", {"color": GOOD, "bold": True})]],
    ["UNet 一步（CFG 两路）", "6 段", "49.7 / 50.0 dB（有 / 无提示词）", "2.67 s", "1.41 s", [("慢 1.9×", {"color": BAD, "bold": True})], [("未接入", {"color": BAD, "bold": True})]],
    ["CLIP 文本编码", "—", "未开始（在 CPU 上）", "—", "—", "—", [("原生节点", {"color": WARN, "bold": True})]],
]
table(s, ML, 1.75, [2.3, 1.1, 3.95, 1.25, 1.25, 1.2, 1.08], rows, size=12.5, row_h=0.62, highlight=(3,), aligns=[L, C, L, R, R, C, C])
text(s, ML, 5.05, CW, 0.35, "其他可汇报的质量指标", size=15, bold=True)
q = [
    ("确定性", "每个模型多次运行，输出逐字节一致"),
    ("可复现", "从 GitHub 重新克隆、从零重建后上卡，TAESD 出图与第一次逐字节相同"),
    ("自检", "14 项不上板检查全过；模拟器 7 种坏法全抓到"),
    ("单算子", "7 个算子 × 20 个边界用例上卡对期望；新注意力单独上卡对 numpy 0.6–1.5%"),
]
qy = 5.5
for i, (k, v) in enumerate(q):
    x = ML + (i % 2) * 6.1
    yy = qy + (i // 2) * 0.62
    text(s, x, yy, 6.0, 0.6, [[(k + "  ", {"bold": True, "color": ACCENT}), (v, {"color": INK})]], size=12)

# ════════════════════════ 6 UNet 优化历程 ════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, 6, "UNet 性能", "五轮优化：单步 12.35 s → 2.67 s（4.6 倍），精度始终约 50 dB", note="每步 = 一次 UNet 前向（CFG 两路）")
cats = ["第一版（20 段）", "① 自写注意力", "② 宽通道卷积→矩阵乘", "③ 稠密 token", "④ 128 行 q", "⑤ 自写矩阵乘（6 段）", "同机 CPU fp32"]
vals = [12.35, 8.19, 5.16, 4.18, 3.12, 2.67, 1.41]
cols = [ACCENT_3, ACCENT_3, ACCENT_3, ACCENT_3, ACCENT_3, ACCENT, CPU]
bar_chart(s, ML - 0.1, 1.6, 7.6, 5.25, cats, vals, cols, maximum=14, major=2, label_size=11)
ox = ML + 7.8
steps = [
    ("①", "自写注意力 flash_attn!", "一块 key / value 给多行 query 共用；64² 自注意力 625 → 259 ms"),
    ("②", "宽通道卷积改走矩阵乘", "Cin > 1024 的卷积原来只有 0.1–0.2 TFLOPS；3×3 拆成 9 个抽头"),
    ("③", "自注意力去掉边框 token", "64² 少算约 25% 的分数、不再要掩码；缩放乘进 q 权重"),
    ("④", "一次处理 128 行 query", "瓶颈是每块 K / V 的搬运；64² 自注意力 173 → 84 ms"),
    ("⑤", "自写矩阵乘 gemm_rows!", "一块权重给 4 块数据用：1.0–1.5 → 2.4–4 TFLOPS"),
]
sy = 1.7
for n_, t_, d_ in steps:
    text(s, ox, sy, 0.4, 0.4, n_, size=15, bold=True, color=ACCENT)
    text(s, ox + 0.4, sy, 4.3, 0.35, t_, size=13, bold=True)
    text(s, ox + 0.4, sy + 0.34, 4.3, 0.6, d_, size=10.5, color=MUTED)
    sy += 1.0

# ════════════════════════ 7 UNet 一步时间构成 ════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, 7, "UNet 性能", "现在一步 2.67 s 花在哪", note="算子部分为真卡逐算子计时（单独连调 4 遍、扣固定开销）")
segs = [
    ("注意力", 1.041, ACCENT, WHITE, "其中 64² 自注意力 0.84 s"),
    ("卷积", 0.495, ACCENT_2, WHITE, "64² / 32² 快路卷积"),
    ("矩阵乘", 0.303, ACCENT_3, INK, "gemm_rows!，2.4–4 TFLOPS"),
    ("其他算子", 0.282, ACCENT_4, INK, "GEGLU、LayerNorm、GroupNorm 等"),
    ("部署权重", 0.50, DEPLOY, WHITE, "每次调用重新部署 1.62 GB"),
    ("主机转手", 0.05, HOST, INK, "段间数据经主机"),
]
total = sum(v for _, v, *_ in segs)
bx0, bw0, by0 = ML, CW, 2.0
x = bx0
for name, v, col, tc, _ in segs:
    ww = bw0 * v / total
    sh = box(s, x, by0, ww, 0.9, fill=col)
    if ww > 0.9:
        box_text(sh, [[(name, {"bold": True, "size": 12, "color": tc})], [(f"{v:.2f} s", {"size": 12, "color": tc, "mono": True})]])
    x += ww
text(s, bx0, by0 + 0.95, 2, 0.3, "0 s", size=10, color=MUTED, align=PP_ALIGN.LEFT)
text(s, bx0 + bw0 - 2, by0 + 0.95, 2, 0.3, "2.67 s", size=10, color=MUTED, align=PP_ALIGN.RIGHT)
rows = [["部分", "每步", "占比", "说明"]]
for name, v, col, tc, d in segs:
    rows.append([name, f"{v:.2f} s", f"{v / total * 100:.0f}%", d])
table(s, ML, 3.55, [2.0, 1.3, 1.1, 5.0], rows, size=12, row_h=0.42, aligns=[L, R, R, L])
ix = ML + 9.75
box(s, ix, 3.55, CW - 9.75, 2.95, fill=BG_SOFT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
text(s, ix + 0.2, 3.7, CW - 10.15, 2.7, [
    [("算子合计 2.12 s", {"bold": True})],
    [("注意力仍是最大的一项；64² 自注意力 0.41 TFLOPS，还有空间。", {"color": MUTED, "size": 11.5})],
    [("部署 0.50 s 去不掉", {"bold": True})],
    [("dort 0.4.1 没有常驻会话，每次调用都重新部署权重；CFG 两路已放在同一次调用里共用。", {"color": MUTED, "size": 11.5})],
], size=12.5, spacing=6)

# ════════════════════════ 8 解码器 ════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, 8, "解码器", "TAESD 与正式 VAE：已在 NPU 上跑赢 CPU，并接入 ComfyUI")
bar_chart(s, ML - 0.1, 1.65, 6.4, 2.6, ["VAE 第一版（6 段）", "VAE 优化后（3 段）", "VAE 同机 CPU"], [5.18, 0.98, 1.71],
          [ACCENT_3, ACCENT, CPU], maximum=6, major=1, label_size=11)
bar_chart(s, ML - 0.1, 4.35, 6.4, 2.0, ["TAESD（NPU）", "TAESD 同机 CPU"], [0.113, 0.185], [ACCENT, CPU], maximum=0.25, major=0.05,
          fmt='0.000" s"', label_size=11)
rx = ML + 6.7
text(s, rx, 1.7, 5.4, 0.35, "正式 VAE 解码器", size=16, bold=True)
text(s, rx, 2.1, 5.4, 2.2, [
    "第一版 6 段程序算对了，但要 5.18 s、比 CPU 慢。按真卡逐算子计时做了三件事：",
    [("· 卷积加列块快路：", {"bold": True}), ("大图卷积从 0.24–1 提到 4.2–4.6 TFLOPS", {})],
    [("· GroupNorm 改在 NHWC 上直接算：", {"bold": True}), ("去掉 1.3 GB/s 的排布转换", {})],
    [("· 代码变小后并成 3 段：", {"bold": True}), ("大特征图不再经主机转手", {})],
    [("结果 0.98 s，比 CPU 快 1.75 倍；精度 58.3 dB 不变；ComfyUI 节点 5.89 → 0.99 s。", {"bold": True, "color": ACCENT})],
], size=12, spacing=5)
text(s, rx, 4.45, 5.4, 0.35, "TAESD 解码器", size=16, bold=True)
text(s, rx, 4.85, 5.4, 1.8, [
    "整个解码器一个设备程序；两次上卡共 15 遍，中位数 0.113 s，比 CPU 快 1.6 倍。",
    "四组 latent 对参照 52.7–65.8 dB；从 GitHub 重新克隆重建后，出图与第一次逐字节相同。",
], size=12, spacing=6)

# ════════════════════════ 9 自研算子 ════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, 9, "组件 · 算子", "自研 NPU 算子（crates/kernels）", note="效果数字为真卡实测")
rows = [
    ["算子", "作用", "效果 / 指标"],
    ["flash_attn!", "注意力：一块 K / V 给 64 或 128 行 q 共用，数值同 calBLAS", "64² 自注意力 625 → 84 ms（7.4×）；32² 58 → 7.3 ms"],
    ["gemm_rows!", "矩阵乘：一块权重读一次给 4 块 16 行用，数值与 calBLAS 逐位相同", "1.0–1.5 → 2.4–4 TFLOPS；UNet 矩阵乘 0.79 → 0.30 s / 步"],
    ["conv2d! 列块快路", "大图卷积按列分块走快路（在 calBLAS 卷积上改）", "VAE 大图卷积 4.2–4.6 TFLOPS"],
    ["gn_stats! / gn_apply!", "NHWC 原生 GroupNorm（+ SiLU），支持 320–2560 通道", "去掉排布转换；代码 292 KB → 约 25 KB"],
    ["ln_nhwc! · geglu!", "Transformer 的 LayerNorm、GEGLU 门控", "UNet 全部 16 个 Transformer 块在 NPU 上"],
    ["timestep_embed! · bias_fold!", "时间步嵌入（支持小数时间步）；22 个 ResNet 的时间步投影一次折进卷积 bias", "时间步嵌入 5 个用例上卡对 diffusers"],
    ["res_add! · copy_win!", "残差加、跳连拼接（只写图的内部，边框保持 0）", "让宽通道卷积能改走矩阵乘"],
    ["pack_int! / unpack_int!", "带边框的图 ⇔ 稠密 token", "64² 自注意力少算约 25% 的分数"],
    ["attn_wide!", "VAE 中间块的宽头注意力（头维 512）", "calBLAS 注意力头维 512 编不出来时的替代"],
]
table(s, ML, 1.7, [2.65, 5.35, 4.13], rows, size=11.5, row_h=0.5, highlight=(1, 2), aligns=[L, L, L])

# ════════════════════════ 10 研发工具组件 ════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, 10, "组件 · 工具", "研发与验证工具：让大模型适配可生成、可测量、可复现")
cards = [
    ("UNet 生成器", "tools/unet_gen.py", ["从 diffusers 的模块结构生成 151 个算子特化、每路 858 次算子调用", "按实测代码量自动切段（每核 ≤ 1 MB），偏移只算一份，设备与主机共用"]),
    ("逐算子计时", "tools/prof.py", ["每个算子单独一个程序在卡上连调 4 遍、扣掉固定开销", "每轮优化都按「单次耗时 × 调用次数」找最大瓶颈"]),
    ("设备程序模拟器", "host/sd15npu/cvxsim.py", ["读编译产物，在 numpy 里按真实调用序列重放", "查对齐、越界、屏障、漏清零、原地读写；7 种坏法自测全抓到"]),
    ("独立参照", "ref/", ["按 torch 语义另写，不照着算子写", "两份：按设备口径舍 bf16 的参照、diffusers fp32"]),
    ("上板检查", "tools/board_check.py", ["静态核对 → 只读查卡 → 单算子 → 整模型，任一步不对就停", "缺省拒绝上板，需显式开关；不刷写、不复位"]),
    ("一键自检与重建", "verify_all.sh · bootstrap.sh", ["不上板能做的 14 项检查，提交前必跑", "新机器从零一键重建；仓库已公开在 GitHub"]),
]
cw_, ch_ = 3.9, 2.45
for i, (t_, p_, lines) in enumerate(cards):
    x = ML + (i % 3) * (cw_ + 0.215)
    y = 1.75 + (i // 3) * (ch_ + 0.25)
    box(s, x, y, cw_, ch_, fill=WHITE, line=LINE, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    text(s, x + 0.25, y + 0.2, cw_ - 0.5, 0.35, t_, size=15, bold=True)
    text(s, x + 0.25, y + 0.56, cw_ - 0.5, 0.3, [[(p_, {"mono": True, "color": ACCENT, "size": 10.5})]])
    text(s, x + 0.25, y + 0.95, cw_ - 0.5, 1.45, ["· " + l for l in lines], size=11, color=MUTED, spacing=5)

# ════════════════════════ 11 质量保证流程 ════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, 11, "方法", "每一处改动走同一条验证链，真机实测与模拟分开标注")
flow = [
    ("独立参照", "torch 语义另写\nbf16 口径 + fp32"),
    ("CPU 模拟", "同一份设备程序\nnumpy 重放查接线"),
    ("单算子上卡", "随机输入\n对 numpy 期望"),
    ("整模型上卡", "对参照 + 多遍\n逐字节一致"),
    ("14 项自检", "verify_all.sh\n不上板全过"),
    ("提交推送", "结果原样入库\ntest/results/"),
]
fw, fg = 1.75, 0.33
for i, (t_, d_) in enumerate(flow):
    x = ML + i * (fw + fg)
    sh = box(s, x, 2.0, fw, 1.5, fill=ACCENT_SOFT if i not in (3,) else ACCENT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08)
    tc = WHITE if i == 3 else INK
    box_text(sh, [[(t_, {"bold": True, "size": 14, "color": tc})]] + [[(ln, {"size": 10.5, "color": WHITE if i == 3 else MUTED})] for ln in d_.split("\n")])
    if i < len(flow) - 1:
        arrow(s, x + fw + 0.05, 2.62, fg - 0.1, 0.26, LINE)
text(s, ML, 4.0, CW, 0.35, "这套方法实际拦下的问题", size=15, bold=True)
text(s, ML, 4.45, CW, 2.3, [
    [("· 编译器静默算错：", {"bold": True}), ("注意力第一版 CPU 模拟通过、真卡输出全错（10.7 dB）；用单算子上卡二分，定位到 docc 的一种写法编译通过但算错，改写后恢复到约 50 dB。", {"color": MUTED})],
    [("· 口径不混：", {"bold": True}), ("数字分「真机实测 / CPU 实测 / 模拟 / 估算」四类标注；例如曾按 ComfyUI 整条估的 CPU 每步 2.4 s，改用同机实测 1.41 s 替换。", {"color": MUTED})],
    [("· CPU 兜底可见：", {"bold": True}), ("后端选 cpu 或落回 CPU 时计数、打日志，不静默；npu-sim 结果标为模拟，不算 NPU 贡献。", {"color": MUTED})],
], size=12.5, spacing=8)

# ════════════════════════ 12 工具链限制与对策 ════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, 12, "经验", "摸清的工具链限制与对策（docc / dort 0.4.1）")
rows = [
    ["现象", "影响", "对策"],
    ["每颗核的代码不能超过约 1 MB（RISC-V 跳转范围）", "大模型一个程序放不下", "切成多段程序；生成器按实测代码量自动切（UNet 6 段、VAE 3 段）"],
    ["「每行一个数」的向量按行切再做广播：编译通过、真卡算错", "注意力第一版输出全错；模拟器查不出", "单算子上卡二分定位，改成整块运算；新算子先单独上卡"],
    ["32² × 640 通道 1×1 卷积，片上缓冲放不下", "编不出来", "改走矩阵乘"],
    ["f32 向量除法不收；运行时条件分支里不能有矩阵累加", "限制算子写法", "除法放到 bf16 做；循环保持展开"],
    ["calBLAS 注意力每 16 行 q 重读一遍 K / V", "64² 自注意力一次 625 ms", "自写 flash_attn!（7.4×）"],
    ["calBLAS 线性层每 16 行重读一遍权重", "1.0–1.5 TFLOPS", "自写 gemm_rows!（2.4–4 TFLOPS）"],
    ["dort 没有常驻会话，每次调用都重新部署权重", "UNet 每步多 0.5 s", "CFG 两路同一次调用共用部署；待组委会确认"],
]
table(s, ML, 1.7, [4.6, 3.1, 4.43], rows, size=11.5, row_h=0.6, aligns=[L, L, L], bold_first_col=False)

# ════════════════════════ 13 工程成果 ════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, 13, "工程交付", "仓库与可复现性")
m4 = [
    ("6 + 3 + 1", "段设备程序", "UNet 6 段、VAE 3 段、TAESD 1 段，全部 doctor 拒收 0 条"),
    ("151", "UNet 算子特化", "生成器自动产出；全部算子代码合计约 2.8 MB"),
    ("14 / 14", "自检项通过", "verify_all.sh，不上板能做的全部检查"),
    ("6 次", "UNet 真卡提交", "每次都对参照、多遍逐字节一致后才提交"),
]
for i, (v, l_, d_) in enumerate(m4):
    x = ML + i * 3.06
    metric(s, x, 1.75, 2.9, v, l_, d_)
text(s, ML, 3.75, CW, 0.35, "仓库与文档", size=15, bold=True)
text(s, ML, 4.2, CW, 2.6, [
    [("仓库：", {"bold": True}), ("github.com/liyingxxxxx/calculate-hackson（公开），main 分支 1ca5f4b；权重、参数块、calbin 不入库，按 README 重出", {})],
    [("复现：", {"bold": True}), ("bootstrap.sh 新机器从零重建 → verify_all.sh 自检 → board_check.py 上板一条命令；目标是在比赛提供的真卡环境里照 README 复现", {})],
    [("结果：", {"bold": True}), ("test/results/npu3-20261114/ 存每轮原始数据（真卡指标、逐算子计时、每段耗时、CPU 对照）；test/PERF.md 性能全表", {})],
    [("文档：", {"bold": True}), ("PROGRESS.md 逐项进度与证据；docs/DESIGN.md 已核实事实与设计决定；docs/HANDOFF.md 交接；VENDORED.md 复用代码的出处与改动", {})],
], size=12.5, spacing=8)

# ════════════════════════ 14 未完成与风险 ════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, 14, "现状边界", "还没做的与风险")
left = [
    [("UNet 未接入 ComfyUI 采样器", {"bold": True})], [("工作流里的去噪仍在 CPU 上", {"color": MUTED, "size": 11.5})],
    [("UNet 比同机 CPU 慢 1.9 倍", {"bold": True})], [("2.67 s 对 1.41 s；其中 0.5 s 是每次部署权重", {"color": MUTED, "size": 11.5})],
    [("CLIP 未上 NPU；去噪循环未进设备", {"bold": True})], [("单个程序的代码上限决定了 UNet 必须分段", {"color": MUTED, "size": 11.5})],
]
right = [
    [("只有 512 × 512 一个档位", {"bold": True})], [("形状是编译期常量，其他尺寸需另编", {"color": MUTED, "size": 11.5})],
    [("UNet 只在一组输入（时间步 500）上对过参照", {"bold": True})], [("接入采样器后需按整条出图再验", {"color": MUTED, "size": 11.5})],
    [("部分新算子缺单独的测试项", {"bold": True})], [("目前靠整模型上卡对参照覆盖", {"color": MUTED, "size": 11.5})],
]
box(s, ML, 1.75, 5.9, 4.6, fill=BG_SOFT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.04)
box(s, ML + 6.23, 1.75, 5.9, 4.6, fill=BG_SOFT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.04)
text(s, ML + 0.3, 1.95, 5.3, 0.35, "功能与性能", size=14, bold=True, color=ACCENT)
text(s, ML + 0.3, 2.45, 5.3, 3.8, left, size=13, spacing=6)
text(s, ML + 6.53, 1.95, 5.3, 0.35, "覆盖面", size=14, bold=True, color=ACCENT)
text(s, ML + 6.53, 2.45, 5.3, 3.8, right, size=13, spacing=6)

# ════════════════════════ 15 下一步 ════════════════════════
s = prs.slides.add_slide(BLANK)
header(s, 15, "计划", "下一步（按优先级）与需要的支持")
plan = [
    ("1", "UNet 接入 ComfyUI 采样器", "完成 UNet + VAE 都在 NPU 上的提示词出图；按当前速度估算 20 步约 54 s（估算），CPU 约 49 s"),
    ("2", "UNet 继续提速，目标每步低于 CPU 1.41 s", "64² 自注意力（0.84 s，0.41 TFLOPS）、64² / 32² 卷积（0.50 s）"),
    ("3", "CLIP 上 NPU", "LLM 式模板现成（层归一化、线性层、因果注意力），需新写 quick_gelu"),
    ("4", "更多分辨率档位与去噪循环进设备", "视代码上限与组委会对常驻会话的答复决定"),
]
py_ = 1.75
for n_, t_, d_ in plan:
    text(s, ML, py_, 0.6, 0.6, n_, size=26, bold=True, color=ACCENT)
    text(s, ML + 0.7, py_ + 0.02, 7.6, 0.4, t_, size=15, bold=True)
    text(s, ML + 0.7, py_ + 0.44, 7.6, 0.6, d_, size=11.5, color=MUTED)
    py_ += 1.12
ax = ML + 8.75
box(s, ax, 1.75, CW - 8.75, 4.45, fill=ACCENT_SOFT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
text(s, ax + 0.25, 1.95, CW - 9.25, 0.35, "需要组委会确认", size=14, bold=True, color=ACCENT)
text(s, ax + 0.25, 2.4, CW - 9.25, 3.7, [
    [("dort 常驻会话或跳过部署", {"bold": True})],
    [("有的话 UNet 每步省 0.5 s（约 19%）", {"color": MUTED, "size": 11.5})],
    [("每核代码上限能否放宽", {"bold": True})],
    [("决定 UNet 能否少分段、去噪循环能否进设备", {"color": MUTED, "size": 11.5})],
    [("指令级模拟器", {"bold": True})],
    [("公开包里没有，算子内部的数值只能上卡验", {"color": MUTED, "size": 11.5})],
], size=13, spacing=6)

prs.save(OUT)
print(f"{OUT}（{len(prs.slides)} 页）")
