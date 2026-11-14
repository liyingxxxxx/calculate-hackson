# SD1.5 算子清单（512×512）

由 `ref/op_inventory.py` 生成（CPU、fp32、diffusers 的参照实现），不要手改；重出：见 README「CPU 参照」。

环境：torch 2.15.0+cpu；模型 `/home/liyingxuan/other/models/stable-diffusion-v1-5`（revision 见该目录的 download-verification.json）。

「对照 calBLAS」一栏：现成 = 模板直接能用；改参数 = 模板能用但要补零、换掩码之类；新写 = 没有模板；不搬 = 在整步程序里靠排布消掉、不起 kernel；主机 = 放 CPU（要记进 NPU 贡献表）。FLOPs 只算卷积、线性层、注意力、矩阵乘；逐元素与归一化按「读写元素」看访存。

### CLIP 文本编码器

输入：一条 prompt，补到 77 个 token（SD 的做法）；CFG 时空 prompt 再跑一次。

CPU fp32 跑一遍 0.1 s；198 次计算类调用、15 种签名（= 要特化的 kernel 数的上限）；另有 182 次纯视图调用不计；总计 13.3 GFLOPs。

| 算子 | 签名数 | 调用数 | GFLOPs | 占比 | 读写元素（亿，未融合） | 对照 calBLAS |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `linear` | 3 | 72 | 13.1 | 98.4% | 0.98 | 现成 |
| `scaled_dot_product_attention` | 1 | 12 | 0.2 | 1.6% | 0.03 | 改参数 |
| `add` | 1 | 25 | 0.0 | 0.0% | 0.04 | 现成 |
| `layer_norm` | 1 | 25 | 0.0 | 0.0% | 0.03 | 现成 |
| `contiguous` | 2 | 24 | 0.0 | 0.0% | 0.03 | 不搬 |
| `mul` | 2 | 24 | 0.0 | 0.0% | 0.14 | 现成 |
| `sigmoid` | 1 | 12 | 0.0 | 0.0% | 0.06 | 改参数 |
| `embedding` | 2 | 2 | 0.0 | 0.0% | 0.38 | 现成 |
| `arange` | 1 | 1 | 0.0 | 0.0% | 0.00 | 不搬 |
| `argmax` | 1 | 1 | 0.0 | 0.0% | 0.00 | 不搬 |

要新写 / 改参数 / 待定的签名：

| 算子 | 输入形状 | 关键参数 | 次数 | 状态 | 说明 | 出现在 |
| --- | --- | --- | ---: | --- | --- | --- |
| `scaled_dot_product_attention` | `[[1, 12, 77, 64], [1, 12, 77, 64], [1, 12, 77, 64]]` | `{"mask": false, "causal": true, "kind": "self"}` | 12 | 改参数 | attention!(mask = full)：因果 + pad 的掩码由主机给 | encoder.layers.0.self_attn, encoder.layers.1.self_attn, encoder.layers.2.self_attn |
| `sigmoid` | `[[1, 77, 3072]]` | `{}` | 12 | 改参数 | CLIP 的 quick_gelu = x · σ(1.702x)：新写一个 unary，或 binary!(mul_sigmoid) 前接一次缩放 | encoder.layers.0.mlp.activation_fn, encoder.layers.1.mlp.activation_fn, encoder.layers.2.mlp.activation_fn |

### UNet（一次前向，批 1）

输入：latent 1×4×64×64（512² 的图）、时间步 999、文本嵌入 1×77×768。CFG 每步要跑两路（批 2 或连调两次），FLOPs 翻倍。

CPU fp32 跑一遍 1.5 s；797 次计算类调用、168 种签名（= 要特化的 kernel 数的上限）；另有 642 次纯视图调用不计；总计 803.3 GFLOPs。

| 算子 | 签名数 | 调用数 | GFLOPs | 占比 | 读写元素（亿，未融合） | 对照 calBLAS |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `conv2d` | 35 | 98 | 443.9 | 55.3% | 7.30 | 改参数 / 现成 |
| `linear` | 23 | 184 | 233.3 | 29.0% | 5.74 | 现成 |
| `scaled_dot_product_attention` | 8 | 32 | 126.1 | 15.7% | 0.71 | 改参数 / 现成 |
| `add` | 12 | 108 | 0.0 | 0.0% | 1.99 | 现成 |
| `dropout` | 12 | 70 | 0.0 | 0.0% | 1.63 | 不搬 |
| `silu` | 15 | 68 | 0.0 | 0.0% | 0.67 | 现成 |
| `group_norm` | 18 | 61 | 0.0 | 0.0% | 0.90 | 现成 |
| `div` | 9 | 55 | 0.0 | 0.0% | 0.70 | 不搬 |
| `layer_norm` | 4 | 48 | 0.0 | 0.0% | 0.69 | 现成 |
| `mul` | 7 | 19 | 0.0 | 0.0% | 1.39 | 现成 |
| `gelu` | 4 | 16 | 0.0 | 0.0% | 0.92 | 改参数 |
| `contiguous` | 4 | 16 | 0.0 | 0.0% | 0.23 | 不搬 |
| `cat` | 9 | 14 | 0.0 | 0.0% | 0.31 | 不搬 |
| `interpolate` | 3 | 3 | 0.0 | 0.0% | 0.05 | 现成 |
| `tensor` | 1 | 1 | 0.0 | 0.0% | 0.00 | 不搬 |
| `arange` | 1 | 1 | 0.0 | 0.0% | 0.00 | 新写 |
| `exp` | 1 | 1 | 0.0 | 0.0% | 0.00 | 新写 |
| `sin` | 1 | 1 | 0.0 | 0.0% | 0.00 | 新写 |
| `cos` | 1 | 1 | 0.0 | 0.0% | 0.00 | 新写 |

要新写 / 改参数 / 待定的签名：

| 算子 | 输入形状 | 关键参数 | 次数 | 状态 | 说明 | 出现在 |
| --- | --- | --- | ---: | --- | --- | --- |
| `conv2d` | `[[1, 4, 64, 64], [320, 4, 3, 3], [320]]` | `{"k": [3, 3], "stride": [1, 1], "pad": [1, 1], "groups": 1, "bias": true}` | 1 | 改参数 | conv2d!（NHWC 带边框）；Cin 4 补到 32 的倍数 | conv_in |
| `conv2d` | `[[1, 320, 64, 64], [4, 320, 3, 3], [4]]` | `{"k": [3, 3], "stride": [1, 1], "pad": [1, 1], "groups": 1, "bias": true}` | 1 | 改参数 | conv2d!（NHWC 带边框）；Cout 4 补齐 | conv_out |
| `gelu` | `[[1, 4096, 1280]]` | `{"approximate": "none"}` | 5 | 改参数 | erf-GELU 的片段现成（frag.rs 的 act!(gelu_erf)，A&S 7.1.26，f32）：GEGLU 把投影拆成门、值两个 linear!，门那个带 act = gelu_erf，再 binary!(mul)；不用新写 kernel | down_blocks.0.attentions.0.transformer_blocks.0.ff.net.0, down_blocks.0.attentions.1.transformer_blocks.0.ff.net.0, up_blocks.3.attentions.0.transformer_blocks.0.ff.net.0 |
| `gelu` | `[[1, 1024, 2560]]` | `{"approximate": "none"}` | 5 | 改参数 | erf-GELU 的片段现成（frag.rs 的 act!(gelu_erf)，A&S 7.1.26，f32）：GEGLU 把投影拆成门、值两个 linear!，门那个带 act = gelu_erf，再 binary!(mul)；不用新写 kernel | down_blocks.1.attentions.0.transformer_blocks.0.ff.net.0, down_blocks.1.attentions.1.transformer_blocks.0.ff.net.0, up_blocks.2.attentions.0.transformer_blocks.0.ff.net.0 |
| `gelu` | `[[1, 256, 5120]]` | `{"approximate": "none"}` | 5 | 改参数 | erf-GELU 的片段现成（frag.rs 的 act!(gelu_erf)，A&S 7.1.26，f32）：GEGLU 把投影拆成门、值两个 linear!，门那个带 act = gelu_erf，再 binary!(mul)；不用新写 kernel | down_blocks.2.attentions.0.transformer_blocks.0.ff.net.0, down_blocks.2.attentions.1.transformer_blocks.0.ff.net.0, up_blocks.1.attentions.0.transformer_blocks.0.ff.net.0 |
| `gelu` | `[[1, 64, 5120]]` | `{"approximate": "none"}` | 1 | 改参数 | erf-GELU 的片段现成（frag.rs 的 act!(gelu_erf)，A&S 7.1.26，f32）：GEGLU 把投影拆成门、值两个 linear!，门那个带 act = gelu_erf，再 binary!(mul)；不用新写 kernel | mid_block.attentions.0.transformer_blocks.0.ff.net.0 |
| `scaled_dot_product_attention` | `[[1, 8, 4096, 40], [1, 8, 4096, 40], [1, 8, 4096, 40]]` | `{"mask": false, "causal": false, "kind": "self"}` | 5 | 改参数 | spatial_attn! 或 attention!(mask = keys)；头维 40 补零到 16 的倍数（scale 给真头维） | down_blocks.0.attentions.0.transformer_blocks.0.attn1, down_blocks.0.attentions.1.transformer_blocks.0.attn1, up_blocks.3.attentions.0.transformer_blocks.0.attn1 |
| `scaled_dot_product_attention` | `[[1, 8, 4096, 40], [1, 8, 77, 40], [1, 8, 77, 40]]` | `{"mask": false, "causal": false, "kind": "cross"}` | 5 | 改参数 | attention!(keys = 128, mask = keys)：77 个上下文 token 补到 128；头维 40 补零到 16 的倍数（scale 给真头维） | down_blocks.0.attentions.0.transformer_blocks.0.attn2, down_blocks.0.attentions.1.transformer_blocks.0.attn2, up_blocks.3.attentions.0.transformer_blocks.0.attn2 |
| `scaled_dot_product_attention` | `[[1, 8, 1024, 80], [1, 8, 77, 80], [1, 8, 77, 80]]` | `{"mask": false, "causal": false, "kind": "cross"}` | 5 | 改参数 | attention!(keys = 128, mask = keys)：77 个上下文 token 补到 128 | down_blocks.1.attentions.0.transformer_blocks.0.attn2, down_blocks.1.attentions.1.transformer_blocks.0.attn2, up_blocks.2.attentions.0.transformer_blocks.0.attn2 |
| `scaled_dot_product_attention` | `[[1, 8, 256, 160], [1, 8, 77, 160], [1, 8, 77, 160]]` | `{"mask": false, "causal": false, "kind": "cross"}` | 5 | 改参数 | attention!(keys = 128, mask = keys)：77 个上下文 token 补到 128 | down_blocks.2.attentions.0.transformer_blocks.0.attn2, down_blocks.2.attentions.1.transformer_blocks.0.attn2, up_blocks.1.attentions.0.transformer_blocks.0.attn2 |
| `scaled_dot_product_attention` | `[[1, 8, 64, 160], [1, 8, 77, 160], [1, 8, 77, 160]]` | `{"mask": false, "causal": false, "kind": "cross"}` | 1 | 改参数 | attention!(keys = 128, mask = keys)：77 个上下文 token 补到 128 | mid_block.attentions.0.transformer_blocks.0.attn2 |
| `arange` | `[]` | `{}` | 1 | 新写 | 时间步嵌入（sin / cos）：设备上一个小 kernel（VU 有 sin / cos 原语）；时间步可能是小数（ComfyUI 由 sigma 插出来），所以不能只靠打包时算好的整数表 | time_proj |
| `cos` | `[[1, 160]]` | `{}` | 1 | 新写 | 时间步嵌入（sin / cos）：设备上一个小 kernel（VU 有 sin / cos 原语）；时间步可能是小数（ComfyUI 由 sigma 插出来），所以不能只靠打包时算好的整数表 | time_proj |
| `exp` | `[[160]]` | `{}` | 1 | 新写 | 时间步嵌入（sin / cos）：设备上一个小 kernel（VU 有 sin / cos 原语）；时间步可能是小数（ComfyUI 由 sigma 插出来），所以不能只靠打包时算好的整数表 | time_proj |
| `sin` | `[[1, 160]]` | `{}` | 1 | 新写 | 时间步嵌入（sin / cos）：设备上一个小 kernel（VU 有 sin / cos 原语）；时间步可能是小数（ComfyUI 由 sigma 插出来），所以不能只靠打包时算好的整数表 | time_proj |

### VAE 解码器

输入：latent 1×4×64×64 ⇒ 图 1×3×512×512。每张图只跑一次。

CPU fp32 跑一遍 2.9 s；149 次计算类调用、45 种签名（= 要特化的 kernel 数的上限）；另有 29 次纯视图调用不计；总计 2514.5 GFLOPs。

| 算子 | 签名数 | 调用数 | GFLOPs | 占比 | 读写元素（亿，未融合） | 对照 calBLAS |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `conv2d` | 13 | 36 | 2471.6 | 98.3% | 12.51 | 改参数 / 现成 |
| `scaled_dot_product_attention` | 1 | 1 | 34.4 | 1.4% | 0.08 | 现成 |
| `linear` | 1 | 4 | 8.6 | 0.3% | 0.18 | 现成 |
| `group_norm` | 7 | 30 | 0.0 | 0.0% | 9.19 | 现成 |
| `silu` | 6 | 29 | 0.0 | 0.0% | 9.14 | 现成 |
| `div` | 5 | 16 | 0.0 | 0.0% | 3.78 | 不搬 |
| `dropout` | 5 | 15 | 0.0 | 0.0% | 3.77 | 不搬 |
| `add` | 4 | 15 | 0.0 | 0.0% | 5.66 | 现成 |
| `interpolate` | 3 | 3 | 0.0 | 0.0% | 1.36 | 现成 |

要新写 / 改参数 / 待定的签名：

| 算子 | 输入形状 | 关键参数 | 次数 | 状态 | 说明 | 出现在 |
| --- | --- | --- | ---: | --- | --- | --- |
| `conv2d` | `[[1, 4, 64, 64], [4, 4, 1, 1], [4]]` | `{"k": [1, 1], "stride": [1, 1], "pad": [0, 0], "groups": 1, "bias": true}` | 1 | 改参数 | conv2d!（NHWC 带边框）；Cin 4 补到 32 的倍数；Cout 4 补齐 | post_quant_conv |
| `conv2d` | `[[1, 4, 64, 64], [512, 4, 3, 3], [512]]` | `{"k": [3, 3], "stride": [1, 1], "pad": [1, 1], "groups": 1, "bias": true}` | 1 | 改参数 | conv2d!（NHWC 带边框）；Cin 4 补到 32 的倍数 | decoder.conv_in |
| `conv2d` | `[[1, 128, 512, 512], [3, 128, 3, 3], [3]]` | `{"k": [3, 3], "stride": [1, 1], "pad": [1, 1], "groups": 1, "bias": true}` | 1 | 改参数 | conv2d!（NHWC 带边框）；Cout 3 补齐 | decoder.conv_out |

### TAESD 解码器

输入：同一个 latent 1×4×64×64 ⇒ 图 1×3×512×512（`/home/liyingxuan/other/models/taesd`）。B1 先用它，正式 VAE 随后。

CPU fp32 跑一遍 0.3 s；84 次计算类调用、25 种签名（= 要特化的 kernel 数的上限）；另有 0 次纯视图调用不计；总计 141.4 GFLOPs。

| 算子 | 签名数 | 调用数 | GFLOPs | 占比 | 读写元素（亿，未融合） | 对照 calBLAS |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `conv2d` | 9 | 35 | 141.4 | 100.0% | 2.63 | 改参数 / 现成 |
| `relu` | 4 | 31 | 0.0 | 0.0% | 2.00 | 新写 |
| `add` | 4 | 10 | 0.0 | 0.0% | 1.00 | 现成 |
| `interpolate` | 3 | 3 | 0.0 | 0.0% | 0.28 | 现成 |
| `mul` | 2 | 2 | 0.0 | 0.0% | 0.02 | 现成 |
| `div` | 1 | 1 | 0.0 | 0.0% | 0.00 | 现成 |
| `tanh` | 1 | 1 | 0.0 | 0.0% | 0.00 | 改参数 |
| `sub` | 1 | 1 | 0.0 | 0.0% | 0.02 | 现成 |

要新写 / 改参数 / 待定的签名：

| 算子 | 输入形状 | 关键参数 | 次数 | 状态 | 说明 | 出现在 |
| --- | --- | --- | ---: | --- | --- | --- |
| `conv2d` | `[[1, 4, 64, 64], [64, 4, 3, 3], [64]]` | `{"k": [3, 3], "stride": [1, 1], "pad": [1, 1], "groups": 1, "bias": true}` | 1 | 改参数 | conv2d!（NHWC 带边框）；Cin 4 补到 32 的倍数 | decoder.layers.0 |
| `conv2d` | `[[1, 64, 512, 512], [3, 64, 3, 3], [3]]` | `{"k": [3, 3], "stride": [1, 1], "pad": [1, 1], "groups": 1, "bias": true}` | 1 | 改参数 | conv2d!（NHWC 带边框）；Cout 3 补齐 | decoder.layers.18 |
| `tanh` | `[[1, 4, 64, 64]]` | `{}` | 1 | 改参数 | TAESD 开头的 tanh(x / 3) · 3：VU 有 tanh 原语，写成一个小的逐元素 kernel 或并进第一个卷积之前的一步 | decoder |
| `relu` | `[[1, 64, 64, 64]]` | `{}` | 10 | 新写 | conv2d! 的 act 只有 silu、none 两种：给卷积模板加一个 relu 分支（抄模板要记 VENDORED）；Block 末尾的 relu(conv + skip) 可用 binary!(relu) | decoder.layers.1, decoder.layers.2.conv.1, decoder.layers.2.fuse |
| `relu` | `[[1, 64, 128, 128]]` | `{}` | 9 | 新写 | conv2d! 的 act 只有 silu、none 两种：给卷积模板加一个 relu 分支（抄模板要记 VENDORED）；Block 末尾的 relu(conv + skip) 可用 binary!(relu) | decoder.layers.7.conv.1, decoder.layers.7.fuse, decoder.layers.8.conv.1 |
| `relu` | `[[1, 64, 256, 256]]` | `{}` | 9 | 新写 | conv2d! 的 act 只有 silu、none 两种：给卷积模板加一个 relu 分支（抄模板要记 VENDORED）；Block 末尾的 relu(conv + skip) 可用 binary!(relu) | decoder.layers.12.conv.1, decoder.layers.12.fuse, decoder.layers.13.conv.1 |
| `relu` | `[[1, 64, 512, 512]]` | `{}` | 3 | 新写 | conv2d! 的 act 只有 silu、none 两种：给卷积模板加一个 relu 分支（抄模板要记 VENDORED）；Block 末尾的 relu(conv + skip) 可用 binary!(relu) | decoder.layers.17.conv.1, decoder.layers.17.fuse |
