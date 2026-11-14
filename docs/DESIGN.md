# 设计与关键事实

## 目标

路线 B 的标杆：由 **ComfyUI 原生工作流**发起推理，**计算全部在 NPU 上**，稳定出图，支持多尺寸。模型 SD1.5，先做 512×512。
按阶段推进、每阶都能单独交（阶段表在 PROGRESS.md）。

## 已核实的事实（F）

| # | 事实 | 怎么核的 | 影响 |
| --- | --- | --- | --- |
| F1 | `dort calbin run` 是**一次性**的：每次调用都部署、喂输入、等完成、读输出。裸字节模式只有一个输入张量（`--in-tensor`）、一个输出张量（`--out-tensor`）。公开工具链（dort 0.4.1）里**没有常驻会话、没有 Python 绑定**；有 `--deploy-only`，但没有「跳过部署」的选项 | `dort calbin run --help`（2026-11-14）；工具链 sysroot 里只有 docc、dort 两个二进制；赛方 cargo 源上没有 dort / pydort | 每次调用都重新部署权重 ⇒ 见 D2。所有输入要打进一个 input 声明、所有输出一个 output 声明 ⇒ 见 D1 |
| F2 | docc 0.4.1 收 CalVX 1.0：`docc calbin from-calvx` 出 calbin，两核各一个 ELF，`docc calbin doctor` 能查结构 | calBLAS 的 tiny（两芯粒）与本仓的 smoke（单芯粒）都出了 calbin，doctor 拒收 0 条 | 本地能把整条编译链走完、查结构，只有「算得对不对」要上板 |
| F3 | 赛方 cargo 源上的 calblas 0.1.0 带视觉模板（`conv2d!`、`upsample2!` 等），自己的工程经 cargo 源依赖就能用，不用拷源码 | smoke 经 cargo 源依赖、编过 | 本仓不放 SDK 源码 |
| F4 | 输入输出声明不开 `double_buffer` 时，doctor 警告 ping / pong 同址 | smoke、tiny 的 doctor 输出 | 单次推理不受影响；连续推理（同一部署跑多次）要开 |
| F5 | 机器上可能另装着别的工具链、老版本的 docc / dort（npu3 的 `/usr/local/bin` 里就有 dort 0.2.8、docc 0.1.4），平时的 ~/.cargo 里可能配着别的源 | 原机器与 npu3 上的 `tools/check_env.sh` | `env.sh` 隔离 RUSTUP_HOME / CARGO_HOME、把工具链的 bin 放到 PATH 最前；check_env 核 docc / dort 是工具链自带的、Cargo.lock 只有赛方源与 crates.io |
| F6 | 设备遍很快：calBLAS 全部 19 个例子 3.9 s；docc 出 tiny 的 calbin 0.26 s | `~/other/calculet-hackathon/logs/build-workspace-elf.log`、`tools/calbin.sh` 的 timing.txt | 编译不是瓶颈（大模型的 docc 耗时与内存还要实测） |
| F7 | 512² 时的计算量：UNet 一次前向 803 GFLOPs（CFG 两路 ×2，20 步就是 32 TFLOPs）；VAE 解码 2515 GFLOPs，是一次 UNet 的 3 倍多；TAESD 141 GFLOPs；CLIP 13 GFLOPs。签名数（每个签名一份 kernel 的上限）UNet 168、VAE 45、TAESD 25、CLIP 15 | `docs/inventory/`（CPU fp32 实测的调用与形状，FLOPs 是按形状算的） | VAE 解码不便宜，B1 先用 TAESD 打通；UNet 的卷积与线性层占 84%，性能功夫花在这两个模板上 |
| F8 | 对照 calBLAS，真要新写的只有：时间步嵌入（sin / cos）、带 ReLU 的卷积（TAESD）。erf-GELU 的片段在核心库里（`act!(gelu_erf)`），GEGLU 拆成两个 `linear!` + `binary!(mul)`；交叉注意力、因果注意力、头维补零 `attention!` 都有参数 | `docs/inventory/README.md` 的对照栏 | 工作量主要在整模型拼装、偏移表与打包、宿主侧与 ComfyUI，而不是写新 kernel |

| F9 | docc 0.4.1 收得下整个 512² 的 TAESD 解码器：35 个卷积（最宽 512、模板快路每行 8 组 × 4 块交错）、f32 的 `tanh_approx`、入口两核各清一半 @act；from-calvx 4.7 s，每核 ELF 231 KB | `tools/taesd.py calbin`、`out/taesd-real/{timing,doctor}.txt`（2026-11-14） | 大宽度的卷积不用拆；编译仍不是瓶颈。UNet / VAE 照这个写法往上加 |
| F10 | cargo 按 feature 组合各记一份指纹，切回编过的组合不重编；设备遍产物 `npu/<crate>.cvx` 只有一个文件名 ⇒ 拿到的是上一个组合的程序 | ktest 的 tanh 测试读到了 convr_512 的声明（2026-11-14） | 带 feature 的一律用自己的 `--target-dir` |
| F11 | 按 `num_cores()` 分行的单核 kernel（calblas `upsample2!` 等）：yolo26 上板那份是**两核都调** | calBLAS `examples/yolo26/yolo26n.cvx` 的入口 | 我们照做；只让 0 号核调的写法（冒烟原来那样）在 num_cores = 2 时只算一半 ⇒ 模拟器当错报。上板用 `ref/ktest.py board up_256` 核这个假设 |
| F12 | ComfyUI 的 LATENT 在 VAE 原空间（KSampler 出口 `process_latent_out` ÷ 0.18215）；ComfyUI 自己加载 taesd 时 vae_scale = 0.18215 | ComfyUI v0.45.3 `comfy/latent_formats.py`、`nodes.py` 的 `load_taesd` | 节点把 LATENT × 0.18215 再喂 TAESD；输出夹到 [0, 1] 就是 IMAGE |
| F13 | `dort calbin run` 还有 `-j`（JSON 输出）、`--deploy-only`、`--dump-layout`（部署后各段的落点）、`--board-lock`（缺省 `/opt/data/locks/board.lock`）、`--lock-wait`（缺省 900 s） | `dort calbin run --help`（2026-11-14，只看帮助） | 部署与推理分开计时：`--deploy-only` 另跑一次相减；现场的板锁按赛方规矩给（`SD15NPU_BOARD_LOCK`） |

| F14 | docc 0.4.1 没接 `vec.round`（「VU 的运算表里没有」） | 时间步嵌入第一版（2026-11-14） | 取整用魔数加减：(x + 1.5·2²³) − 1.5·2²³（\|x\| < 2²²） |
| F15 | VU 拿「VB 里的标量」（立即数也算）和向量做运算时，向量要在一行 512 B 以内或整行 512 B 的倍数 | 时间步嵌入 [1, 160] f32（640 B）docc 报「与 VB 里的标量做」选不出指令 | 小向量补到整行（128 个 f32 / 256 个 bf16）；时间步嵌入把 cos、sin 两半并进 3 整行一起算 |

| F16 | **公开环境里，除了真机没有任何能执行 kernel 的东西**：npu 的原语在主机上一调就是 `unreachable!("…⛔ 执行")`；calDNN 的图 API 只编排出作业表，注释写「发射由主机运行库（dort 的 jit 路径）做」；docc 只有 calbin / elf / gen 三类子命令（打包、检查、比对、反汇编）；dort 要真机；calBLAS 文档里的指令级模拟器（calvxus 的 `cvx run`）不在公开包里。`-elf` 与 `-cvx` 两个设备 target 现在出的 CalVX 逐字节相同（`-elf` 提示「ELF 要 docc 收 CalVX 1.0，还没接上」） | 读 npu `src/lib.rs`、calDNN `src/{lib,job}.rs`；`docc --help`；工具链 sysroot 的 bin（2026-11-14）。找 calvxus（2026-11-14）：公开仓库里只当「旁边另有一份检出」引用（`../calculet-calvxus/target/release/cvx`、`~/src/complier/calculet-calvxus`）；calBLAS `examples/qwen36_35b_a3b/deploy/Cargo.toml` 依赖 `calculet-calvxus` 0.1.0（calculet 源），但源上查 `calculet-calvxus` / `calvxus` / `calvx` / `cvx` 都是 404；Gitea 公开的组织只有 calculet、仓库只有 calBLAS / calDNN / npu，全站搜 calvxus / cvx 为空；赛题页不提模拟器 | kernel 内部算得对不对只能上板验（或向组委会要 `cvx`）；我们的 numpy 模拟器只查到 kernel 调用这一级（D9） |
| F17 | calDNN 的图运算（`Op`）里没有二维卷积、上采样、GroupNorm：它的 `Conv` 是一维因果 depthwise（Qwen3.5 一类的线性注意力用），`NormKind` 只有 Rms / RmsOffset / L2 / Layer / LayerBias | calDNN `src/graph.rs` | SD 的图现在用 calDNN 表达不了；calBLAS 自己的 CV 例子 yolo26 也没走它（自己的作业表 + 整步程序）。我们照 yolo26 的路子；「被图调用」要等组委会答复能不能加 `Op` 变体 |

| F18 | 真卡要满足：设备端 `OS up / RT up`、**板上跑 DeviceRT**（不是的话这一版 dort 开设备会被拒，`dort device` 里写「拒」，要重刷 DeviceRT）、没有别的进程在用。`dort calbin parse` 是纯静态的（不开设备），能在没卡时核我们的包 | 本机的卡（只读 `dort device`，2026-11-14）；`dort calbin parse --help` | 上板一律走 `tools/board_check.py`：先只读查这三条，不满足就停；刷写、复位不由我们做 |
| F19 | **真机实测**：`cos_approx`（时间步嵌入用）的输出在板上**全部落在 1/128 的绝对网格上**（期望值只有 37–47% 恰好在网格上），多半往 0 截；最大差 2.5/128 ≈ 0.0195（t = 999），余弦最差 0.99996；t = 0 逐位相同。同一批里 f32 的 `tanh_approx` 最大差 2.4e-7（最多 2 个 bf16 ULP），卷积最差 1 ULP(输出最大值)，上采样逐位相同 | npu3 真卡 `board_check.py`（2026-11-14，`out/ktest/temb/*/board.json`，板上读回值 × 128 全是整数） | cos_approx 的分辨率比 bf16 在 0.5 附近（≈ 0.002）粗约 10 倍；时间步嵌入要喂 UNet（B3），到时对 diffusers 看端到端影响，不够就换成 f32 多项式（规约后在 [−π, π] 上）。`ref/ktest.py` 的 temb 门槛在换掉之前只能收到 3 ULP(max) |
| F20 | **真机实测**：npu3 的卡（PCI 0000:07:00.0）满足 F18：`OS up / RT up`、2 个芯粒、板上 DRT（rt 3.1 / ABI 1.1、线协议 rev 1），dort 0.4.1（线协议 rev 2）「放行（按 rev 1 说）」。一次 `dort calbin run` 的墙钟很短：小 kernel 约 2 ms，TAESD 只部署 3 ms（参数块 2.6 MB，RT 日志「1 ms」），整次（写 0.3 MB ⇒ 跑 ⇒ 读 17 MB）0.11 s | `out/board/device-20261114-154020.log`、`out/taesd-board/image/{dort.log,metrics.json}` | F1 的「每次调用都重新部署」对 TAESD 这种 2.6 MB 的权重不是瓶颈；UNet（bf16 约 1.7 GB）部署多久要实测，D2 先不改 |

| F21 | 正式 VAE 要用的 kernel，docc 0.4.1 收不收（2026-11-14 探针，只编不跑）：卷积 3×3 512→512 @128²（快路放不下 ⇒ 原路）、@64² 带残差（快路）、1×1 512→256 @256²、3×3 128→128 @512²（原路）、32→512、128→32 都收；`group_norm!`（NCHW，G = 32，silu）在 64² × 512、256² × 512、512² × 128 都收；`nhwc_to_nchw!` / `nchw_to_nhwc!` 512² × 128 收；`upsample2!` 512 通道按两个 256 通道的窗口各调一次收。**`attention!` 头维 512 不收**：「calt-place：gib1 放不下：要到 131072 字节，只有 65536」（rhs [D, 128] bf16 = 128 KB，TU 的操作数上限 64 KB）；头维 256、128 收 | `$CALCULET_HOME/work/vprobe`（探针工程，不进交付件；`results.txt`、`logs/`） | VAE 中间块的单头 512 维注意力要新写按头维分段的 kernel（D14）；卷积在 128² 以上大多走原路（慢，calBLAS 文档说约 4 倍），先求对、性能放 B5 |

| F22 | **docc 0.4.1 每颗核的代码不能超过约 1 MB**：VAE 的 45 个 kernel 编进一个程序，docc 出 ELF 时报「riscvc：链接：main0.o：.text 处到 `.LBB25_1` 的距离 -1120872 超出 21 位」（RISC-V 的跳转 ±1 MB）；docc 没有放宽它的选项。单个 kernel 的代码量（探针，每个连着一个最小的入口）：64² × 512 的 `group_norm!` 292 KB（16 组的循环被整个展开）、其余 `group_norm!` 112 KB、`attn_wide!` 215 KB、512 通道快路的卷积 106 KB、原路卷积 26–51 KB、排布转换 35–40 KB；VAE 合计约 2.7 MB | `tools/calbin.sh vae`（2026-11-14）、`$CALCULET_HOME/work/vprobe/logs/` | 大模型要切成几段程序，每段 < 1 MB（D15）；换掉排布转换 + calblas GN（D13 → NHWC 原生 GN）能省很多代码。**128² × 512 的 `group_norm!` 单个 kernel 就超**（缺省切 16 段，docc 把外层 16 组也展开，约 1.1 MB）⇒ 本仓抄一份加 `parts = N`，切 64 段是 232 KB（64² 的反过来：缺省 4 段 292 KB、64 段 788 KB，保持缺省）。切 6 段之后每核 ELF：s1 753 KB、s2 910 KB、s3 766 KB、s4 559 KB、s5 723 KB、s6 560 KB，doctor 拒收都是 0 |

| F23 | **真机实测**：正式 VAE 的 6 段程序在 npu3 上第一次就算对（3 遍逐字节一致，对 bf16 参照 58.3 dB），但 6 段合计 5.18 s，比同机 CPU fp32 慢约 3 倍。每段的折合算力（估算）：64² 的快路卷积约 1 TFLOPS，128² 0.8、256² 0.6、512² 只有 0.24 TFLOPS；部署 314 MB 参数只要 0.12 s | `test/PERF.md`、`test/results/npu3-20261114/vae/` | 性能的大头在大图卷积走不了快路（输入 3 行 + 16 像素放不进 384 KB 的 VB，512² × 128 通道要 400 KB）与 GroupNorm 的排布转换（D13）；先给宽图的卷积按行切半走快路，再换 NHWC 原生 GN |

calBLAS 文档里写的、我们还没亲自核的（上板时核）：KS1 的 exp / rsqrt 是查表近似（≤ 1 ULP），除法精确；DDR 实测约 84 GB/s；
35B（21 GB）光部署权重 195 秒（粗算 0.1 GB/s 量级）。

## 设计决定（D）

- **D1 宿主侧接口。** 每个设备程序一个 calbin；所有输入（latent、时间步表、文本嵌入……）按固定偏移打进**一个** input 声明，所有输出一个 output 声明
  （F1 的裸字节模式只认一个输入、一个输出张量）。偏移表用 Rust 的 `const fn` 算，打包工具与设备程序共用同一份（calBLAS 的 d3_turbo 就这么做）。
  Python 侧（ComfyUI 节点）只做三件事：按偏移表拼输入字节、调 `dort calbin run --raw-in/--raw-out`、按偏移表解输出。
- **D2 ComfyUI 接法：先走「整个去噪循环在一个设备程序里」。** F1 意味着每调一次 dort 就重新部署一次权重（UNet bf16 约 1.7 GB，按文档的数粗算十几秒），
  所以「每一步调一次 UNet」（接法 1，最原生）在公开工具链上不可行。先做接法 2：CLIP、N 步 UNet（含 CFG 与调度器）、VAE 解码尽量在一次 dort 调用里做完；
  ComfyUI 侧给一个自定义采样器（SamplerCustom 的 SAMPLER）或文生图节点，工作流其余部分（加载、提示词、出图）照原生的来。
  拆成几个程序还是合成一个，等量出部署时间与 docc 的耗时再定。已向组委会提的问题：有没有常驻会话 / Python 绑定 / 跳过部署——有了就补接法 1。
- **D3 排布。** UNet 与 VAE 的特征图一律用 calBLAS 卷积模板的约定：NHWC、四周补两圈边框（边框是 0，3×3 卷积的 padding 就是它），拼接靠通道窗口、不搬数据。
  `group_norm!` 现在按 NCHW 的连续段读 ⇒ 先插排布转换跑通，再写 NHWC 原生的 GroupNorm+SiLU（A1 候选，省两次搬运）。
- **D4 档位。** 形状是编译期常量 ⇒ 先只做 512²（latent 64²）；其它尺寸以后按档位加，运行时按 latent 尺寸挑程序，不支持的尺寸点名拒绝。
- **D5 数值口径。** 激活 bf16、卷积与矩阵乘 f32 累加（模板的做法）。两种参照：diffusers fp32（看出图质量）、按 torch 语义写的 bf16 numpy 参照（逐算子比）。
  exp / rsqrt 是近似指令，不承诺逐位，容差写进已知限制。
- **D6 CPU 承担的部分**（要进 NPU 贡献表）：分词、初始噪声（按种子生成）、输入输出的打包与解包、PNG 编码。其余目标是全部在 NPU 上。
- **D7 时间步嵌入在设备上算。** ComfyUI 的采样器由 sigma 插出时间步，可能是小数，打包时算好的整数表不够用；VU 有 sin / cos 原语，写一个小 kernel
  （320 维，只有几百个元素）。这是 A1 的第一个新 kernel。
- **D8 TAESD 的 ReLU：** 给卷积模板加一个 relu 分支（从 calblas 的 `conv2d!` 抄出来改，记 VENDORED）；Block 末尾的 relu(conv + skip) 用现成的 `binary!(relu)`
  或卷积的残差参数之后再 relu。

- **D9 接线先在 CPU 上查死。** 上板机会少、一轮要排队，所以凡是「主机侧 / 接线」的错都要在上板前排除：
  模拟器读**编出来的 `.cvx`**（不是另写的作业表）按真实地址重放，查对齐、越界、原地、屏障、核数、未初始化内存，并用 torch 写的独立参照比数；
  模拟器自己由 `ref/test_cvxsim.py` 的变异测试守着。它不查 kernel 内部（那要上板），也不是板子的逐位参照（近似指令）。
- **D10 上板先逐个 kernel，再整模型。** `crates/ktest` 每个 feature 一个只调一个 kernel 的程序，kernel 特化就是整步程序里那份（`#[path]` 引同一个 kernels.rs）；
  整模型在板上算错时用它们二分。新写 / 新改的路径（relu、res_relu、tanh、512 宽快路、两核分行）各有一个。
- **D11 ComfyUI 接法（B1 这一段）。** 自定义节点 `SD15 NPU · TAESD Decode`：LATENT ⇒ IMAGE，后端 `npu` / `npu-sim` / `cpu` 三选一，
  `npu` 没开上板开关就报错，**不静默落回 CPU**；`cpu` 每用一次计数、打日志。ComfyUI 装在单独的 venv（和参照环境不混版本），钉 v0.45.3。
  工作流其余部分（加载、提示词、采样）现在是 ComfyUI 原生节点在 CPU 上跑，B2–B4 逐段换成 NPU。

- **D12 上板一条命令、先空跑。** `tools/board_check.py` 把上板的步骤与门槛定死（重出 calbin ⇒ 静态核对 ⇒ 只读查卡 ⇒ 逐个 kernel ⇒ 整模型，任何一步不对就停）；
  没卡时用测试替身 `ref/fake_dort.py` 把整套空跑（`ref/test_board_kit.py`，在 verify_all 里），保证真卡难得的那一次不浪费在流程自己的错上。

- **D13 正式 VAE 的 GroupNorm：先走排布转换 + calblas `group_norm!`。** 每个 GN 是三步：`nhwc_to_nchw!`（带边框的图 ⇒ 稠密 NCHW）⇒ `group_norm!`（可带 silu）⇒ `nchw_to_nhwc!`（只写内部，边框保持 0）。三个都是 calblas 现成的，板上出错的面小；代价是每个 GN 多两次整图搬运，逐通道的 w、b 要按「每行一个值」展开（全部 30 个 GN 约 220 MB 参数）。跑通之后再写 NHWC 原生的 GroupNorm+SiLU（A1，UNet 也要用），换掉这三步。
- **D14 正式 VAE 中间块的注意力走稠密 token。** GN（NCHW）⇒ `transpose2d!` ⇒ [4096 token, 512] ⇒ `linear!`（多行、HF 权重、f32 bias）出 q、k、v ⇒ 注意力 ⇒ `transpose2d!` + `nchw_to_nhwc!` 回到带边框的图 ⇒ 1×1 卷积（to_out）带残差。注意力单头 512 维，`attention!` 收不下（F21） ⇒ 从 calblas `attention!` 抄出一份，把头维按 128 一段切开：q·kᵀ 在 TU 里沿段接着累加，p·v 按段各一个累加器（记 VENDORED）。

- **D15 正式 VAE 切成 6 段程序、主机串起来。** 每段一个 calbin、一次 `dort calbin run`，上一段的 out_y 原样当下一段的 in_x（F22）。切点选在「下一段只经 GroupNorm 的排布转换、残差、1×1 卷积读这张图」的地方：它们只读内部，所以上一段输出的边框不用清。每段只部署自己的权重（参数块按段打），@act 只放这一段用到的图。代价是段间的特征图经主机转手（最大一张 512² × 256 通道 136 MB），计算不在 CPU 上，记在 NPU / CPU 分工里。以后换成 NHWC 原生的 GroupNorm、代码变小了，再并段。

## 未定 / 要问组委会

- 常驻会话、Python 绑定、跳过部署（决定 D2 能不能换成接法 1）；
- 能不能提供指令级模拟器（calBLAS AGENTS.md 里写的 calvxus `cvx run`）：上板前验 kernel 内部（F16）；
- 现场板子的排队方式与单次时长；统一复测用的设备、固件、工具链版本；
- 能不能给 calDNN 加 `Op` 变体（路线 A 的「被图调用」要用）；SDK 源码能不能进公开仓库。
