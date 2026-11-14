# 进度

每做完一项就在这里勾上，写日期与证据（证据目录 `$CALCULET_HOME/logs/`、本仓 `out/`、`docs/inventory/` 与 `test/`）。2026-11-14 之前的条目是在原机器上做的，路径写的是原机器的（`~/other/calculet-hackathon` = 那台的 CALCULET_HOME）；之后在比赛提供的 npu3 上（`/home/sweet/hachson`）。

| 阶 | 内容 | 状态 |
| --- | --- | --- |
| A0 | 环境、编译链、CPU 参照、算子清单 | ✅ 2026-11-14 |
| A1 | 新写的算子（quick_gelu、erf-GELU / GEGLU、时间步嵌入、NHWC GroupNorm+SiLU……），各配独立参照与边界测试 | 进行中：时间步嵌入**真机实测**过了（2026-11-14 npu3；cos_approx 分辨率 1/128，DESIGN F19）；quick_gelu、NHWC GroupNorm 未写 |
| B1 | VAE 解码上 NPU（先 TAESD，再正式 VAE），接进 ComfyUI | ✅ 2026-11-14：TAESD 与正式 VAE 都**真机跑通**、ComfyUI 解码走 npu；正式 VAE 优化后 0.98 s，比 CPU 快约 1.75 倍（V7） |
| B2 | CLIP 文本编码器上 NPU | |
| B3 | UNet 一次前向对 diffusers | 进行中：U0 可行性探针、U1 独立参照做完（2026-11-14） |
| B4 | ComfyUI 原生工作流 512² 出图（去噪循环在设备上，见 DESIGN D2） | |
| B5 | 性能、多尺寸档位、一键拉起、文档 | |

## A0 环境、编译链、CPU 参照、算子清单

- [x] 2026-11-14 赛方工具链装进隔离目录（`~/other/calculet-hackathon/{rustup,cargo}`，`env.sh`）：rustc 1.98.1-calculet、docc 0.4.1、dort 0.4.1。
      赛方 install.sh 原样执行（先建了隔离 CARGO_HOME 的 `bin/`，否则最后一步链 docc / dort 会失败）。
- [x] 2026-11-14 calBLAS 全部例子设备遍编通（19 个 `.cvxo`，3.9 s）：`logs/build-workspace-elf.log`。
- [x] 2026-11-14 tiny 整条本地链：`--features step` 出整步程序 ⇒ tiny-pack 打包 ⇒ `docc calbin from-calvx` ⇒ doctor 拒收 0 条（警告 3 条：输入输出没开双缓冲）。
      产物 `~/other/calculet-hackathon/work/out/tiny/`。
- [x] 2026-11-14 dort 能力核对（只看 `--help`，没开设备）：`calbin run` 一次性、裸字节模式单输入单输出、没有常驻会话与 Python 绑定 ⇒ DESIGN F1 / D2。
- [x] 2026-11-14 本仓骨架：工作区经赛方 cargo 源依赖 npu / caldnn / calblas；`tools/check_env.sh`（隔离目录、工具链自带的 docc / dort、Cargo.lock 来源）；
      `tools/calbin.sh`（设备遍 ⇒ calbin ⇒ doctor，编译与 docc 分开计时）。
- [x] 2026-11-14 冒烟 `crates/smoke`：经 cargo 源用 calblas 的 `conv2d!` + `upsample2!`（TAESD 开头那一段的形状），单芯粒两核整步程序 ⇒ calbin，doctor 拒收 0 条。
      `out/smoke/{timing.txt,doctor.txt}`。
- [x] 2026-11-14 CPU 参照环境（`ref/setup_ref_env.sh`）：`~/other/calculet-hackathon/venv`，torch 2.15.0+cpu、diffusers 0.43.0、transformers 5.23.0
      （全部版本冻结在 `ref/requirements.lock`）。权重：SD1.5 钉 revision 451f4fe（fp32 safetensors 与配置，3.97 GiB，163 s）、TAESD 614f768；
      sha256 在 `~/other/models/{stable-diffusion-v1-5,taesd}/download-verification.json`。日志 `logs/setup-ref-env.log`。
- [x] 2026-11-14 SD1.5 算子清单（`ref/op_inventory.py` ⇒ [`docs/inventory/`](docs/inventory/README.md)，512²、CPU fp32、批 1）：
      CLIP 13.3 GFLOPs / 15 种签名；UNet 一次前向 803 GFLOPs / 168 种签名（卷积 55%、线性 29%、注意力 16%）；VAE 解码 2515 GFLOPs / 45 种签名
      （是一次 UNet 前向的 3 倍多）；TAESD 141 GFLOPs / 25 种签名。对照 calBLAS：**真要新写的只有两样**——时间步嵌入（sin / cos）、带 ReLU 的卷积（TAESD）；
      其余都是现成模板或改参数（Cin 4 / Cout 3–4 补齐、注意力的掩码与头维补零、GEGLU 拆两个 linear!、quick_gelu）。日志 `logs/op-inventory.log`。
- [x] 2026-11-14 换机器准备：`ref/setup_ref_env.sh` 不再写死本机的 Python 路径（`uv` 按版本号 3.12 自取解释器），有 `requirements.lock` 就照它装
      （`REF_UPGRADE=1` 才装最新并重写 lock）；`download-verification.json` 不再把自己算进文件清单。照 lock 重跑一遍通过。
      交接文档：`~/other/calculet-hackathon/交接提示词.md`（附赛题原文与本仓全部文件）。
- [x] 2026-11-14 ComfyUI 钉 v0.45.3（提交 71096d1a），装进**单独的** venv（`$CALCULET_HOME/comfy-venv`，torch 与参照环境同版 2.15.0+cpu；
      不和参照环境混，免得顶掉 ref/requirements.lock 钉的版本），版本冻结在 `comfy/requirements.lock`：`comfy/setup_comfyui.sh`。
      自定义节点 `comfy/sd15_npu`（链进 custom_nodes，不改 ComfyUI 源码），执行器三种：cpu / npu-sim / npu（`host/sd15npu/runner.py`）。
      `--quick-test-for-ci` 加载通过。日志 `logs/setup-comfyui.log`。
- [x] 2026-11-14 整步程序的 numpy 模拟（`host/sd15npu/cvxsim.py`）：**解析编出来的 `.cvx` 入口**、按真实地址逐个 kernel 重放，
      查 64 B 对齐、越界、原地、漏屏障、只在一颗核上的屏障、按 num_cores 分行的单核 kernel 只让一颗核调、读没清零的内存（workspace 开头填 NaN）。
      自测 `ref/test_cvxsim.py`：拿真 `.cvx` 故意改坏 7 种，全部抓到。
- [x] 2026-11-14 单 kernel 测试台：`crates/ktest`（每个 feature 一个只调一个 kernel 的整步程序，kernel 特化 `#[path]` 引 TAESD 那份）+ `ref/ktest.py`
      （torch 语义另算期望；gen / sim / calbin / board）。6 个测试（conv_in、convr_64、conv_out、convr_512、tanh、up_256）× 边界用例
      （zeros、neg、big、tiny）模拟全过、6 个 calbin doctor 拒收 0 条。上板命令备好，缺省拒绝。
- [x] 2026-11-14 换机器：`tools/bootstrap.sh`（新机器从零，幂等；赛方 install.sh 先核 sha256 再执行；三个公开仓库钉提交；权重、ComfyUI 都钉版本）、
      `tools/verify_all.sh`（不上板能做的全部检查，11 项）。入口 `env.sh` 改成相对路径，`setup_ref_env.sh` 的权重目录改成跟 CALCULET_HOME 走。
      在一个全新目录（`~/other/fresh-test/calculet-hackathon`）里从零跑 bootstrap 验过，见下面「换机器实测」。

- [x] 2026-11-14 对照赛题文档「工具链」一节逐项核对：补装可选组件 rustfmt 1.9.0、clippy 0.1.98、rust-src（`bootstrap.sh` 也会装）；
      文档用的 `-elf` target 编我们的程序与 `-cvx` 出的 CalVX 逐字节相同；公开环境里除了真机没有能执行 kernel 的东西（DESIGN F16），
      calDNN 的图运算表达不了 SD 的二维卷积等（F17）。

## 上板准备（2026-11-14，负责人要求「保证这一套可以在真卡上跑起来」）

- [x] 找过能在上板前跑 kernel 的东西：calBLAS 文档里的指令级模拟器 calvxus `cvx` 在公开资料里找不到，公开环境里除了真机没有能执行 kernel 的东西（DESIGN F16）。
- [x] 本机的卡（只读 `dort device`）：设备端 OS / RT 起着，但板上跑的**不是 DeviceRT**，这一版 dort 开设备被拒（「拒」）；另有一个服务进程在用这块卡。
      要用它得重刷 DeviceRT 并和在用的服务协调——刷写不由我们做（硬规矩 3）⇒ **没往这块卡上发任何作业**。
- [x] `tools/preflight.py`（不开设备）：`dort calbin parse`（dort 自己标「纯静态，不开设备」）解析我们的 8 个 calbin，doctor 拒收 0 条；
      子模型清单里的输入 / 输出张量名、字节数与主机侧发的一致（TAESD：in_x 295936、out_y 17040384），参数块是同一份。故意改错字节数、子模型名、参数块都抓得到。
- [x] `tools/board_check.py`：真卡上一条命令跑完（重出 calbin ⇒ 静态核对 ⇒ 只读查卡：OS / RT、没被拒、没别人在用 ⇒ 7 个单 kernel 全部用例 ⇒
      TAESD 只部署一次 + 整次 3 遍，查逐字节一致、对两份参照、部署与推理分开计时 ⇒ 可选 ComfyUI），任何一步不对就停、退出 1。
      `ref/ktest.py board`、`tools/taesd.py board` 改成按门槛判过 / 不过（门槛是首轮的，上板后按实测收紧）。
- [x] `ref/fake_dort.py`（dort 的测试替身，照真 dort 的 --help 严格解析命令行，「跑」之前核 calbin 与本地 .cvxo 是同一次构建，再用模拟器算）+
      `ref/test_board_kit.py`：全套空跑 + 6 种该停的情形（没开开关、被拒、有人在用、RT 没起来、输出改坏、两次不一致、dort 报错）全过。
      空跑抓到一个真 bug：上板结果里的比较是 numpy 的 bool，写 JSON 时崩——真卡上跑完第一个用例就会崩，已修。结果另存 *-dryrun、标「测试替身」。
- [x] 2026-11-14 **真机实测**（npu3 的卡 `/dev/calculet0`，PCI 0000:07:00.0；负责人确认可以用，板锁用 `/opt/data/locks/board-0000:07:00.0.lock`）：
      `SD15NPU_ALLOW_BOARD=1 SD15NPU_BOARD_LOCK=… tools/board_check.py --comfy` **6 步全过、退出 0**，用时约 1.5 分钟。
      记录 `out/board/summary-20261114-154020.json`，日志 `$CALCULET_HOME/logs/board-check-20261114-154020.log`。
      - 第 2 步只读查卡：`OS up / RT up`、2 个芯粒、板上 DRT rt 3.1 / ABI 1.1（线协议 rev 1），dort 0.4.1「放行（按 rev 1 说）」，没有别的进程（DESIGN F20）。
      - 第 3 步 7 个单 kernel、20 个用例全过（19 个必过，tanh/tiny 只记录）（`out/ktest/<测试>/<用例>/board.json`；dort 每次 1–70 ms）：

        | 测试 | 逐位相同 | 最大差 ULP(输出最大值) | 余弦（最差） | 说明 |
        | --- | --- | --- | --- | --- |
        | up_256 | 100% | 0 | 1 | 两核按 num_cores() 分行的假设成立（DESIGN F11） |
        | conv_in（4 例） | ≥ 99.99% | 0.06 | 1.0000000 | relu 分支 |
        | convr_64（4 例） | ≥ 99.999% | 0.50 | 1.0000000 | res_relu |
        | conv_out | 99.99% | 0.50 | 1.0000000 | 512 宽快路 |
        | convr_512 | 99.999% | 1.00 | 1.0000000 | 512 宽 + 残差 |
        | tanh（3 例 + tiny） | ≥ 99.999% | 0.00（最大差 2.4e-7，最多 2 个 bf16 ULP） | 1.0000000 | tiny（1e-30 量级）板上整片出 0，只记录、不拦 |
        | temb（5 例） | 19–100% | 2.50（t999） | 0.99996 | **cos_approx 的输出全在 1/128 网格上**（DESIGN F19） |

      - 第 4 步 TAESD 整个解码器（latent = image）：3 遍逐字节一致；对 bf16 参照余弦 0.9999987、PSNR 60.3 dB（模拟 60.6），对 fp32 55.9 dB（模拟 55.9）；
        只部署 0.003 s、整次 0.111 / 0.112 / 0.112 s ⇒ 推理（含写入 0.3 MB、读回 17 MB）约 0.11 s（估算：整次中位减只部署）。图 `out/taesd-board/image/board.png`。
      - 第 5 步 ComfyUI 原生工作流（euler 20 步，CLIP / UNet 在 CPU，**TAESD 解码走 npu**）：成功，服务起来 4.0 s、整条 50.1 s；
        节点日志「backend=npu，0.12 s，dort 墙钟 0.113 s」，没有落回 CPU。图 `$CALCULET_HOME/ComfyUI/output/sd15npu_taesd_00002_.png`（旁边 VAEDecode 的对照图一致）。
- [x] 2026-11-14 按实测收紧上板门槛（各留约一倍余量，依据写在代码注释里）：`ref/ktest.py` 卷积 8 ⇒ 2 ULP(max)、余弦 0.9999 ⇒ 0.99999；
      tanh 4 ⇒ 1 ULP(max)、0.999 ⇒ 0.99999；temb 4 ⇒ 3 ULP(max)、0.999 ⇒ 0.9999（cos_approx 的分辨率卡着，换多项式前收不动）；
      `tools/taesd.py` 40 / 35 dB ⇒ 50 / 45 dB、余弦 0.9999 ⇒ 0.99995（板上只测过 image；按四组里最差的 randn:0 的模拟数留余量）。
      拿已有的真机结果核过：新门槛下必过用例全过。
- [x] 2026-11-14 TAESD 另外三组 latent 上板（**真机实测**，npu3，同一把锁；先只读查卡）：都是 3 遍逐字节一致。
      randn:0 余弦 0.9999903、对 bf16 参照 52.7 dB、对 fp32 46.8 dB；zeros 0.9999993、65.8 / 59.4 dB；big 0.9999943、56.1 / 50.7 dB（和模拟最多差 1.5 dB）；每次整次 0.11 s、只部署 0.003–0.004 s。
      结果在 `test/results/npu3-20261114/taesd_latents/`。`tools/taesd.py` 的门槛按最差的 randn:0 再收：余弦 0.99995 ⇒ 0.99998（PSNR 50 / 45 dB 不变）。

## B1 TAESD 解码上 NPU

- [x] 2026-11-14 kernel：`crates/kernels` 的 `conv2d!` 抄 calblas 0.1.0（VENDORED.md），加 `act = relu`、`act = res_relu`（残差之后 relu，Block 末尾的 relu(conv + x) 融进卷积尾处理）；
      新写 `tanh_scale!`（f32 里 tanh_approx；docc 0.4.1 收 f32 的 `q.tanh.approx`）。
- [x] 2026-11-14 整步程序 `crates/taesd`：单芯粒两核，41 个作业（清零 2、tanh 1、卷积 35、上采样 3）、40 条屏障；`layout.rs` 的偏移表设备与主机共用
      （`crates/layout-dump` ⇒ JSON）。@act 137 MB（每个分辨率 a / b / c 三张图，入口整块清零 ⇒ 边框全程是 0）、@w 2.6 MB、输入 0.3 MB、输出 17 MB。
- [x] 2026-11-14 打包：`host/sd15npu/taesd.py`（权重 `[9][Cin][Cout]` bf16 + bias f32，Cin 4 ⇒ 32、Cout 3 ⇒ 32 补 0）；参数块逐字节可复现
      （sha256 `ed33d702…`）。带真权重的 calbin `out/taesd-real/cal`：docc 4.7 s，doctor 拒收 0 条（警告 2 条：输入输出没开双缓冲，预期内）。
- [x] 2026-11-14 独立参照 `ref/taesd_ref.py`（照 diffusers 模块结构逐层走、每个 kernel 出口舍 bf16；另出 diffusers fp32）。模拟对参照（`tools/taesd.py check`）：

      | latent | 对 bf16 参照 余弦 / 最大差 / PSNR | 对 fp32 PSNR |
      | --- | --- | --- |
      | image（合成测试图经 TAESD 编码器） | 0.999999 / 0.0117 / 60.6 dB | 55.9 dB |
      | randn:0 | — / — / 52.7 dB | 46.8 dB |
      | zeros | 1.000000 / 0.0078 / 67.3 dB | 59.4 dB |
      | big（±8，过 tanh 饱和段） | 0.999994 / 0.0195 / 56.2 dB | 50.7 dB |

      合成测试图来回一趟（TAESD 编码 ⇒ 解码）PSNR 27.3 dB，fp32 与模拟一样（TAESD 本身的画质）。图在 `out/taesd-sim/image/`。
- [x] 2026-11-14 ComfyUI 原生工作流 `comfy/workflows/sd15_taesd_npu.api.json`：DiffusersLoader → CLIPTextEncode ×2 → KSampler（euler 20 步，CPU）
      → **SD15 NPU · TAESD Decode（npu-sim）** → SaveImage，旁边一路 ComfyUI 自带 VAEDecode 作对照；无界面跑通（`comfy/run_workflow.py`，84 s），两张图一致。
- [x] 2026-11-14 **上板（真机实测，npu3）**：`tools/board_check.py --comfy` 全过（明细见上一节）。TAESD 512²：3 遍逐字节一致，对 bf16 参照 60.3 dB、
      对 fp32 55.9 dB；只部署 0.003 s，推理约 0.11 s（估算）；ComfyUI 原生工作流解码走 npu、没有落回 CPU。
### B1b 正式 VAE 解码器（2026-11-14 定的计划；不上板的部分都在 CPU 上做，上板要负责人点头）

2026-11-14 在 `wip/vae-decoder` 分支上做，真卡上验过之后合进 `main`。

为什么先做它：画质比 TAESD 高一大截（ComfyUI 里直接看得到）；它的卷积、GroupNorm、ResNet、4096 token 的注意力，
UNet（B3）几乎全要用 ⇒ 先在一个「只跑一次前向、好对 diffusers」的模型上把这些做对。结构：post_quant_conv（1×1）⇒ conv_in（4→512）@64²
⇒ 中间块（ResNet、注意力、ResNet）⇒ 4 个上采样段（各 3 个 ResNet，通道 512 / 512 / 256 / 128，前三段末尾 ×2 上采样 + 卷积）⇒ GN+SiLU ⇒ conv_out（128→3）。
2.5 TFLOPs / 张，98% 是卷积。

- [x] 2026-11-14 V0 可行性探针（DESIGN F21）：要用的卷积、`group_norm!`、排布转换、512 通道上采样 docc 都收；**`attention!` 头维 512 不收** ⇒ D14 新写分段的注意力。
      GroupNorm 先走排布转换 + calblas 现成的（D13）。
- [x] 2026-11-14 V1 独立参照 `ref/vae_ref.py`：diffusers `AutoencoderKL.decode` 的 fp32；同一结构按 kernel 出口舍 bf16 的参照（GN、卷积、注意力的切法照 D13 / D14；
      conv_out 折进 (x + 1) / 2，输出直接是 [0, 1]）。`python -I ref/vae_ref.py`：bf16 参照对 fp32 —— image 57.1 dB、randn:0 53.0、zeros 57.5、big 57.2 dB
      （余弦都 ≥ 0.99999）；测试图来回一趟 fp32 31.6 dB（TAESD 是 27.3 dB）。CPU 上一次 fp32 约 1.9 s、bf16 参照约 2.4 s。
- [x] 2026-11-14 V2a `sd_kernels::attn_wide!`（`crates/kernels/src/attn.rs`，D14；VENDORED.md 记了出处）：头维按 128 一段，分数整行存 VB、只算一遍。
      探针（rows 4096、单头、512 维）设备遍 + docc 1.5 s，doctor 拒收 0 条。**还没有数值验证**（下一条的测试与模拟器）。
- [x] 2026-11-14 V3 整步程序 `crates/vae` + 打包 `host/sd15npu/vae.py`。整个解码器编成一个程序时 docc 报跳转超距：45 个 kernel 约 2.7 MB，
      超过每核约 1 MB（DESIGN F22）⇒ **切成 6 段**（D15，feature `s1`…`s6`，主机把上一段的 out_y 原样当下一段的 in_x）。
      128² × 512 的 calblas `group_norm!` 单个就约 1.1 MB ⇒ 本仓抄一份加 `parts = N`，切 64 段之后 232 KB（VENDORED.md）。
      6 段每核 ELF 559–910 KB，带真权重的 calbin docc 合计约 50 s，doctor 都是拒收 0 条；6 段参数块合计 314 MB。
- [x] 2026-11-14 V4 模拟器认 group_norm / 排布转换 / 转置 / linear / attn_wide；`tools/vae.py sim` 6 段串起来，对两份独立参照（对 bf16 参照 / 对 fp32）：
      image 59.8 / 57.1 dB、randn:0 54.2 / 53.0、zeros 55.2 / 57.9、big 61.3 / 57.2 dB；6 段 150 个作业、numpy 约 25 s。
      `preflight.py` 认 6 段（14 个 calbin）；`verify_all.sh` 加了 VAE 的 calbin、模拟、ComfyUI 三项（VERIFY_FULL 16 项全过）。
- [x] 2026-11-14 V5 ComfyUI 节点 **SD15 NPU · VAE Decode**（`runner.VaeDecoder`，npu / npu-sim / cpu；LATENT 原样喂，不乘 0.18215）+ 工作流
      `comfy/workflows/sd15_vae_npu.api.json`。npu-sim 出的图和 ComfyUI 自带 VAEDecode 相比 55.7 dB。`board_check.py --vae` 用测试替身空跑通过。
- [x] 2026-11-14 V6 **上板（真机实测，npu3，负责人点头）**：`board_check.py --no-build --tests "" --no-taesd --vae --comfy` 全过。
      正式 VAE：3 遍逐字节一致，对 bf16 参照余弦 0.999998、**58.3 dB**，对 fp32 56.9 dB；6 段整次之和中位数 **5.18 s**（s1 0.05、s2 0.10、s3 0.24、s4 0.98、s5 1.44、s6 2.36 s），
      只部署合计 0.12 s。ComfyUI 正式 VAE 工作流成功，节点 5.89 s，出的图和自带 VAEDecode 相比 **55.6 dB、最大差 5/255**。
      结果在 `test/results/npu3-20261114/vae/`，性能在 `test/PERF.md`。`tools/vae.py` 的门槛按实测收紧到 50 / 48 dB、余弦 0.99998。
      **比同机 CPU fp32（1.71 s）慢约 3 倍**：算对了，但还没优化（DESIGN F23）。
- [ ] V2b 单 kernel 测试：`crates/ktest` 已加 `attn_wide`、`gn_128` 两个程序（编得过），`ref/ktest.py` 的期望值与用例还没写；
      之后再补原路卷积（128² 512→512）、1×1 shortcut、`linear!` 多行、512 通道上采样。整段算错时用来二分（这次第一遍就对了，没用上）。
- [x] 2026-11-14 V7 性能（**真机实测**，npu3）：先做逐 kernel 计时 `tools/prof.py`（`crates/prof` 每个 kernel 一个程序、卡上就地连调 4 遍）——
      大头不是卷积，是 GroupNorm 前后的**排布转换**（约 1.3 GB/s，合计约 2.7 s，DESIGN F24）。按测出来的次序改：
      ① 卷积的**列块快路**（`sd_kernels::conv2d!`：快路放不下整行就按列切块）：128² 以上的 3×3 卷积 1.75–2.6 ⇒ 4.2–4.6 TFLOPS，整条 5.18 ⇒ 4.51 s；
      ② **NHWC 原生 GroupNorm**（`gn_stats!` / `gn_apply!`，D16）：不转排布、w / b 不展开（参数 314 ⇒ 96 MB），一个 GN 从 15–487 ms 降到 1.6–26.5 ms，整条 ⇒ 1.23 s；
      ③ **并成 3 段**（D17：s3–s6 合成一段，约 540 KB 代码）：大特征图不再经主机转手，整条 ⇒ **0.98 s**。
      每一步都对参照（58.3 / 56.9 dB）、3 遍逐字节一致。**比同机 CPU fp32（1.71 s）快约 1.75 倍**；ComfyUI 的 VAE 节点 5.89 ⇒ 0.99 s。
      结果在 `test/results/npu3-20261114/vae-opt/`（含三份逐 kernel 计时），性能表在 `test/PERF.md`。
- [ ] V8 还能抠的：GroupNorm 统计那一遍（512² 一次约 20 ms，3.5–6.5 GB/s）、`attn_wide!`（0.51 TFLOPS）、把注意力那一段也并进去（要先把它的 GN 换成 NHWC 版）。

## B3 UNet 一次前向上 NPU（2026-11-14 开始；计划在 DESIGN D18 / D19）

- [x] 2026-11-14 U0 可行性探针（DESIGN F25）：UNet 要用的卷积（35 种形状）、带边框 token 的注意力（自 / 交叉，4 种分辨率）、上下文 k / v 的 `linear!`、
      LayerNorm、宽通道的 GroupNorm、GEGLU 全部编得过。为此：修了列块快路在 W < 64 时除以 0 的 bug（TAESD、VAE 编出来逐字节不变）；
      NHWC GroupNorm 改成通道按 64 一段（支持 320–2560 通道；VAE 换上之后**真机重测**：58.3 / 56.9 dB、3 遍逐字节一致、0.98 s，和原来一样）；
      新写 `ln_nhwc!`（NHWC LayerNorm）与 `geglu!`（GEGLU 的后半），都还没上卡。
- [x] 2026-11-14 U1 独立参照 `ref/unet_ref.py`：diffusers fp32 与按 D18 的切法逐 kernel 舍 bf16 的参照（时间步投影折进 conv1 的 bias、投影当 1×1 卷积、GEGLU 拆开）。
      `python -I ref/unet_ref.py`：bf16 参照对 fp32 —— t = 999、500.5 时余弦 ≥ 0.999996、信噪比 51–52 dB，t = 1 时 41 dB（有提示词、空提示词一样）；CPU 上一次 fp32 约 1 s。
- [ ] U2 时间步那一路：`timestep_embed!`（已有）⇒ `linear!` 320（补到 384）→ 1280 ⇒ SiLU ⇒ 1280 → 1280 ⇒ SiLU ⇒ 全部 ResNet 的 time_emb_proj 一次算完 ⇒ 加进各 conv1 的 f32 bias（小 kernel）。
- [ ] U3 偏移表与切段：先按探针的代码量估每段 < 1 MB（约 6–8 段），跳连的图按通道窗口放、跨段经主机转手。
- [ ] U4 整步程序 `crates/unet` + 打包 `host/sd15npu/unet.py` + 模拟器认新模板（`ln_nhwc`、`geglu`、`attention`）⇒ 模拟对两份参照。
- [ ] U5 上板（负责人点头）：先单 kernel（ln_nhwc、geglu、带边框 token 的注意力），再整个 UNet，量每步时间。
- [ ] U6 ComfyUI：自定义 MODEL（`apply_model` 调 NPU 的 UNet，每步一次、CFG 两路同一次调用）⇒ 原生 KSampler 照用；再看要不要把去噪循环放进设备（D2 / D19）。

之后：B2 CLIP（LLM 式模板现成：`layer_norm!`、`linear!`、`attention!(mask = full)`；新写 quick_gelu）⇒ B3 UNet 一次前向 ⇒ B4 去噪循环进设备（D2）⇒ B5 性能与多尺寸。

## A1 新写的算子

- [x] 2026-11-14 时间步嵌入 `sd_kernels::timestep_embed!`（DESIGN D7）：t（可以是小数）· 频率表 ⇒ Cody–Waite 两段 2π 规约 ⇒ `cos_approx`，
      sin 那一半走 cos(a − π/2)，cos、sin 并进 3 整行一起算（docc 的两条限制，DESIGN F14 / F15）。常数表 `host/sd15npu/temb.py`。
      测试 `ref/ktest.py temb`：t = 999、0、1、451.37、0.25，期望是 diffusers `get_timestep_embedding` 的 f32 舍 bf16；模拟全过，calbin doctor 拒收 0 条。
      上板要标定 `cos_approx` 的误差。
- [x] 2026-11-14 时间步嵌入上板标定（**真机实测**，npu3）：5 个用例全过；`cos_approx` 的输出全部落在 1/128 的绝对网格上（多半往 0 截），
      最大差 2.5/128 ≈ 0.0195（t999）、余弦最差 0.99996，t0 逐位相同（DESIGN F19）。比 bf16 在 0.5 附近的分辨率粗约 10 倍：
      B3 接 UNet 时对 diffusers 看端到端影响，不够就换 f32 多项式（改之前先和负责人说）。
- [ ] quick_gelu（CLIP）、NHWC 原生的 GroupNorm+SiLU（VAE / UNet）……

## 换机器实测

- [x] 2026-11-14 在全新目录 `~/other/fresh-test/calculet-hackathon` 里（隔离的 RUSTUP_HOME / CARGO_HOME 是空的、没有 venv、没有权重）从零跑 `tools/bootstrap.sh`：
      赛方 install.sh 核 sha256 后执行、三个公开仓库切到钉住的提交、tiny 设备遍、参照环境 + 权重（3.97 GiB，151 s）、ComfyUI、`verify_all.sh`（VERIFY_FULL=1）
      **11 项全过**，从装工具链到自检完约 6 分钟。新下的 14 个权重文件 sha256 与原机器逐个相同；ComfyUI 的依赖版本与 lock 相同；
      TAESD 参数块 sha256 与原机器相同（`ed33d702…`）。日志 `logs/bootstrap-fresh-20261114.log`。
      没测到的：本机已有 rustup 与 uv，所以「没有 rustup / uv 时自动装」那两步没走到（走的是官方安装器，装进隔离目录）。
- [x] 2026-11-14 交接到 `sweet@npu3`（192.168.10.109）的 `/home/sweet/hachson`：仓库 + env.sh + 赛题原文 + 参赛建议 + 交接提示词 + tar 包 + 原机器的证据日志
      （`logs/原机器-20261114/`），传完逐文件核 sha256。那台机器的情况（只读摸过）与边界写在 `docs/HANDOFF.md` 第 0 节：系统里有老版本 dort 0.2.8 / docc 0.1.4、
      卡是共享的（锁文件按 PCI 地址起名，卡在 0000:07:00.0）、bootstrap 要的站点都通。**还没在 npu3 上跑 bootstrap，也没碰它的卡。**
- [x] 2026-11-14 npu3 上从零跑 `tools/bootstrap.sh`（这次 rustup 是新装进隔离目录的，uv 用系统的 `/usr/bin/uv`）：赛方 install.sh 核过 sha256、
      三个公开仓库切到钉住的提交、参照环境 + 权重（`/home/sweet/models`）、ComfyUI，从开始到自检完约 10 分钟（15:25–15:36）；`verify_all.sh` 12 项全过，`VERIFY_FULL=1` 13 项全过。
      TAESD 参数块 sha256 `ed33d702…4955` 与原机器相同；`taesd.py check` 的 image 组对 bf16 参照 60.6 dB、对 fp32 55.9 dB，与原机器相同。
      PATH 上最前面是工具链自带的 docc / dort 0.4.1（系统 `/usr/local/bin` 里的老版本没用到）。日志 `$CALCULET_HOME/logs/bootstrap-npu3-20261114-152526.log`、
      `verify-20261114-153614.log`。然后上了卡，见「上板准备」。
- [x] 2026-11-14 推到 GitHub（https://github.com/liyingxxxxx/calculate-hackson ，公开，负责人确认）：`main` = 真卡上验过的稳定版，`wip/vae-decoder` = 正式 VAE 进行中。为了能在比赛提供的真卡环境里照 README 复现：去掉了文档与脚本里对非公开设备栈的点名，`check_env.sh` 改成白名单（docc / dort 必须是工具链自带的、Cargo.lock 只许赛方源与 crates.io；拿 npu3 上的老版本 dort / docc 验过会报）；赛题原文收进 `docs/赛题原文.md`；README 的复现步骤改成 `git clone … sd15-npu` ⇒ `bootstrap.sh` ⇒ `test/run_example.sh` ⇒ `board_check.py`；交接文档 `docs/HANDOFF.md` 重写。
- [x] 2026-11-14 **从 GitHub 克隆复现（npu3，真机实测）**：在全新目录（`/home/sweet/fresh-clone/calculet`，干净环境 `env -i`，PATH 里故意留着 `/usr/local/bin` 的老版本 dort / docc）`git clone https://github.com/liyingxxxxx/calculate-hackson.git sd15-npu`（main e5280f7） ⇒ `bootstrap.sh`：约 6 分钟，`verify_all.sh` 12 项全过；TAESD 参数块 sha256 `ed33d702…` 相同，14 个权重文件的核对清单逐字相同。再 `SD15NPU_ALLOW_BOARD=1 SD15NPU_BOARD_LOCK=… bash test/run_example.sh --comfy`：只读查卡通过 ⇒ TAESD 3 遍逐字节一致、60.3 / 55.9 dB ⇒ ComfyUI 工作流成功、节点 `backend=npu`；出的三张图（TAESD、ComfyUI 的 NPU 解码与 CPU 对照）和 `test/results/npu3-20261114/` 里存的**逐字节相同**。
- [x] 2026-11-14 真卡性能整理进仓库：`test/PERF.md` + `test/results/npu3-20261114/perf.json`（**真机实测**，两次上卡的全部计时）。TAESD 512² 整次墙钟中位数 0.113 s（15 遍，0.111–0.115 s），只部署 0.003 s，推理约 0.11 s（估算，折合约 1.29 TFLOPS）；同机 CPU（Ryzen 9 9955HX、torch fp32 16 线程）0.185 s ⇒ 约 1.6 倍。一次 dort 调用的固定开销约 1.3 ms。ComfyUI 整条约 50 s，NPU 只占 0.12 s。
- [x] 2026-11-14 `test/`：简单示例 `run_example.sh`（有卡先只读查卡再跑 TAESD，没卡走 npu-sim；`--comfy` 再跑 ComfyUI）+ npu3 真卡那次的原始结果 `results/npu3-20261114/`（图、指标、dort 日志、20 个单 kernel 用例）。没卡的那条路跑过：image 60.6 / 55.9 dB。

## 踩过的坑

- 原机器平时的 shell 设着别的工具链的环境变量：check_env 第一次就抓到，`env.sh` 统一隔离（现在 check_env 改成只认工具链自带的 docc / dort、Cargo.lock 只许赛方源与 crates.io）。
- `docc calbin from-calvx --no-params` 不往 calbin 里写参数块，doctor 会拒「清单点名了 param_blk 但盘上没有」：`tools/calbin.sh` 在只验结构时补同长的全零稀疏文件，
  并在 timing.txt 里写明「只验结构，不能拿去上板算数」。
- `cargo run -p tiny-pack` 要加 `--bin tiny-pack`（同一个包里还有 tiny-ref）。
- 清单脚本里模块的 forward hook 返回了 `stack.pop()` 的值：PyTorch 拿非 None 的返回值顶替模块输出，CLIP 拿到一个字符串当隐状态。钩子一律返回 None。
- 初版参赛建议误写「erf-GELU 只有 MiniCPM 例子里有」：其实 calblas 核心库 `frag.rs` 有 `act!(gelu_erf)`，`linear!` 能带 `act = gelu_erf`（已更正）。
- **cargo 切回编过的 feature 组合不重编，而设备遍产物 `npu/<crate>.cvx` 只有一个文件名** ⇒ 留着的是上一次编的那个组合的程序（ktest 的 tanh 测试拿到了 convr_512 的程序）。
  带 feature 的一律用自己的 `--target-dir`（`tools/calbin.sh`、`ref/ktest.py` 都改了）。
- 冒烟程序原来把 `upsample2!` 包在 `if core_rank() == 0` 里：它按 `num_cores()` 分行，num_cores 是 2 时只算一半的行。yolo26 上板那份是两核都调 ⇒ 改成两核都调，
  模拟器把「只让一颗核调」当错报。
- 宏参数和宏里的 `const` 同名会自引用（`upsample2!(…, c = C, …)` 展开成 `const C: usize = C;`）：传字面量。
- ComfyUI 的 LATENT 在 VAE 原空间（KSampler 出口 ÷ 0.18215），TAESD（diffusers 格式）吃 UNet 空间 ⇒ 节点里先 × 0.18215（ComfyUI 自己加载 taesd 也是 vae_scale = 0.18215）。
- 节点里的 `print` 在服务被 SIGTERM 时没刷出来：改用 logging。
