# 简单VLA-RL RoboTwin2`lift_pot` 复现与排障手册

## 1. 结论先行

该项目目前在L40已完成`RoboTwin2 / lift_pot` 的可解释闭环验证：

- 当前应使用作者 SimpleVLA-RL 提交`462f23d` 的 `twin2_lift_pot_sft1000_rl1000.sh` 作为实验契约来源。
- 作者脚本的 RoboTwin 配置是 Piper 双臂`demo_randomized.yml`不是 ALOHA-AgileX；检查点名中的`aloha`表示 OpenVLA 平台的常量，即`25`动作块`14`动作/本体感受维度。
- 公开的 SFT 检查点在验证有效的评估种子上会暂时出现，然后消失；这是策略能力基准，而不是环境或评估器故障。
- 公开的 RL 参考检查点在同一环境中、同一输入和同一二进制成功判定下成功。已在`100100007`、`100100013`、`100100017`三个有效的评估种子得到`3/3` 成功，均在第 `100`达成。
- 因此 `RoboTwin2 -> SimpleVLA wrapper -> checkpoint -> action -> binary reward`链路已正确。正式复现实验应从公开的SFT checkpoint运行作者原始的binary-reward RL，而非继续修改评估器。

此处的 `3/3`是正确性oracle，不是论文规模成功率。论文/作者脚本的完整评测仍是其种子表和仅验证设置下的批量统计。

## 2. 当前可用全链路

```text
作者实验脚本
  `examples/twin2_lift_pot_sft1000_rl1000.sh`
        |
        | 任务：robotwin2_lift_pot
        | 配置：demo_randomized / Piper 双臂
        | 控制：14D absolute qpos + normalized gripper，25-step chunk
        v
RoboTwin2 `lift_pot`
        |
        | head RGB 240x320，joint_action.vector 14D
        v
RobHFRollout.process_input
        |
        | center crop, prompt, processor, proprio bounds normalization
        v
OpenVLA-OFT / SimpleVLA checkpoint
        |
        | generate_action_verl(do_sample=True, temperature=1.6)
        v
action (25, 14)
        |
        | RoboTwin `take_action()`
        v
binary `eval_success` / `check_success`
```

关键文件：

- 作者实验参数：[例子/双胞胎2_抬起_抱歉，您提供的文本“pot”是英文单词，没有提供足够的上下文信息来确定其确切含义。请提供更多上下文或具体问题，以便我能够更准确地帮助您翻译。_SFT1000_rl1000.sh](../examples/twin2_lift_pot_sft1000_rl1000.sh)
- 发布与RoboTwin接口：[版本发布/部署/滚动_rollout.py](../verl/workers/rollout/rob_rollout.py)
- 作者评估种子表：[机器人窗口2的种子/机器人窗口2_评估_种子.json](../verl/utils/envs/robotwin2/seeds/robotwin2_eval_seeds.json)
- 过程事实记录：[执行.md](execution.md)
- 本次诊断脚本：[简单变量_发布_合约_评估.py](../simplevla_rollout_contract_eval.py)、[机器人赢_纯净_专家_探针.py](../robotwin_pristine_expert_probe.py)

## 3. 复现时固定的口径

### 3.1 代码与环境

在L40上使用隔离目录，避免污染用户已有的工作树：

| 项目 | 固定对象 |
|---|---|
|SimpleVLA-RL| `/public/home/s2250018/wsy/VLA/SimpleVLA-RL-462f23d`，提交`462f23d91b0190694dd5ede12860c9c1a4f12ed1` |
|RoboTwin 基础|提交`6be6922d74f9b729837d6bcaa703721ec4553779` |
|RoboTwin 实验涵盖|上述 SimpleVLA 提交的`modified_codes/robotwin2`，覆盖到 `/public/home/s2250018/wsy/VLA/RoboTwin-462f23d` |
| 资产 | 只链接既有 `embodiments`、`objects`、`background_texture`，不复制、不改原资产 |
|conda 环境| `robotwin-pristine`，Python 3.10、PyTorch`2.4.1+cu124`、库罗博`0.7.5`，为了L40`sm_89` 编译 |
|Transformer|作者 fork`4.40.1`，`tokenizers==0.19.1` |

所有的 Python 运行必须设置：

```bash
export PYTHONNOUSERSITE=1
```

原因：用户目录的 `transformers==4.47.1` 与 `tokenizers==0.21.4` 会优先遮蔽作者依赖。它不一定单独导致 0%，但会让结果失去作者依赖口径。

### 3.2 里程碑 约定

SFT检查点：

```text
/public/home/s2250018/wsy/VLA/checkpoints/lift_pot_sft_eval_l40
```

- 四个安全tensor分片。
- `proprio_projector--20000_checkpoint.pt` 独立保存，加载模型后必须显式调用 `load_proprio_projector_weights()`。
- 该投影仪与文件内容比较，通过；BF16下最大绝对误差为`0.003692`。

RL 参考检查点：

```text
/public/home/s2250018/wsy/VLA/checkpoints/lift_pot_rl_reference_original
```

- 七个安全tensor分片。
- proprio 投影仪 已在 safetensor 内；不能要求额外 投影仪 文件。
- 与 SFT 同名的参数不相等，说明确实包含了 RL 更新；不是同一检查点的重命名副本。

两者共同条件：

- `lift_pot_1k` 统计键。
- 14D行动、14D本体感受。
- 二十五个动作块。
- 模型动作输出`(25, 14)` 且均为有限数。

### 3.3 种子与成功率口径

不要随意用 `100000`、`100063`等种子作为只用于验证的分母。

- `100063`这是用于调试的种子，而不是仅用于验证的数据样本。
- 作者评估种子表起始为`100100004`、`100100007` 等。
- `setup_demo()`必须与作者的 wrapper 保持一致：`now_ep_num=trial_id`，`seed=trial_seed`在单个种子探针中，两者都传递相同的种子。
- 先执行专家 `play_once()`；仅 `plan_success=True` 且 `check_success=True`只有种子才能作为“环境可解”的模型诊断样本。

已测前 8 个作者评估种子：

|种子| 专家结果 | 用途 |
|---|---:|---|
| `100100004` | `check_success=True`，但 `plan_success=False` | 不作稳定模型分母 |
| `100100007` | 成功 |RL 成功|
| `100100008` | 失败 | 不作模型失败计数 |
| `100100013` | 成功 |RL 成功|
| `100100016` | 失败 | 不作模型失败计数 |
| `100100017` | 成功 |RL 成功|
| `100100018` | 成功 |尚未测 RL|
| `100100022` | 成功 |尚未测量RL|

这一步非常重要：环境的专家失败和模型失败是两种不同事件。混在一起统计，会把 planner 或随机初始状态问题误判为模型能力问题。

## 4. 已检查内容、结果与排除方法

### 4.1 环境层：RoboTwin 能否正确执行？

检查：

1. 创建一个孤立的RoboTwin工作树，以避免本地更改污染已有的服务器工作树中的约36个本地更改。
2. 复用资产，并解析配置中的 `${ASSETS_PATH}`。
3. 在 L40 上为`sm_89`重新编译 CuRobo。旧二进制面向`sm_90`，会报 `no kernel image is available for execution on the device`。
4. 用作者 Seed 跑专家`play_once()`。
5. 使用“初始 qpos 重复 25 次”调用`take_action()`确认 action 接口没有出现异常漂移。

结果：

- Piper 环境可以稳定创建，输出`240x320x3`头RGB与14D`joint_action.vector`。
- 身份动作后，qpos的最大漂移约为`5e-4`，没有双解归一化、形状或控制接口错误。
- 多个作者评估种子的专家成功。环境层可用。

### 4.2 配置与实现：到底选 Piper 还是 ALOHA？

错误：检查点名称中`aloha`不能直接推出 RoboTwin 物理体现是`aloha-agilex`。

方法：读取作者提交的原始实验脚本和`modified_codes/robotwin2/task_config/demo_randomized.yml`。该配置明确为：

```yaml
embodiment: [piper, piper, 0.6]
```

结果：本任务使用了 Piper 双臂。`ROBOT_PLATFORM=ALOHA`只控制 OpenVLA 的 action 常量（25 段、14D）。之前混用 ALOHA-AgileX 配置得到的 0% 不是作者实验口径，不能报告。

### 4.3 依赖层：evaluator、dlimp、Transformers 是否正确？

曾遇到的问题：

- 在 Conda 中克隆的 pip 内部不匹配。`python -m pip` 报 `ImportError: cannot import name 'get_runnable_pip'`。
- `dlimp`在 GitHub 上使用 TLS/HTTP 下载时遇到错误。
- `tensorflow_datasets`、`tensorflow_graphics`、`tensorflow_metadata`缺少依赖链，或者与 TensorFlow 2.15 的 Protobuf 不兼容。
- 用户级较新版 Transformers 包含作者的 fork。

处理：

- 不改原 `simplevla`环境，只修隔离环境的 pip。
- 经 `codeload.github.com`下载 dlimp 归档文件，然后进行可编辑安装。
- 保持 TensorFlow`2.15.0`，用 `--no-deps`补齐兼容的 TFDS/TF Graphics 依赖，避免 resolver 下载 TensorFlow 2.21。
- 安装作者的 Transformers fork，并强制`PYTHONNOUSERSITE=1`。

结论：dlimp 是 OpenVLA-OFT 数据管道/图像随机裁剪依赖，不是 RoboTwin SFT checkpoint 0% 的根本原因。它和 evaluator 依赖缺失会导致 import 或输入处理失败，不能用“安装成功”替代真实 rollout 验证。

### 4.4 检查点 是否完整、 proprio 是否正确？

检查：

1. 读取 `dataset_statistics.json`，确认 `lift_pot_1k`与 14D 动作/本体感受。
2. 加载所有分片，检查缺失权重。
3. SFT 的独立投影仪与磁盘状态_字典对比。
4. RL 的投影器必须从 safetensors 加载，不能使用 SFT 专用加载规则。
5. 检查输出的形状、有限性和范围。

结果：检查点文件和投影仪均正常；SFT 和 RL 权重确实不同。不是“下载坏了”、不是投影仪未加载、不是 7D/14D 混用。

### 4.5 评估器/部署是否与训练时的 SimpleVLA 包裹器一致？

最危险的错误是“自己写的探针看起来是对的，但和正式评估者不同”。本次逐项对比：

- 提示词ID：相同。
- 注意力掩码：相同。
- 中心裁剪后的图像张量：相同。
- proprio：相同。
- 正确的精度路径：处理器图像保持FP32，在`torch.autocast(device_type="cuda", dtype=torch.bfloat16)` 内推理。

早期的探针将图像提前转换为BF16，并且在自动裁剪之外进行推理；已经修正。修正后仍然得到相同的SFT失败，因此精度路径不是根本原因。

解码检查：

- 贪婪模式下，`generate_action_verl()` 与 `predict_action()` 输出完全一致。
- 在采样模式下，它们的不同是预期的行为，不能将差异视为bug。
- 作者正式设置是 `do_sample=True, temperature=1.6`。

### 4.6 二进制奖励是否真的可以触发？

这是最终的判断实验。保持环境、种子、图像、提示、 proprio、解码器和动作执行器不变，只更换检查点：

|检查点| `100100007` | 轨迹现象 |
|---|---:|---|
|SFT| 失败 | 靠近、闭夹爪、短暂抬起后掉锅 |
|RL 参考| 成功 |第 100 步 达到`eval_success=True` 且 `check_success=True` |

再在`100100013`、`100100017`成功。因此二进制奖励、RoboTwin成功判断和当前评估者得到直接正例，排除了“测试永远为0”的假设。

## 5. 失败尝试：为什么失败，如何避免

### 5.1 将 OpenVLA-OFT 官方仓库作为 RoboTwin 评估器

OpenVLA-OFT 官方仓库支持 LIBERO 仿真和实体 ALOHA；不提供 RoboTwin`lift_pot`评估器。RoboTwin 接入是 SimpleVLA/RoboTwin 的附加链接。

教训：首先明确“谁拥有环境、谁拥有模型、谁拥有adapter”的界限。不要将跨仓库集成的错误归咎于一个仓库。

### 5.2 使用 ALOHA-AgileX 或修改后的 RoboTwin 工作树进行评估

服务器已更改了带有 RoboTwin 工作树的设置。在使用它或 ALOHA 配置时，尽管 14D 形状相同，但关节限制、初始状态、规划器和相机配置可能会有所不同。

教训：形状对齐不是语义对齐。每次重现必须同时锁定代码提交、配置文件、资产版本和种子生成方法。

### 5.3 无效种子直接计入0%

`100000`或专家规划失败的种子不能用于模型的成功率。环境不一定能为每一个随机种子生成可执行的专家轨迹。

教训：先运行专家，再运行模型；分别记录 environment-invalid、model-fail、model-success。

### 5.4 只看“模型加载成功”

模型可以加载并正确处理形状，但仍可能存在以下问题：投影器未加载、归一化键错误、图像精度路径不同或动作语义错误。

教训：加载检查只能证明文件可读。必须继续进行投影状态对照、身份动作、包装输入等价性测试以及正例奖励测试。

### 5.5 直接做 16/256/1000 回合长跑

早期的 ALOHA 随机化会反复碰撞并重新采样；长跑既慢又无法解释 0%。

经验：首先构建最小判别实验：一个专家成功的种子、一个模型的检查点、一个200步的滚动。只有它正确，再扩大样本。

### 5.6 在 H800 上修改环境

H800 路径曾有 CuRobo/Warp 预编译二进制问题，并且用户后来明确 H800 正在被占用，禁止操作。

经验：环境调试优先，L40；训练才使用H800。不要为了解决一个import/ABI问题而占用正式的训练资源。

## 6. 建议的可复用排障顺序

任何新的模型改动或新任务，都按下面顺序走；不要跳步。

1. **写实验契约。**固定 Git 代码提交、任务配置、实现、相机、检查点 SHA、依赖版本、种子表和成功判定。
2. **隔离工作区。**新建工作树/conda环境；资产只链接；不修改用户原始目录和公开检查点。
3. **验证环境。**单个种子设置，检查观察键、图像形状和状态维度；执行25步身份动作。
4. **验证专家。**在候选评估种子上运行`play_once()`；只将专家成功种子放入模型诊断集。
5. **验证 checkpoint。**分片、统计键、动作/本体维度、投影权重、模型逻辑文件、输出有限/范围。
6. **验证输入完全等价。**将探针的令牌、掩码、像素和本体与正式包装器逐元素进行比较。
7. **验证操作。** 确认 `(chunk, dim)`，确认绝对/增量动作语义，检查动作执行者不报错。
8. **做正例对照。**在相同的种子上运行已知成功的检查点。没有正例时，0% 无法解释。
9. **做负例对照。**在相同条件下从起点开始比赛，区分策略能力与系统故障。
10. **再批量评测。**记录每个episode的环境有效性、计划、动作错误、奖励、终止步骤。只对同类有效的episode计算成功率。

一个通用诊断记录至少包含：

```text
commit / config hash / checkpoint hash / dependency versions
seed / now_ep_num / expert plan-success / expert check-success
observation keys and shapes / stats key / projector source
prompt / do_sample / temperature / first action shape and range
action execution error or none / eval_success / check_success / finish step
```

## 7. 正式复现的下一步

用户在使用 H800 后，按照作者`twin2_lift_pot_sft1000_rl1000.sh`在独立作者工作树启动：

### 7.1 H800 训练代理应该读取什么

训练代理先读本文件的第 3、6、7 节，再读 `execution.md` 的 `2026-09-17` 条目。

- **本文件**这是规范：固定提交、RoboTwin覆盖、Piper配置、检查点契约、种子和训练参数；根据它修改环境并启动训练。
- **`execution.md`** 是历史：记录机器路径、过去失败原因、已有隔离目录和不可碰对象；用于避免重复错误，不应从旧命令直接复制配置。
- **作者脚本**是参数原文：`examples/twin2_lift_pot_sft1000_rl1000.sh`如果本文件与脚本冲突，请以作者的提交中脚本为准，并记录差异。

启动前不可省略的检查：

1. 创建一个新的 H800 专用 conda 环境和一个新的 SimpleVLA/RoboTwin 工作树；不得修改用户原始工作树、公开的 SFT 或 RL 参考检查点。
2. SimpleVLA 固定`462f23d91b0190694dd5ede12860c9c1a4f12ed1`罗布双基础固定`6be6922d74f9b729837d6bcaa703721ec4553779`，仅叠加该 SimpleVLA 提交的`modified_codes/robotwin2`。
3. 使用皮尔`demo_randomized.yml`，不要改成 `demo_randomized_aloha.yml`保持动作/本体`14D`、片段`25`、已看到指示、`temperature=1.6`。
4. H800 必须重新安装或编译以匹配其 CUDA compute capability 的 CuRobo/Warp 扩展。L40 环境下的 CuRobo 是专为`sm_89` 编译，**无法复制到 H800**；先做一次单 seed 专家`play_once()`，然后进行模型的冒烟测试。
5. 固定作者 Transformer 分支`4.40.1`、`tokenizers==0.19.1`，并设 `PYTHONNOUSERSITE=1`；运行前打印实际导入路径和版本。
6. 用新输出目录。先运行 `val_only` 的 `100100007/100100013/100100017`烟雾；RL参考应能重现正例，SFT可作为负例。正例未通过时禁止提交长训练。
7. 在经过烟雾测试后，才使用作者提供的 8 卡、1000 试验次数、GRPO 二进制奖励参数启动训练；首次保存检查点时也先回到相同的烟雾集。

- 8个GPU，`num_trials_per_task=1000`，`n_samples=8`，`train_batch_size=64`。
- `action_token_len=14`，`action_chunks_len=25`。
- `twin2_task_config=demo_randomized`，`instruction_type=seen`，`use_proprio=True`，`temperature=1.6`。
- 组，二进制验证器，`val_before_train=True`。
- 起点使用公开的SFT检查点；输出目录必须是新目录，不能覆盖公开的SFT/RL参考。

在开始训练之前，先进行一次包含 1 到 3 个已验证的评估种子（evaluation seed）的单一烟雾测试（val-only smoke test）；训练结束后，每次保存检查点时都使用相同的烟雾集进行回归。模型改动也应先经过该烟雾集，然后再扩展到完整的评估种子表。

## 8. 当前限制

- 目前未完成作者规模的 `1000`试验、`8`samples 的正式 RL 训练或完整的 val-only 统计。
- 当前 `3/3`只证明当前L40诊断链路存在可靠正例；不应与论文总成功率并列。
- 当前按用户要求不连接、不查询、不提交、不停止任务。正式训练需用户明确解除限制后再开始。
- `execution.md` 是按时间追加的原始操作账本；本文件是面向复现和排障的教学总结。两者应同时保留。
