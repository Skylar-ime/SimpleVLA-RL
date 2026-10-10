# RoboTwin 2.0`lift_pot` 复现计划

## 目标和验收

在 `ex/fuxian`当前提交中，使用单节点 8 张 H800 和公开 SFT checkpoint，完成了两个连续阶段：

1. `val_only`：验证环境、检查点和推理链路，获得冻结口径的SFT成功率。
2. 二进制奖励基准：使用仓库原始 GRPO 与 0/1 成功奖励进行训练，获得可供后续小幅模型改动比较的短期任务基准。

验收结果包括完整的配置、日志、检查点、每个episode的结果、汇总的成功率以及rollout视频。只得到一个最终的成功率，不算完成重现。

## 已确认的任务配置

| 项目 | 值 |
| --- | --- |
|任务套件| `robotwin2_lift_pot` |
|检查点| `Haozhan72/twin2_lift_pot_seed1k_sft_aloha_25chunks_20k`（名称包含 aloha；统计/动作范围显示实际为 Piper 双臂）|
|GPU|8 x H800，单节点|
|动作令牌长度| `14` |
|动作片段| `25` |
|最大回合步数| `200` |
|指令类型| `seen` |
|观察|RGB + 身体位置/本体感觉|
|未规范化键| `robotwin2_lift_pot_1k` |
|基础算法|分组|
|基础奖励|成功=`1`失败=`0` |
|KL系数| `0.0` |
|学习率| `5e-6` |

仓库当前 `lift_pot` 种子文件包含 1000 个训练种子和 200 个评估种子。正式评估前必须确认脚本实际使用的种子集合，避免训练和评估口径混淆。

## 当前执行记录（2026-09-14）

- H800 工人：`h800-1`8 张 H800；PyTorch`2.4.1+cu124`CUDA工具包`12.4`。
- 服务器工作根已迁移到 `/public/home/s2250018/wsy/VLA`：仓库 `/public/home/s2250018/wsy/VLA/SimpleVLA-RL`，RoboTwin`/public/home/s2250018/wsy/VLA/RoboTwin`，酷跑`/public/home/s2250018/wsy/VLA/curobo-src`，开放虚拟实验室-光学场测试`/public/home/s2250018/wsy/VLA/openvla-oft`；后续命令统一使用这些路径。
- RoboTwin 2.0 已复制到服务器 SimpleVLA-RL 工作区，资产与 Curobo 均可加载；仓库根`assets` 为指向嵌入资产目录的软链接。
- 规范 SFT 工作副本：`/public/home/s2250018/wsy/VLA/checkpoints/lift_pot_sft_merged`。它与旧 `/wsy/checkpoints/lift_pot_sft_work`有 72 个有效的文件与 SHA-256 相同；旧的只读归档不会被删除，以后也不会再引用。
- 已完成 256 条验证条目（代码实际为 IID 128 + OOD 128；不是计划文字中的 200），保存 rollout MP4。
- 当前结果：`robotwin2_lift_pot_train_iid=0.000`，`robotwin2_lift_pot_eval_ood=0.000`，`all=0.000`。
- 这个结果证明了端到端链路可以运行，但不满足论文SFT参考值。`10.1%`。
- 旧基准进程（PID）`3054129`）只完成了 val-before-train；不能据此断定过滤器删除了所有训练样本。
- 当前按照原始参数重启的8个GPU基线（PID`3678888`已完成首个训练滚动批次：`512/512` 失败，`Accuracy distribution: 0.00:64`，过滤后为 0；无 actor 更新/检查点，作业已停止。
- 结论：严格的二进制基线被当前的SFT检查点/环境组合阻止；先做版本/实现/检查点对齐，不修改基线过滤参数以冒充重现。
- 兼容性 A/B：RoboTwin`6be6922`（2025-08-10）+ 派珀、同源 + ALOHA、离散/采样解码器均为0；仅切换版本/实现/解码器未恢复成功率。
- 当前优先级：审计 checkpoint 原始`modeling_prismatic.py`处理器、本体/动作统计及覆盖差异；在此之前不投入正式100轮。
- 论文对照：报告中的 RoboTwin2.0`lift_pot`OpenVLA-OFT SFT 为 10.1%，RL 为 64.1%；当前 0% 明确表示复现口径仍未对齐。
- 当前 SFT 统计为 Piper 14-D 关节范围；名称中`aloha`不能用于选择 ALOHA 环境。采样温度`1.6`Piper 的 16 条验证仍然有效。`0/16`。
- 旧 RoboTwin 资产快照已迁入`VLA/checkpoints`而且，SHA-256 是正确的；资产版本并不是已证实的主要原因。
- 判断实验：下载公开RL`lift_pot`检查点，短测在同一环境中。如果 RL 不为零，问题在 SFT 检查点；如果 RL 仍然为零，问题在环境/动作契约中。RL 完整的检查点包含 proprio projector，在加载时使用`actor_rollout_ref.model.resume=True`。

## 阶段 0：冻结复现起点

- 记录 `git rev-parse HEAD`、`git status --short` 和当前分支。
- 为本次复现建立独立配置、日志和输出目录。
- 不清理现有的脏工作树；只提交本次新增或明确修改的文件。
- 固定检查点提交、RoboTwin 提交、Python 和包版本。

## 阶段 1：连接并检查 H800 节点

本机已配置 SSH 别名`zhongxin`。建议路径：

```bash
ssh zhongxin
squeue -u "$USER"
```

如果没有活动的 8 卡作业，按照集群现有的规则申请单节点 8 张 H800。获得 worker 后进入该节点，然后再检查：

```bash
hostname
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count()); print(torch.cuda.get_device_name(0))'
python -c 'import flash_attn, verl; print("flash_attn OK"); print("verl OK")'
```

通过条件：8 张 GPU 全部可见；型号、显存、驱动正常；PyTorch CUDA 可用；关键包可导入。历史记录显示该集群 worker 曾提供 8 张约 80 GB 的 H800，但每次作业仍需重新验证。

## 阶段 2：准备代码和 RoboTwin

在服务器项目盘创建独立工作区，检出与本地相同的提交。RoboTwin 使用独立克隆，并记录提交。

应用仓库适配脚本前先检查脚本内容。其参数顺序为：

```bash
bash copy_overwrite_robotwin2.sh /absolute/path/to/RoboTwin /absolute/path/to/SimpleVLA-RL
```

这个脚本会覆盖RoboTwin文件并删除资源文件，因此请勿直接在唯一的RoboTwin环境中执行。完成后运行最小的import检查，并确认MuJoCo/EGL在无显示器的worker上可以渲染。

## 阶段 3：准备公开 SFT checkpoint

下载并固定修订版：

```bash
huggingface-cli download \
  Haozhan72/twin2_lift_pot_seed1k_sft_aloha_25chunks_20k \
  --local-dir /path/to/checkpoints/lift_pot_sft_original
cp -a /path/to/checkpoints/lift_pot_sft_original /path/to/checkpoints/lift_pot_sft_work
```

只对 `lift_pot_sft_work`执行仓库所需的checkpoint overwriting/兼容性脚本。然后记录目录大小、文件列表和修订版本。

## 阶段 4：单次episode烟雾测试

先用单进程、单 GPU、固定评估种子运行一个episode。检查：

- 环境可以重置，RGB 观测和 qpos 观测都存在，并且形状正确。
- 检查点可以加载，并且动作输出为预期的14维、25个片段。
- 环境最多运行 200 步并正常结束。
- 成功判定能返回布尔值，视频能保存。
- 无 NaN、显存溢出、EGL 或相机错误。

该阶段失败时只修复运行链路，不改变任务、奖励或模型超参数。

## 阶段 5：`val_only`

从 `examples/run_openvla_oft_rl_twin2.sh` 派生专用 `lift_pot` 配置。至少固定：

```text
data.task_suite_name=robotwin2_lift_pot
trainer.val_only=True
trainer.val_before_train=True
model.action_token_len=14
model.num_action_chunks=25
rollout.instruction_type=seen
rollout.use_proprio=True
rollout.unnorm_key=robotwin2_lift_pot_1k
```

先跑少量固定种子来确认多 GPU rollout 的汇总是否正确，然后再跑完整的 200 个评估种子。如果框架的验证参数会重复采样或默认使用 1000 次试验，应该显式地将评估集合冻结，而不是将“运行次数”误认为是“独立种子数”。

输出：

- SFT 总成功率与成功数/总数
- 每个种子的成功标志、步数和终止原因
- 固定数量的成功/失败视频
- 吞吐、峰值显存和总耗时

## 阶段 6：原始二进制奖励基线

只有 `val_only` 通过后启动训练。以仓库脚本为基准，保持原始关键值：

```text
algorithm.adv_estimator=grpo
algorithm.kl_ctrl.kl_coef=0.0
trainer.total_epochs=100
trainer.val_before_train=True
trainer.val_only=False
data.train_batch_size=64
data.val_batch_size=256
data.n_samples=8
data.traj_mini_batch_size=8
actor_rollout_ref.actor.optim.lr=5e-6
actor_rollout_ref.actor.ppo_mini_batch_size=128
```

奖励实现使用 `complete`字段：成功episode奖励为`1.0`，否则为 `0.0`。不要在这个阶段加入距离、阶段进度和接触状态等塑造信号。

首先进行短跑预检，例如一次训练更新加上一次验证；检查8卡利用率、rollout汇总、参数更新、checkpoint保存和恢复。通过后再跑正式的100个epoch。如果官方checkpoint或论文使用了不同的隐藏参数，在结果中应明确标注差异。

## 阶段 7：冻结可比较基线

训练结束后至少比较：

- 训练前 SFT`val_only` 成功率
- 最佳验证检查点成功率
- 最后检查点成功率
- 三个或更多的固定随机种子的平均值和方差；在资源不足时优先完成一个种子，并明确标记为预实验

将可复现的命令、配置和结果写入单独的实验记录。后续模型的小改动必须使用相同的评估种子、episode 数、成功判定和视频采样规则。

## 后续的 rich-feedback 接口

`lift_pot`适合用作第一个富有反馈的短期任务，因为它已经具备了仿真环境中的物体位置、双臂末端距离、接触/抓取条件以及物体方向等信息，可以组合成分阶段的反馈。这项工作是在冻结二元基准之后进行的：保留原始的成功率作为最终指标，同时增加诊断字段或形状奖励，以避免改变基准定义。

## 停止条件

出现以下情况时停止长跑并保留日志：

- 评估种子集不明确或与训练集重叠。
- 多卡结果与单卡 smoke test 的动作/成功判定不一致。
- 检查点被原地修改且无法追溯。
- 在训练过程中遇到NaN值、持续的OOM（内存溢出）、环境进程泄露或长时间闲置的GPU。
- 恢复训练无法获取连续的 step、优化器和调度器状态。
