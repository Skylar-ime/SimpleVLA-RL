# 简单VLA-RL`RoboTwin2 / lift_pot` 复现报告

本报告记录 `ex/fuxian`在分支上对 SimpleVLA-RL 的`RoboTwin2 / lift_pot`重现过程和当前结果。实验目标是首先使用公开的 SFT checkpoint 进行验证，然后运行作者的 binary-reward RL 流程；实际执行阶段只能使用 L40，未操作用户正在使用的 H800 节点。

## 实验口径

代码固定为 SimpleVLA-RL 版本`462f23d91b0190694dd5ede12860c9c1a4f12ed1`。RoboTwin 基础版本固定为 commit`6be6922d74f9b729837d6bcaa703721ec4553779`并叠加该 SimpleVLA 提交中的`modified_codes/robotwin2`。任务配置采用作者实验口径的 `demo_randomized`，其实现是 Piper 双臂；检查点名称中的`aloha`对应 OpenVLA-OFT 的动作常量，并不意味着这里应该使用 ALOHA-AgileX 物理环境。模型使用单个头相机图像、语言指令和 14 维的 proprioception 输出 25 个动作 chunk，每个动作有 14 维。RoboTwin 执行器将每个 chunk 作为绝对关节位置和归一化夹爪值执行，单回合最多 200 步控制。

环境和模型运行在 L40 专用隔离环境中`robotwin-pristine`中文。该环境使用 Python 3.10 和 PyTorch`2.4.1+cu124`、为了L40`sm_89`编译的 CuRobo，以及作者 Transformers fork`4.40.1` 和 `tokenizers==0.19.1`；运行时设置 `PYTHONNOUSERSITE=1`，防止其他 Transformer 版本覆盖作者依赖。公开的 SFT 检查点位于`/public/home/s2250018/wsy/VLA/checkpoints/lift_pot_sft_eval_l40`包括四个权重 shard 和独立的`proprio_projector--20000_checkpoint.pt`检查点的`dataset_statistics.json` 使用 `lift_pot_1k`动作和本体感受均为14维。

## 复现过程

在开始训练之前，我们已经检查了环境、检查点和动作接口。RoboTwin 可以创建 Piper 环境并返回`240x320x3`头部的RGB值与14维`joint_action.vector`；将初始关节状态重复为 25 步的恒定动作后，前后 qpos 的最大漂移约为`5e-4`，未发现动作维度错误、二次归一化或执行器接口错误。随后在作者评估种子中先运行专家规划器，只把专家`plan_success=True` 且 `check_success=True`种子用于可解释的模型诊断样本。公开的RL参考检查点在`100100007`、`100100013`、`100100017`三个有效的种子均成功，并在第100个控制步骤达到成功判定；结果为`3/3`。公开的SFT检查点在相同条件下会靠近锅、关闭夹爪并短暂抬起，但随后掉落。这个对照证明了当前RoboTwin、SimpleVLA包装器、检查点输入输出、动作执行和二进制奖励判定能够形成闭环；`3/3`只属于小规模正确性检查，不是论文规模成功率。

之后启动了从公开 SFT checkpoint 开始的二进制奖励强化学习。作者设置了使用`n_samples=8`，并保留 `filter_accuracy=True`、奖励区间过滤和 GRPO 训练；当 L40 显存不足时，将全局`actor.ppo_micro_batch_size` 调整为 `4`，使四个卡 FSDP 每卡实际处理一个样本，并启用梯度检查点。这样保持了作者的采样组大小、奖励逻辑、学习率、动作块、温度和环境配置，只对与硬件相关的显存参数进行适配。训练输出目录为`/public/home/s2250018/wsy/VLA/outputs/lift_pot_l40_4gpu/twin2_lift_pot_l40_4gpu_sft_start_binary_n8_gc_v2`，主要日志为 `/public/home/s2250018/wsy/VLA/outputs/lift_pot_l40_4gpu/logs/train_n8_gc_v2.log`。

训练前验证在 256 个episode 上完成。初始SFT策略的IID、OOD和总体成功率均为`0.046875`，即在256个验证episode中有12个成功。之前使用每组4个样本时，第一批rollout可能全部得到零奖励，过滤器会丢弃整组，无法形成actor更新；改为作者的每组8个样本后，出现了`0.12` 和 `0.38`有效的组已成功更新为 FSDP actor。第一次actor更新耗时约为`389.4 s`，说明 RL 训练链路已经实际执行，而不是只完成模型加载或环境启动。

## 最新 checkpoint 与结果

最终在L40上推进到训练更新`step=42`，验证推进到 `step=43`然后因为Ray工作器连接失败以及Python工作器的`_PyGILState_NoteThreadState: Couldn't create autoTSSkey mapping`错误退出。退出前磁盘上最新 checkpoint 是：

```text
/public/home/s2250018/wsy/VLA/outputs/lift_pot_l40_4gpu/twin2_lift_pot_l40_4gpu_sft_start_binary_n8_gc_v2/actor/global_step_39
```

这个检查点是当前最新的可恢复模型；运行在`step=42/43`更新和验证没有在崩溃前形成更新的 checkpoint。`global_step_39` 对应日志中的聚合验证结果为：

| 指标 | 数值 | 含义 |
|---|---:|---|
|训练IID| `0.375` |训练分布中的验证成功率，在128个IIDepisode中，有48个成功。|
|评估 OOD| `0.422` |在评测分布中，验证成功率约为 54/128。|
|所有| `0.398` |IID 与 OOD 合并后的总体成功率，在 256 个episode 中有 102 个成功和 154 个失败。|

日志中的 `step=39` 验证视频文件名使用 `step=40`，原因是验证调用接收的是 `global_steps + 1`；不能据此认为保存了 `global_step_40`远程发布目录中的这些视频共有256个，成功和失败的计数为`102/154`，与 `all=0.398`一致。视频只保存了 rollout 画面；模型动作、关节状态、奖励和 seed 仍以日志及运行时记录为准，未随 MP4 一起写入完整轨迹文件。

训练更新日志中的 `train_verify_score/all`是当前采样训练批次的二进制验证器平均值，而不是完整验证集的成功率；例如`step=42` 的值为 `0.281`，表示该训练批次的成功率约为28.1%。`reward`这是经过当前 GRPO/binary-reward 处理后的批量奖励均值。`step=42` 为 `2.500`不能直接作为episode的成功率，因为同一episode的多次采样、过滤和优势计算都会影响它。`PPO KL` 用来监控新旧策略分布的偏移，`step=42` 为 `0.002`，表示这次更新相对保守；`grad norm` 是梯度范数，`2.667`表示本次反向传播的梯度规模。它们用于判断训练稳定性，不替代IID、OOD和总体成功率。

## 结果边界

当前结果可以确认三件事：环境和 wrapper 能运行；binary reward 能在已知正例上触发；从公开 SFT checkpoint 开始的 RL 已完成多次真实 actor update，并产生了`global_step_39`目前无法声称已经完成了论文规模的正式复现，因为训练在后续工人错误中止，未完成作者计划的完整训练步骤，也没有获得最终训练检查点或完整的独立评估统计。`global_step_39` 的 `0.398`这是本次L40运行的最新验证结果，不能与论文报告的最终任务成功率直接比较。

后续若继续实验，应从 `global_step_39`恢复时，首先在同一个有效的评估种子上重新测试，并保留IID/OOD分开的统计；确认恢复和重新测试正常后，再决定是否继续训练。环境、检查点、种子、解码器和奖励判定均应保持不变，以避免将新配置下的分数与本报告结果混淆。
