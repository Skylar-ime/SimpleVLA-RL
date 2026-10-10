# 作者环境追踪审计（2026-09-15）

## 结论

公开资料无法给出作者训练/评测环境的逐项精确清单。最强可复现候选是：RoboTwin 上游 2025-08-10 左右代码线（本地历史副本 commit`6be6922d74f...`，对应上游合并`4be3e685858d...`），SimpleVLA 的私有预发布代码，OpenVLA-OFT 支持到 2025-08-03；但这个配对没有被作者公开声明，置信度为“候选”，而不是事实。

## 已证实

- 论文：RoboTwin2.0 使用 AgileX Piper；每个任务 100 个 hold-out 场景；训练使用 1000 演示/任务和 1000 RL 场景/任务。RoboTwin 配置为 25 个动作片段、256 个动作令牌；rollout 温度 1.6；训练硬件为 8×A800 80GB。`lift_pot`表格中的 SFT 10.1 和 RL 64.1。
- 论文模型的输入包括单视角图像、语言指令和本体感知；官方的 OpenVLA-OFT 还使用手腕图像。公开检查点名称中的`aloha`无法单独证明应该跑 ALOHA。
- SFT 检查点 创建于 2025-08-12：`Haozhan72/twin2_lift_pot_seed1k_sft_aloha_25chunks_20k`。公开RL检查点同日发布。
- SimpleVLA 公共 RoboTwin 支持在仓库历史中到 2025-09-30 的 PR 合并`2d0fb0d...`才出现，晚于 checkpoint 约 7 周；因此 checkpoint 的原始训练代码很可能是未公开的私有代码。
- 官方SETUP只提供范围性依赖：Python 3.10、PyTorch 2.4.0/cu124、veRL v0.2/v0.3；没有锁定作者的完整lockfile、镜像、CUDA minor、Ray、RoboTwin commit或资产SHA。

## 未证实 / 不能冒充精确答案

- 未找到作者 Dockerfile、conda lock、requirements lock、训练日志中的 git SHA、RoboTwin 资产清单、processor/model 版本绑定。
- 在 GitHub 问题中提到的 `torch/CUDA/Ray/8×H100` 是其他用户的环境，而不是作者的环境。
- 当前公开分支、历史 RoboTwin、Piper/ALOHA、贪心/采样 decoder 的提升_验证结果均为0；这意味着公开配对尚未重现，但不能据此推断论文结果是错误的。

## 推荐冻结的公开近似栈

```text
Python 3.10
PyTorch 2.4.0 + cu124
veRL 0.2.x（先按官方脚本依赖安装）
SimpleVLA-RL: ex/fuxian（当前实验补丁）
RoboTwin: 6be6922d74f...（2025-08-10 历史副本）
OpenVLA-OFT: 2025-08-03 左右支持代码
robot: Piper；task: lift_pot；single-view + proprio；25 chunks；256 tokens
```

这个栈仅用于A/B和离线重放，不标记为“作者精确环境”。要达到严格复现，必须向作者索取commit/资产/处理器manifest，或者拿到作者运行日志中能证明上述版本的SHA。

## 下一步

1. 在服务器上创建隔离环境，补充官方评估员缺失的`dlimp` 等依赖；不改正在使用的 `simplevla` 环境。
2. 先进行离线单步回放：训练图像、相机顺序、提示`Out:`、 proprio BOUNDS、14-D 动作顺序/反归一化、标签令牌与预测令牌逐一核对。
3. 只有当 replay 合同成立时，`val_only`非零，才跑 16 条 smoke test，再跑原始 binary-reward baseline。
4. 请同步向作者请求精确的环境；论文作者联系方式：李浩赞（`zhan72426@gmail.com`）、宁定（`dingning@mail.tsinghua.edu.cn`）。

## 来源

- 论文与实验配置：https://arxiv.org/abs/2509.09674
- 官方安装说明：https://github.com/PRIME-RL/SimpleVLA-RL/blob/main/SETUP.md
- 官方 RoboTwin 启动脚本：https://github.com/PRIME-RL/SimpleVLA-RL/blob/main/examples/run_openvla_oft_rl_twin2.sh
- SFT检查点：https://huggingface.co/Haozhan72/twin2_lift_pot_seed1k_sft_aloha_25chunks_20k
- RL检查点：https://huggingface.co/Haozhan72/twin2_lift_pot_seed1k_sft_20k_batch64_rl_seed1000_max200_step40
- 作者历史：https://github.com/zhan72/SimpleVLA-RL-dev2.0/tree/dev-oft-robotwin
- RoboTwin 上游历史：https://github.com/RoboTwin-Platform/RoboTwin/commits/main/
