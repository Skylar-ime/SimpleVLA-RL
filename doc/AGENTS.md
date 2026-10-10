# SimplerVLA-RL 重现约束

本文件约束 `doc/` 下的复现记录。仓库级代码改动仍应遵循仓库根目录规则。

## 当前目标

- 分支：`ex/fuxian`
- 环境：RoboTwin 2.0
- 任务：`lift_pot`
- 计算资源：单节点 8 张 NVIDIA H800
- 初始模型：公开 SFT 检查点`Haozhan72/twin2_lift_pot_seed1k_sft_aloha_25chunks_20k`
- 执行顺序：先 `val_only`，通过后再跑仓库原始二进制奖励基线
- 当前阶段不修改奖励定义，不引入丰富反馈，不追求复现全部任务

## 分支和改动边界

1. 只在 `ex/fuxian` 分支开展原文复现。
2. 不修改 `main`，不覆盖用户已有改动。
3. 重现基准与后续的 rich-feedback 开发分开提交、分开配置、分开输出目录。
4. 任何与论文或仓库默认设置不同的参数，必须记录参数名、旧值、新值和原因。
5. 不要将机器地址、私钥、访问令牌或账号密码提交到 Git。

## 远程执行边界

- 这台机器用于编辑代码、查看日志和发起 SSH；RoboTwin、模型、训练环境与数据放在 H800 服务器。
- 已知入口是本地 SSH 映射`zhongxin`它连接登录节点，不代表当前已有 GPU worker。
- 在登录节点后，使用调度器申请或确认 8 卡 H800 作业，然后进入分配到的 worker 执行。
- 在未确认 worker、CUDA、驱动、磁盘和网络状态之前，不要开始下载或训练。
- 大文件放服务器公共存储或项目数据盘，不同步回本机仓库。

## 检查点规则

1. 下载公开的SFT检查点后保留一份只读副本。
2. `overwrite_vla_ckpt_utils.sh`它将修改检查点目录。只能对工作副本进行操作。
3. 每次实验记录 checkpoint 仓库名、revision/commit、下载路径和工作副本路径。
4. `unnorm_key` 固定为 `robotwin2_lift_pot_1k`，除非数据统计明确要求改变。

## 环境和奖励规则

- RoboTwin 版本与 SimpleVLA-RL 当前提交必须固定并记录。
- `copy_overwrite_robotwin2.sh` 的实际参数顺序是：`<robotwin_path> <simplevlarl_path>`。
- 这个脚本会覆盖RoboTwin文件并删除资源；在运行前先做一个可恢复的备份，或者使用独立克隆。
- 基准奖励保持仓库原始定义：episode 成功为`1`，失败为 `0`。
- `lift_pot` 的成功判定、最大步数 `200`、相机和 proprio 配置在 baseline 阶段保持不变。
- 后续的 rich-feedback 奖励必须作为独立实验实现，不能污染原始的 binary baseline。

## 验证门槛

按以下顺序执行；上一阶段失败时停止，不直接启动长训练：

1. 远程 8 卡、驱动、CUDA、PyTorch、FlashAttention、vLLM/verl 检查。
2. RoboTwin 单次环境启动、观察和动作形状检查。
3. 加载检查点 SFT。
4. 小样本评估 烟雾测试。
5. 完整 `val_only`，保存每个episode的成功结果和汇总成功率。
6. 原始二进制奖励基线，保留训练、验证、检查点和环境日志。

## 实验记录

- 主要决策、环境变更、执行结果和失败原因统一追加到 `doc/execution.md`不要为单次任务新建 Markdown 日志。

每个运行目录至少保存：

- Git 分支、提交、脏差异摘要
- 服务器主机名、GPU型号和数量
- Conda 环境与关键包版本
- 完整启动命令和展开后的配置
- 检查点与数据路径
- 随机种子、episode 数、训练步数
- 日志文件、TensorBoard 或 W&B 日志
- 成功率、失败原因分类和代表性滚动视频

只有当配置、代码、检查点、种子和评估标准都能追溯时，才会将结果标记为“可复现”。
