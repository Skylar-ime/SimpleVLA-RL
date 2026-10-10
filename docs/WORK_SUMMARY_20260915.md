# 抬起_复现工作总结

## 目标

- 分支：`ex/fuxian`
- 任务：RoboTwin2.0`lift_pot`
- 机器：单节点 8×H800，服务器工作根`/public/home/s2250018/wsy/VLA`
- 顺序：SFT`val_only`，以及原始二进制奖励基线

## 已做

1. 校验 SSH：本地`zhongxin` -> 登录节点 -> `h800-1`；8 张 H800、CUDA/PyTorch/conda 可用。
2. 迁移并修复服务器目录：
   - 仓库：`VLA/SimpleVLA-RL`
   - RoboTwin：`VLA/RoboTwin`
   - Curobo：`VLA/curobo-src`
   - 检查点/输出：`VLA/checkpoints`、`VLA/outputs`
3. 创建独立的 SFT 工作副本；保留原始文件。下载并验证公开的 RL 参考；7 个模型片段的 SHA-256 通过。
4. 增加诊断开关：验证 trials、显式 seed、RoboTwin 路径、`predict_action`、验证采样；默认行为未改。
5. 写入并同步 `AGENTS.md`、`PLANS.md`、`EXECUTION_LOG.md`。

## 失败/结果

| 实验 | 结果 | 备注 |
|---|---:|---|
|当前 RoboTwin + Piper，SFT`val_only` | `0/256` |链路正常，MP4 已保存|
|SFT + 普尔，温度`1.6`，固定 16 条 | `0/16` | 原脚本采样口径 |
|原始二进制基准首批训练滚动| `512/512` 失败 | `Accuracy 0.00:64`，过滤后无更新|
|RoboTwin`6be6922`+ 萍儿| `0/8` | 仅换源码无效 |
|RoboTwin`6be6922`+ 拉哈| `0/2` | 可运行但失败 |
|历史 ALOHA + 官方`predict_action` | `0/16` |解码器A/B无改进|
|历史 ALOHA + 采样解码器| `0/16` |温度A/B无改善|
|公开 RL 参考 + 历史 ALOHA| `0/16` | 模型能加载、动作非零 |
|公开 RL 参考 + 当前 Piper| `0/2` | 无成功 |
| `demo_clean`你好| `0/2` | 无成功 |
|历史 ALOHA 8 条种子扫描| 中止 | 物体稳定性检查失败，非模型分数 |
|官方 RoboTwin OpenVLA-OFT 评估器| 未完成 | 历史包缺 `prismatic.models`；完整包缺 `dlimp` |

旧版 RoboTwin 资产（2025-08-11）已迁移至服务器；`embodiments.zip`、`objects.zip`SHA-256 与本地快照一致。资产版本不是已证实的主因。

## 已确认

- 检查点包含4个SFT模型片段，`proprio_projector--20000_checkpoint.pt`，没有连续`action_head`应该远离离散动作令牌。
- `dataset_statistics.json`的 14-D 绝对关节范围匹配 Piper 双臂；模型名中的`aloha`不能直接据此选择 ALOHA 环境。
- 动作生成不全是零：令牌、解码动作、动作范围都正常；问题不是显卡、Ray、EGL 或基本环境启动。
- LORA 目录已存在；`lm_head`可以证明已经合并，完整的LoRA合并尚未完全审计。

## 当前问题

公开检查点在当前分支、历史RoboTwin、Piper/ALOHA、贪婪/采样解码器下均未重现论文中的非零SFT。原因尚未证实，最可能是作者在训练时的模型代码、处理器、数据动作契约、环境提交/资产组合与公开检查点不完全一致。不能把当前`0%`当论文复现时，也不能启动100个epoch的基线。

## 后续顺序

1. 获取作者在训练时的精确提交记录、数据包、处理器/模型文件以及评估脚本；优先恢复官方评估器的完整依赖（独立环境安装）`dlimp` 等）。
2. 进行离线单步重播：使用训练轨迹图像 + qpos + 指令，逐项检查相机顺序。`num_images_in_input`、提示`Out:`、 proprio BOUNDS 归一化、14-D 动作顺序和反归一化。
3. 使用相同的 replay 比较 checkpoint 标签令牌与预测令牌；确认后再跑 16 条，再跑完整`val_only`。
4. 只有 `val_only`只有出现可解释的非零结果时，才启动原始二进制基线；保持奖励为0/1，不添加丰富反馈。
5. 如果无法获得精确匹配，则改为“服务器重新执行少量提升”_冻结该模型 -> RL"路线，并单独标记为新基线，不冒充论文复现。
