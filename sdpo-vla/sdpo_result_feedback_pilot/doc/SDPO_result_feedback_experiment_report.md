# SDPO 机器人结果反馈实验汇报

日期：2026-09-28  
覆盖目录：`sdpo_result_feedback_pilot`、`sdpo_result_feedback_pilot_v5`、`sdpo_result_feedback_pilot_v6`、`sdpo_result_feedback_pilot_v7`

## 一 结论先行

这组实验没有建立“结果反馈条件化自蒸馏能够稳定提升无反馈部署控制能力”的证据。更准确的结论是：

1. 反馈文本确实能改变教师分布。V8 中 `causal` 的平均教师方向准确率约 37.6%，V9 中目标方向反馈约 53.45%，均高于原始结果反馈约 25.20%；V10 中目标方向反馈的平均教师方向准确率约 63.7%，高于原始结果反馈约 31.3%。
2. 教师分布改变没有转化为部署成功率提升。V8 的 `result`、`binary`、`shuffled`、`neutral` 均为 13/64；V9 为 11/64、13/64、1/64、7/64；V10 为 11/64、13/64、0/64、6/64。原始结果反馈始终没有超过二元反馈。
3. 主要瓶颈逐渐从“没有可观测反馈”转向“视觉地基、接触几何、标签定义和长时域执行”。P0/P1 诊断显示，给模型特权目标方向后 Qwen 3B 成功率可达 42/64；使用干净的 `direction_action` 监督后升至 57/64（89.1%），但 RGB-only 仍只有 8/64。Qwen 7B 在同一干净监督下 RGB 闭环为 25/64，说明容量有帮助但不能替代可靠地基和闭环执行。


核心判断：当前工作最有价值的产出不是正向效果，而是把失败原因从“反馈机制是否存在”逐步拆解为“**任务信号、教师语义、视觉方向识别、动作标签和长时域控制**”几个可验证问题。

## 二 统一实验协议与证据边界

### 2.1 任务和策略

- 模型：Qwen2.5-VL-3B-Instruct；语言解码器注意力投影 LoRA，约 3.69M 可训练参数；视觉编码器、视觉投影器和基础权重冻结。
- 环境：基于 panda-gym/PyBullet 的自定义 PandaPush 变体，绿色方块推向蓝色目标；结果不能当作原始 `PandaPush-v3` 成绩。
- 观测：448×448 RGB，任务指令，固定动作定义，可选末端 xyz；actor不接收方块/目标真值坐标、结果反馈或教师动作标签。
- 动作：A/B/C/D/E 五个单 token，分别表示世界坐标 +x/−x/+y/−y/保持；只读取回复起始位置的五个 logits。
- 单步：一次 `set_action` 后推进 0.16 s；目标动作增量为 2 cm；成功条件为方块到目标的三维距离严格小于 2 cm；部署最多 12 或 32 个决策，依场景协议而定。
- 蒸馏：教师在学生更新前计算并缓存五动作分布；V1 至 V6 主要用反向 KL，V7 后切换到 `alpha=0.5` 的 JSD；没有 GRPO、行为克隆、专家动作损失或额外奖励损失。
- 反馈：`contact`、`object_motion`、`goal_progress`、`success`；`directional` 反馈额外包含由目标向量导出的轴向信息，信息量高于干净的仅结果反馈，不能与纯结果反馈等同。

### 2.2 统计口径

- “教师方向准确率”是与模拟器主导目标轴的诊断匹配，不是任务成功率，也不检查夹爪是否处于可推动位置。
- 单种子 64 场景只能给出探索性效应；不能把 64 个 episode 当作 64 个独立训练重复。

## 三 版本演进总览

| 阶段 | 要解决的主要问题 | 关键改动 | 主要结果 | 当前结论 |
|---|---|---|---|---|
| base Phase A / B v1 | 接触和结果变化太少 | 初始协议、结果/二元/打乱对照 | 0/64 成功；结果与对照几乎相同 | 不能进入大规模训练 |
| B v2 | v1 近乎无接触 | 末端缩至 4.4–5.2 cm，加入 neutral | 54 个有效 reset，仍 0 成功；结果仅比冻结多 0.122 mm | 物理信号增加但仍不足 |
| B v3 | 需要能完成任务的局部方向场景 | directional 场景 | 冻结 11/64，结果/二元/打乱/neutral 均 13/64 | 提升来自场景，不是反馈 |
| B v4 | 需对齐 SDPO 原始训练流程 | EMA、反向 KL、mask、长方向场景 | 结果 11/64，二元 13/64，解释反馈 3/64 | 数值结果难转成动作修订 |
| V5 | v4 的几何和教师解释混淆 | 轴向接触校准，causal 与 causal-shuffled | 结果 8/64，二元 9/64，causal 6/64 | 教师改变未带来部署收益 |
| V6 | V5 没有同状态自生成解决方案 | 八候选同状态分组，自解决方案条件 | solution-only 6/64，solution+binary 8/64；causal 仅跑 2 更新 | 机制更像 SDPO，但实验未完成且 mask 定义仍有问题 |
| V7 / Stage D | 修复 V5/V6 对 SDPO 的偏离 | 真正任务成功兄弟才算 solution，JSD | grouped snapshot 状态一致性错误，中途失败 | 暴露工程一致性问题 |
| V8 / Stage E | 隔离“仅环境反馈” | `directional_long`，无一步成功 solution | result/binary/shuffled/neutral 都 13/64；causal 0/64 | 教师差异不等于部署差异 |
| V9 / Stage F | 评估反馈是否包含目标方向泄漏 | 增加 directional 对照 | binary 13/64，result 11/64，directional 7/64，causal 1/64 | 方向信息能改教师，反而可能损害部署 |
| V10 / Stage I | 增加错误方向控制并重复验证 | `wrong_directional`，12 更新 | binary 13/64，result 11/64，wrong 9/64，directional 6/64，causal 0/64 | 结果反馈仍未超过二元反馈 |
| P0/P1 | 区分反馈瓶颈与视觉/标签瓶颈 | 冻结方向、干净方向标签、3B/7B | 3B 方向监督 57/64；RGB-only 8/64；7B RGB 25/64 | 视觉地基和标签质量是主要瓶颈 |

## 四 base 文件夹：协议、校准和 Phase A




发现：如果不先校准末端高度和恢复状态，后续“反馈无效”可能只是物理协议错误。这个结论直接推动 v2、v3、V5 的场景重构。

### 4.2 Phase A v1：原始随机推物

目的：在训练前检查反馈是否有信息，以及五动作教师是否会因反馈改变分布。

内容：128 个开发场景；每场景 1 次尝试、1 次重放、5 个反事实动作，共 896 次交互；不训练、不提供动作监督。

结果：仅 5/128 尝试发生接触，反事实接触 20/640，仅 8/128 场景有信息性进展。结果反馈教师相对学生平均 TV 为 0.119，期望进展差为 −0.0185 mm；binary、shuffled、neutral 分别为 −0.0193、−0.0255、−0.0145 mm，差异均在噪声和稀疏接触范围内。

问题：模型几乎没有机会观察动作后果；教师分布有变化不能证明它知道哪个动作更好。待验证：需要提高接触覆盖后再判断反馈语义，且要做多 seed 和不依赖单步进展的长时域评估。

### 4.3 Phase A mechanism v1：近接触但重置不稳定

目的：在保持目标方向随机的前提下提高接触密度。

结果：128 个请求场景中 110 个有效、18 个因机器人与方块重叠被拒绝；尝试接触 23/110，反事实接触 120/550，信息性进展 72/110。结果反馈期望进展 −0.118 mm，shuffled −0.139 mm，binary −0.0196 mm，neutral +0.0434 mm。

问题：无效 reset 改变了有效样本组成，也说明“缩短距离”不能直接等价于“增加可用监督”。待验证：统一 reset 拒绝规则、保留完整拒绝率，再跑固定同一状态的教师配对。

### 4.4 Phase A directional v2：物理信号通过，但机制门仍未通过

目的：把场景改成目标相对方块的四个轴向，末端位于背向目标的 4.5–5.2 cm 近接触位置，在不提供目标方向标签的前提下产生可测量局部进展。

结果：128/128 reset 有效；尝试接触 15/128，反事实接触 85/640，信息性进展 71/128。教师加权反事实进展差：result +0.275 mm，shuffled +0.151 mm，binary −0.426 mm，neutral −0.681 mm；平均 TV 分别为 0.116、0.116、0.0996、0.0886。

判断：这是“任务终于有局部信号”的工程成功，不是 SDPO 成功。没有训练，也没有部署收益证据。它为 Stage B v3/v4 提供场景基础。

## 五 base 文件夹：Stage B v1 到 v4

### 5.1 Stage B v1：结果反馈对照的第一次训练

目的：比较 `frozen`、`result`、`binary`、`shuffled`；每训练组 10 次更新、每次 256 个真实交互，共 2,560 个训练交互；固定 64 个开发场景。

结果：四组成功率均为 0/64。平均进展约 −23.005 至 −23.006 mm；结果组训练接触率 3.516%、运动率 3.125%，shuffled 实际改变仅 8.047%，未匹配 donor 0.117%。结果相对冻结进展差 −0.00110 mm，相对 binary +0.000055 mm，相对 shuffled +0.000001 mm。动作几乎坍缩为 A。

发现问题：2306/2560 条结果反馈为同一个 `(no contact, no motion, unchanged, failure)` 元组；名义上“丰富反馈”实际没有可变信用信息。此时继续跑三 seed 没有意义。

待验证：必须增加有效接触、保证 shuffled 改变率可解释，并在能产生任务成功的方向场景中重新跑。

### 5.2 Stage B v2 near-contact：提高接触覆盖

目的：针对 v1 的低接触率，使用近接触场景并增加 neutral 控制，区分提示自复制效应。

结果：每组请求 64 个场景，仅 54 个有效，10 个因重叠被拒绝；五组均 0/54 成功。结果组训练接触/运动率均 8.125%；结果相对冻结进展差 +0.122 mm，但相对 binary、shuffled、neutral 的差异约为 0。所有部署动作仍高度集中在 A。

发现问题：接触覆盖虽翻倍，reset 质量变差；结果信息仍不能改变闭环动作。这个结果不能解释为 SDPO 无效，因为任务仍缺少足够长时域成功样本。

待验证：修复重叠 reset，继续使用 directional 场景；同时记录有效/无效交互成本，不能用补采掩盖组间差异。

### 5.3 Stage B v3 directional：场景可完成，但反馈特异性仍不见

目的：验证方向化场景是否能让局部推力形成任务成功，并观察 result/binary/shuffled/neutral 是否分离。

结果：冻结 11/64，结果 13/64，binary 13/64，shuffled 13/64，neutral 13/64。平均进展分别为 11.267、12.330、12.330、12.330、12.330 mm；结果相对冻结 +3.125 个百分点、+1.063 mm，但相对 binary/shuffled/neutral 为 0 个百分点。训练接触率约 7.1%–7.9%，动作仍主要为 A。

判断：场景改造提高了所有方法的上限，但没有证明结果反馈带来额外收益。待验证：需要更严格的 SDPO 对齐、同状态自解决方案和更长方向场景。

### 5.4 Stage B v4：SDPO 对齐的丰富反馈

目的：针对 v3 的四个混淆因素进行修复：改用独立 LoRA EMA 教师（速率 0.01）、反向 KL、只对失败且有非空结果的样本 mask、使用 0.13–0.16 m 长方向场景、增加 `result_interpreted` 和 `hard_shuffled`。

机制门：32 个固定场景中只有 4 次尝试满足 mask；结果反馈教师期望进展 1.91 mm/step，解释反馈 3.36 mm/step，binary 3.75 mm/step，hard-shuffled 3.33 mm/step，neutral 3.11 mm/step。样本太少，区间跨零，只能说明配对流程工作，不能说明教师有稳定优势。

部署结果（6×256 训练交互/组，64 dev）：

| 条件 | 成功 | 平均进展 |
|---|---:|---:|
| frozen | 11/64 | 25.609 mm |
| result | 11/64 | 26.547 mm |
| result_interpreted | 3/64 | 13.653 mm |
| binary | 13/64 | 28.220 mm |
| hard_shuffled | 12/64 | 27.058 mm |

发现问题：结构化“保留/修订”解释改变教师最多，但部署最差，说明语言解释不是可靠的动作修订器。result 比 frozen 只多 0.937 mm，反而比 binary 少 2 个百分点。待验证：必须引入同状态真实成功兄弟，避免把当前策略对失败动作的解释误当成可执行方案。

## 六 v5 文件夹：校准后的因果反馈

### 6.1 实验目的和改动

V5 针对 v4 的两个问题：共享末端偏移使 x/y 动作接触概率不对称；教师不能稳定把数值结果翻译成动作修订。内容包括 axis-specific 接触校准、`result`、`causal`、`causal_shuffled`、`binary`、`frozen`，每个训练组 6×256=1,536 个交互，64 个 dev 场景；教师为 EMA，速率 0.01，反向 KL。

### 6.2 结果

| 条件 | 成功 | 平均进展 |
|---|---:|---:|
| frozen | 8/64 | 23.543 mm |
| result | 8/64 | 23.774 mm |
| causal | 6/64 | 21.475 mm |
| causal_shuffled | 6/64 | 21.446 mm |
| binary | 9/64 | 25.102 mm |

result 相对 frozen 仅 +0.231 mm；相对 binary −1.563 个百分点。causal 与 causal_shuffled 几乎相同，均低于 result，说明教师对“真实解释”和“错配解释”的区分没有转化为控制收益。

### 6.3 发现和待验证

V5 的重要局限是独立过渡：教师没有同一初始状态上由当前策略生成的成功解决方案，因此它更像“反馈改变动作分布”的测试，不是完整原始 SDPO 循环。这一缺口直接推动 V6。

待验证：同一 reset 状态采样多个候选，只有同状态真实任务成功候选才能作为 solution；不能把局部进展或接触直接标成成功。

## 七 v6 文件夹：分组自解决方案机制

### 7.1 实验目的和内容

V6 让一个原始观察产生 8 个候选动作，在同一状态中寻找正向候选；条件为 `solution_only`、`solution_plus_binary`、`solution_plus_causal`、`solution_plus_causal_shuffled`。温度 1.5 提高候选多样性；EMA 速率 0.01；反向 KL；每组 8×256=2,048 个训练交互，64 个 dev 场景。

### 7.2 已完成结果

| 条件 | 成功 | 平均进展 |
|---|---:|---:|
| frozen | 8/64 | 23.543 mm |
| solution_only | 6/64 | 18.814 mm |
| solution_plus_binary | 8/64 | 23.912 mm |
| solution_plus_causal | 未完成 | 无可报告结果 |

`solution_only` 的教师方向准确率从第 1 次更新的 50.0% 升到第 8 次的 70.0%，平均教师学生 TV 约 0.12–0.14；`solution_plus_binary` 方向准确率约 19.6% 升到 29.6%，TV 从 0.114 降到 0.087。说明同状态分组确实能造成可测量教师变化，但不代表部署改善。

### 7.3 关键问题

1. V6 把“正向一步进展”当作 solution，尽管整项任务仍失败，改变了成功 mask；V7 明确修复为“只有实际任务成功才算 solution”。
2. `solution_plus_causal` 只完成前 2 次更新，日志没有完成标记和最终 summary，不能与其他条件比较。
3. 即使 solution-only 教师指标上升，闭环成功率仍低于 frozen，说明自生成候选可能只是强化局部动作偏好。

待验证：使用真实任务成功兄弟、统一 feedback-only 分支，并完成 causal/shuffled 的全量更新。

## 八 v7 文件夹：SDPO 对齐、反馈隔离和 V10

### 8.1 V7 / Stage D：修复 SDPO 对齐和状态分组

V7 修复 V5/V6 的四个混淆：

- 只有同一初始图像/姿态下的真实任务成功候选才作为 solution；
- 有 solution 的行抑制环境反馈，没有 solution 的失败行才使用反馈；
- 使用 JSD `alpha=0.5`，更接近原始 SDPO 配置；
- 学生只看原始状态，教师看原始状态和 solution/反馈，部署始终无反馈。

Stage D 的 `directional_local` 分组训练在第 5 次更新后暴露 `Candidate group contains different original states`；snapshot/rerun 修复后仍只有部分日志，没有形成完整 comparison。它是工程失败而不是模型结果，但非常重要：同状态候选的状态恢复、RNG、控制器目标和 Python 状态必须一起冻结，否则“同状态自蒸馏”假设不成立。

待验证：先通过完整 snapshot 审计、候选状态哈希和多次重放，再开始任何因果效果解释。

### 8.2 V8 / Stage E：仅环境反馈 SDPO

目的：消除 V7 中“成功 sibling 主导教师上下文”的遮蔽。改用 `directional_long`：单步可以有可测量进展，但没有候选一步完成任务，因此每个有效样本都使用环境反馈而不含 solution。每条件 24 次更新、3,072 条反馈行，64 个 dev 场景。

结果：

| 条件 | 成功 | 平均进展 | 平均教师方向准确率 |
|---|---:|---:|---:|
| sdpo_result | 13/64 | 28.366 mm | 26.6% |
| sdpo_binary | 13/64 | 28.295 mm | 19.4% |
| sdpo_shuffled | 13/64 | 28.312 mm | 23.9% |
| sdpo_neutral | 13/64 | 28.295 mm | 21.2% |
| sdpo_causal | 0/64 | 2.401 mm | 37.6% |

`result` 的教师确实不同，但部署成功与 binary、shuffled、neutral 完全相同；causal 过度改变动作分布并出现 1 次安全失败。V8 反馈审计通过；早期 hash 失败来自新增 null 字段序列化，不是执行轨迹改变。

发现：教师侧的可辨识性已经出现，学生侧的稳定控制转移仍未出现。待验证：加长训练并不自动解决问题，必须同时检查动作覆盖、长期 credit assignment 和学生是否看得见目标。

### 8.3 V9 / Stage F：方向反馈诊断

目的：把“结果反馈不足”和“反馈中隐含目标方向”分开，加入 `sdpo_directional`。运行 12 次更新，64 个 dev 场景；条件为 result、binary、causal、directional。

结果：

| 条件 | 成功 | 平均进展 | 平均教师方向准确率 |
|---|---:|---:|---:|
| sdpo_binary | 13/64（20.3%） | 28.295 mm | 16.99% |
| sdpo_result | 11/64（17.2%） | 26.118 mm | 25.20% |
| sdpo_directional | 7/64（10.9%） | 17.216 mm | 53.45% |
| sdpo_causal | 1/64（1.6%） | 4.458 mm | 35.09% |

方向反馈提高教师轴向匹配，却降低部署成功；causal 也比 binary 差很多。这里的 `directional` 反馈包含目标向量导出的轴和符号，信息量明显高于纯结果反馈，不能把它当作干净的“反馈更丰富”结论。

待验证：将方向信息作为显式特权输入单独做上限实验；对纯结果反馈做视觉估计版本；加入错误方向控制和重复 seed，检查教师方向正确是否真的与闭环转移相关。

### 8.4 Stage G：冻结策略的局部自反思探针

目的：在不训练的情况下，测量模型能否把结果文本转成新的候选动作，避免把“教师改变”误认为“模型可执行”。

结果：64 场景、448 次交互、320 次反事实执行，所有场景有信息性进展。相对学生分布的期望进展差：

- 原始 result：+0.249 mm，95% bootstrap 区间 [0.086, 0.428] mm，TV 0.098；
- 规则 causal：+1.412 mm，区间 [0.960, 1.882] mm，TV 0.208；
- result_interpreted：+1.044 mm，区间 [0.619, 1.497] mm，TV 0.136；
- binary：+0.142 mm，区间 [−0.026, 0.331] mm；
- hard_shuffled：−0.082 mm，区间 [−0.309, 0.157] mm；
- neutral：−0.007 mm，区间 [−0.196, 0.178] mm。

第二次 self-reflection 探针给出的建议动作闭环成功率为 34.375%，但这些建议没有进入部署训练。说明“模型能在冻结状态下提出更好的候选”与“SDPO 学生能长期执行这些候选”是两个不同命题。

待验证：把 reflection 作为独立策略或 teacher 上限，做同状态配对部署；不能只看反事实期望进展。

### 8.5 Stage H：反思反馈训练

目的：比较 `sdpo_reflection`、`sdpo_result`、`sdpo_binary` 和 frozen，判断语言化自反思是否比结构化结果更可用。每个训练组 10×32=320 个交互，64 个 directional_local dev 场景。

结果：frozen、binary、result、reflection 全部为 14/64；平均进展均约 3.7335 mm。reflection 教师平均 TV 约 0.34，明显高于 result 约 0.084，但部署没有改善。

发现：反思文本提高了教师分布差异，却没有改变闭环行为；这再次证明分布指标不能替代部署指标。待验证：需要更长训练和更合适的长场景，但首先应解决方向识别和动作覆盖，而不是继续堆叠文本解释。

### 8.6 V10 / Stage I：方向、错误方向和结果反馈的受控对比

目的：在 V9 基础上加入 `sdpo_wrong_directional`，检查方向信息是否真的被因果使用；每组 12×128=1,536 个训练交互，64 个 dev 场景。

结果：

| 条件 | 成功 | 平均进展 | 平均教师方向准确率 |
|---|---:|---:|---:|
| sdpo_binary | 13/64（20.3%） | 28.295 mm | 24.8% |
| sdpo_result | 11/64（17.2%） | 26.195 mm | 31.3% |
| sdpo_wrong_directional | 9/64（14.1%） | 22.136 mm | 41.7% |
| sdpo_directional | 6/64（9.4%） | 16.577 mm | 63.7% |
| sdpo_causal | 0/64 | 1.844 mm | 42.8% |

结果反馈仍未超过 binary；directional 教师诊断最强但部署最差；wrong_directional 处于中间，说明“教师方向准确率”与闭环成功不是单调关系。causal 反复导致动作重分配和性能坍缩，说明保留/修订解释可能是强但错误的控制信号。

待验证：需要跨 seed 的配对 test、动作覆盖和长期轨迹分析；还需把目标方向、动作标签、接触几何分离，才能判断是 teacher 错误还是 student 执行不足。

### 8.7 试跑、重跑和未完成条件的处理

目录中还保留了若干用于查错的 smoke、failed launcher 和 rerun 产物。它们对工程判断有价值，但不应混入正式效果表：

- base 的 `model_smoke`、`v4_sdpo_smoke`，v6 的 `v6_grouped_smoke`/`v2`/`v3`，以及 v7 的 `v7_smoke_result`/`v7_debug_result` 只验证模型加载、梯度、状态恢复或极小样本闭环；例如 v7 smoke 为 3/8，不能代表 64 场景结果。
- v5 的 `stage_b_v5_calibrated.failed_import_path.*` 是导入路径失败后的保留目录，没有科学结果。
- v7 Stage D 的初次运行、rerun 和 snapshotfix 都没有形成完整多条件 comparison；最终应以通过状态一致性审计的完整重跑为准。
- V8 的单条件初跑、V9 的早期目录、V10 的前两次目录均被后续完整目录覆盖；本报告使用 `stage_e_v8_feedback_only_rerun_01`、`stage_f_v9_directional_feedback_02` 和 `stage_i_v10_directional_control_03`。
- v6 `solution_plus_causal` 只有 2 次更新并中止；Stage H 的 smoke 只有 2 次更新，最终 pilot 才用于对比。

这一区分保证“管线能运行”“教师分布有变化”和“无反馈部署变好”三个层级不会被混写。

## 九 P0/P1：视觉、标签和模型容量诊断

### 9.1 P0 冻结方向上限

在相同 64 个 held-out `directional_long` 场景上：

| 条件 | 闭环成功 | 平均进展 |
|---|---:|---:|
| Qwen 3B frozen RGB | 11/64（17.2%） | 0.02561 m |
| Qwen 3B frozen target direction | 42/64（65.6%） | 0.09135 m |
| Qwen 3B oracle-supervised target direction | 43/64（67.2%） | 0.09613 m |

冻结目标方向已经带来几乎全部提升，oracle 监督只增加 1 个成功场景。这说明早先 67.2% 主要是输入信息特权，不是 LoRA 训练效果。

### 9.2 P1 干净方向标签

把状态相关、可能偏向接触动作的 `oracle_action` 换成确定性的主导目标方向 `direction_action`：

- held-out 方向准确率：71.9% → 100.0%；
- target-direction 闭环：42/64 → 57/64（65.6% → 89.1%）；
- 平均进展：0.09135 m → 0.12802 m；
- 分方向成功：+x 13/14、+y 16/18、−y 14/14、−x 14/18。

有趣之处是原始 oracle-label 准确率反而由 71.9% 降至 65.6%，但行为变好；因为原始 oracle 选择的是某个状态下的接触/进展动作，不等同于主导目标方向。标签质量比“更贴近模拟器 oracle”更重要。

中间状态轨迹实验收集 503 个 train、469 个 held-out expert intermediate states，方向准确率达 100%，但闭环仍为 57/64；因此继续增加方向分类数据不能解决接触几何、动态重规划和长时域执行。

### 9.3 Qwen 3B 与 7B 容量对比

- frozen RGB：3B 为 11/64，7B 为 0/64；7B 在当前 prompt/action 设置下全部偏向 E。
- 干净方向监督：3B RGB 闭环 8/64、目标方向 57/64；7B RGB 闭环 25/64、目标方向 57/64；7B held-out RGB 方向准确率 100%，3B 仅 21.9%。
- 原始 oracle 监督：3B RGB 16/64、目标方向 43/64；7B RGB 10/64、目标方向 44/64。

结论：容量只有在监督目标与行为目标一致时才帮助视觉方向；容量本身不能解决 RGB grounding、动作循环和长时域执行。

发现问题：方向特权可以把任务做得很好，但它绕过了本实验想验证的视觉反馈链；干净方向监督提高了分类和闭环，却仍留下接触与重规划失败。待验证：先建立不含目标真值的可靠视觉方向估计，再重新比较结果反馈和二元反馈；同时必须增加跨 seed 和更长 horizon，避免把单一 64 场景上限当作泛化结论。

## 十 跨版本发现的问题

### 10.1 任务和物理层

1. 初始末端距离过远时，反馈几乎全为无接触；近接触又引入机器人-方块重叠 reset。
2. 末端高度、轴向偏移和动作缩放直接决定接触覆盖；早期动作缩放 bug 已修复，但旧烟雾结果不能混入学习结论。
3. 局部进展不是任务成功。V6 的 solution mask 错把局部正进展当作成功，必须永久区分 `success`、`progress` 和 `contact`。
4. 方向反馈含目标轴/符号，属于强几何信息，不能包装为普通结果反馈。

### 10.2 教师和算法层

1. `teacher_student_TV`、loss、teacher oracle accuracy 增加，只能证明提示改变了分布，不能证明动作可执行。
2. causal 解释常常强烈改变 A/C/D 分布，却反复降低部署；可能存在坐标、时间、反事实语义或 prompt 方向错误。
3. V5/V6 的反向 KL 与 V7 的 JSD 不同；不同算法配置不能直接横向比较绝对成功率。
4. shuffled 必须保留完整结果元组、相同 attempted action 和 success 分层，并记录 donor 改变率；否则控制组可能只是重复主导无效元组。
5. grouped SDPO 要求同一图像、姿态、目标、RNG、控制器缓存和电机目标都恢复；Stage D 暴露的状态分组错误说明仅保存一个整数状态句柄不够。

### 10.3 视觉和执行层

1. 3B frozen RGB 主要输出 A；7B frozen RGB 主要输出 E，说明模型大小不等价于视觉地基。
2. 干净方向标签能把目标方向闭环推到 89.1%，但中间状态数据不再提升，表明接触几何、重规划和动作停止条件成为下一瓶颈。
3. 方向教师准确率和成功率不单调；教师可能学到目标轴，却没有学到“从当前末端位置如何接触并持续推动”。

## 十一 待实验验证问题和优先级

### P0 必须先验证

1. **同状态重放完整性**：对每个 8 候选组验证图像、物体位姿、末端位姿、RNG、控制器目标、elapsed steps 全部一致；重复动作误差写入审计。
2. **纯结果反馈的教师门**：在不提供目标方向的情况下，用固定配对状态比较 result/binary/shuffled/neutral 的教师动作变化和反事实长期成功；至少 3 个 seed，不能只看单步进展。
3. **反馈语义方向**：逐项测试原始 RGB、目标方向特权、错误方向、遮挡目标和遮挡方块，区分反馈理解与视觉识别。
4. **长时域 credit assignment**：记录首次接触、连续推动、停止时机、目标过冲和动作切换；当前 12 步或 32 步成功率不能解释失败位置。

### P1 之后验证

1. 在同一场景、同一 reset、同一 student 初始权重下比较 result 与 binary，至少 3 seed；报告场景配对差异和区间。
2. 用感知估计的 contact/motion/progress 替代模拟器真值，验证结果反馈是否依赖不可部署的 privileged signal。
3. 用干净 `direction_action` 作为学生可学性上限，再测试 SDPO；禁止把 oracle action 输入当作真实部署结论。
4. 比较 3B/7B 只在相同 prompt、相同 LoRA、相同动作契约下进行；扩展到 OpenVLA/SimpleVLA 前先做动作归一化和输入接口审计。
5. 如果仍要声称“丰富反馈优于奖励或二元反馈”，需增加相同反馈字段的 dense-reward RL 基线；当前实验没有这个对照。

## 十二 最终判断

从 base 到 V10，实验链条形成了清晰的否证路径：

1. v1 先发现反馈稀疏；
2. v2 提高接触但发现 reset 混淆；
3. v3 改成可完成的方向场景，证明场景本身能提高成功；
4. v4/v5 对齐 SDPO 和几何协议，发现教师解释仍不能转成部署收益；
5. v6/v7 引入同状态自解决方案，发现 solution mask 和 snapshot 一致性是独立问题；
6. V8 隔离环境反馈，证明教师分布不同但部署仍可相同；
7. V9/V10 加方向和错误方向控制，证明教师轴向准确率提升不等于控制成功；
8. P0/P1 最后把主要瓶颈定位到视觉 grounding、标签定义和长时域执行。

因此当前最稳妥的汇报口径是：**SDPO 结果反馈机制已完成多轮工程和机制诊断，但在该五动作 Qwen PandaPush 试点上尚未证明能提升无反馈闭环控制；后续应优先验证视觉与执行瓶颈，再决定是否投入更大规模 SDPO 训练。**

## 十三 主要证据索引

- 协议和总体边界：`sdpo_result_feedback_pilot/doc/EXPERIMENT_PLAN.md`、`sdpo_result_feedback_pilot/doc/IMPLEMENTATION_NOTES.md`
- base 结果：`sdpo_result_feedback_pilot/artifacts/stage_b_v1/comparison.json`、`stage_b_v2_near_contact/comparison.json`、`stage_b_v3_directional/comparison.json`、`stage_b_v4_sdpo_aligned/comparison.json`
- V5：`sdpo_result_feedback_pilot_v5/doc/V5_EXPERIMENT.md`、`artifacts/stage_b_v5_calibrated/comparison.json`
- V6：`sdpo_result_feedback_pilot_v6/doc/V6_EXPERIMENT.md`、`artifacts/stage_c_v6_grouped/*/summary.json`、`solution_plus_causal.log`
- V7/V8/V9：`sdpo_result_feedback_pilot_v7/doc/V7_EXPERIMENT.md`、`doc/V8_EXPERIMENT.md`、`doc/V9_RESULTS.md`、`artifacts/stage_e_v8_feedback_only_rerun_01/comparison.json`、`stage_f_v9_directional_feedback_02/stage_f_v9_comparison.json`
- V10：`sdpo_result_feedback_pilot_v7/artifacts/stage_i_v10_directional_control_03/stage_i_v10_comparison.json`
- 自反思：`sdpo_result_feedback_pilot_v7/artifacts/stage_g_local_probe_01/probe/report.json`、`stage_g_self_reflection_02/report.json`
- P0/P1 与容量：`sdpo_result_feedback_pilot_v7/doc/P0_P1_CAPACITY_RESULTS.md`、`doc/FAST_ORACLE_VALIDATION.md`
