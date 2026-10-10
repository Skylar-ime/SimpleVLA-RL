# SDPO 论文与代码汇报内容

扩展版：[SDPO_VLA_全链条汇报内容.md](../sdpo-vla/sdpo_vla_research/doc/SDPO_VLA_全链条汇报内容.md)，涵盖 VLA 难点、相似工作、研究缺口、迁移方案与实验设计。本文件保留为 SDPO 原理论文讲解材料。

适用：约 12–15 分钟技术组会。主线：**标量奖励信息太少；让模型看过反馈后回评原答案，再把这种判断蒸馏回原策略。**

论文：Hübotter 等，*Reinforcement Learning via Self-Distillation*，arXiv:2601.20802v2（2026）。代码：[lasgroup/SDPO](https://github.com/lasgroup/SDPO)，本文核对版本 `7c457fc`。下文实验数字均为论文报告值，未做独立复现。

## 第 1 页｜标题与一句话结论

**屏幕文字**

- SDPO：通过自蒸馏进行强化学习
- 核心：把环境反馈变成逐 token、逐候选词的训练信号
- 无需外部强教师或额外奖励模型

**讲法**：模型先答题，环境给反馈。再让“看过反馈的自己”评价刚才每一步选词是否合理；最后把评价学进“没看过反馈的自己”。

**配图**：仓库的 [SDPO 总览图](https://github.com/lasgroup/SDPO/blob/main/figures/sdpo-fig.png)。

## 第 2 页｜背景：GRPO 的信息瓶颈

**屏幕文字**

- 代码题常只给 `通过 / 未通过`，GRPO 据此给整条回答一个组内相对优势。
- 同一回答中的正确思路和错误步骤得到相近方向的更新，定位错误困难。
- 一组尝试若全为 0 分，组内优势也为 0，难以继续学习。

**讲法**：单元测试明明能指出“第 73 行除零”或失败样例；只把结果压成 0 分，就丢掉了错误发生在哪里的信息。

**配图**：左侧“代码、测试反馈”，右侧“0/1 分数”，突出信息损失。

## 第 3 页｜任务设置：从 RLVR 到 RLRF

**屏幕文字**

- RLVR：问题 `x`、回答 `y`、标量奖励 `r`。
- RLRF：额外保留文本反馈 `f`，例如报错、失败用例、评审意见。
- 没有文本反馈时，同题其他成功回答也可充当示范。

**讲法**：SDPO 的“反馈”有两种来源。代码环境提供解释性文字；只提供分数的环境则用同批次成功样本作线索。成功样本仍靠奖励筛选，不能说完全不用奖励。

**配图**：论文的 [RLVR / RLRF 对照图](https://github.com/lasgroup/SDPO/blob/main/figures/sdpo-fig-training-loop.png)。

## 第 4 页｜核心机制：同一能力，两种上下文

**屏幕文字**

1. 学生：只看题目 `x`，生成原回答 `y`。
2. 自教师：看 `x + f`，**重新计算原回答 `y` 的概率**。
3. 对比两者在原回答每一步的下一词分布，更新学生。

**讲法**：教师不需要再生成一份完整改写。它对原回答做一次带反馈的前向计算。若学生很想写某个词，教师看过错误提示后不赞同，这个位置就收到清楚的纠偏信号。

**配图**：两条上下文汇入同一段原回答，底部画“学生概率 / 教师概率”对比；可裁用论文 Figure 2。

## 第 5 页｜公式：把反馈变成密集优势

**屏幕文字**

设学生分布 `p_t = πθ(· | x, y<t)`，自教师分布 `q_t = πteacher(· | x, f, y<t)`。

`L_SDPO = Σ_t KL(p_t || stopgrad(q_t))`  （论文的基础形式）

对候选词 `a`，直观优势：`A_t(a) = log q_t(a) − log p_t(a)`。

- `A_t(a) > 0`：教师更认可，增加该词概率。
- `A_t(a) < 0`：教师更不认可，降低该词概率。

**讲法**：GRPO 对一条回答主要给一个成败方向；SDPO 可在每个位置比较多个候选词。`stopgrad` 只更新学生，避免教师反过来迎合原错误。上式 `KL(学生 || 教师)` 是反向 KL；实际实验按场景使用 Jensen–Shannon 或反向 KL。

**配图**：引用论文的 [token 概率差示例](https://github.com/lasgroup/SDPO/blob/main/figures/sdpo-fig.png)。

## 第 6 页｜一个代码错误的直观例子

**屏幕文字**

原回答含错误边界：`range(1, n + 1)`；测试反馈提示结果不应包含 `n`。

- 学生原来偏向 `n + 1`。
- 自教师看到反馈后，更偏向 `n`。
- 蒸馏把局部修改方向写进参数，下一次无需反馈也能更早避错。

**讲法**：这里用简化示意说明“在哪一步改、改向哪里”。论文真实示例是区间边界错误；展示原图时以原图中的代码和颜色标注为准。

**配图**：论文 Figure 4 的 [密集归因示意](https://arxiv.org/pdf/2601.20802)。

## 第 7 页｜训练流程与代码对应

**屏幕文字**

`采样 G 条回答 → 环境评分与反馈 → 按同题成功样本/失败反馈组装教师提示 → 教师回评原回答 → 蒸馏更新学生 → 循环`

**代码地图**

- [训练入口](https://github.com/lasgroup/SDPO/blob/main/training/verl_training.sh)：调用 `verl.trainer.main_ppo`。
- [教师数据构造](https://github.com/lasgroup/SDPO/blob/main/verl/trainer/ppo/ray_trainer.py#L672)：同题成功回答按 `uid` 查找，填入模板；将原 `responses` 接到教师提示后，生成 `teacher_input_ids`。
- [损失计算](https://github.com/lasgroup/SDPO/blob/main/verl/trainer/ppo/core_algos.py#L1085)：KL / JSD、top-K 尾部、重要性采样权重、mask。
- [学生与教师前向](https://github.com/lasgroup/SDPO/blob/main/verl/workers/actor/dp_actor.py#L808)：教师在 `torch.no_grad()` 下计算，对学生反向传播。

**讲法**：关键不是新建一条训练流水线，而是在 verl 的 rollout 和 actor 更新之间加入教师提示与分布匹配。代码虽仍计算 GRPO 优势等数据，`loss_mode=sdpo` 时 actor 的策略损失走自蒸馏分支。

## 第 8 页｜工程设计：稳定性、显存、有效样本

**屏幕文字**

- **稳定性**：EMA 教师缓慢跟随学生；代码也支持 trust-region 教师。
- **显存**：只取学生 top-K 候选及一个尾部概率桶，教师在同一候选集合上比较。
- **有效样本**：没有成功示范、也没有可用文本反馈的样本被 `self_distillation_mask` 排除。
- **偏移修正**：可对生成分布与当前策略的差异使用截断重要性权重。

**讲法**：EMA 的教师与学生同源、持续更新，但实现上保留独立参数。top-K 是近似，不等于完整词表 KL。无任何有用反馈时，SDPO 这一损失没有训练信号。

**代码来源**：[EMA 更新](https://github.com/lasgroup/SDPO/blob/main/verl/workers/actor/dp_actor.py#L132)、[掩码构造](https://github.com/lasgroup/SDPO/blob/main/verl/trainer/ppo/ray_trainer.py#L773)、[top-K 损失](https://github.com/lasgroup/SDPO/blob/main/verl/trainer/ppo/core_algos.py#L1106)。

## 第 9 页｜实验：没有丰富反馈也有效

**屏幕文字**

- 场景：SciKnowEval 四类科学问答、工具调用；同题成功回答充当反馈。
- 论文汇总最终准确率：SDPO **70.2%**，强 GRPO **66.6%**。
- Olmo3-7B-Instruct 的 Chemistry：SDPO 约 **50 分钟**达到 GRPO **5 小时**准确率。
- 回答长度平均短逾 3 倍；Chemistry 的 Olmo 个例最高约 11 倍。

**讲法**：这里比的是训练效果和论文条件下的收敛，不是无条件节省总算力。每轮仍有教师前向计算；不同任务也有个别指标未胜过 GRPO。

**配图**：[Chemistry 训练曲线](https://github.com/lasgroup/SDPO/blob/main/figures/chemistry-accuracy-response.png)；表格可用 [仓库图](https://github.com/lasgroup/SDPO/blob/main/figures/table-no-rich-feedback.png)。

## 第 10 页｜实验：丰富反馈时更强

**屏幕文字**

- 场景：LiveCodeBench v6，Qwen3-8B，训练时用公开单测反馈，验证时用保留单测。
- 最终准确率：SDPO **48.8%**，GRPO **41.2%**，高 **7.6 个百分点**。
- 达到 GRPO 最终准确率所需生成次数约少 **4 倍**。
- 消融：logit 级优于 token 级，token 级优于序列级；序列级仍优于 GRPO。

**讲法**：消融支持两层收益：反馈文字本身有帮助；把信息细化到候选词概率后，收益进一步增加。这里的 4 倍指标是“生成次数”，不要读成训练总时长缩短 4 倍。

**配图**：[LCBv6 曲线](https://github.com/lasgroup/SDPO/blob/main/figures/lcbv6-accuracy.png)。

## 第 11 页｜测试时自蒸馏

**屏幕文字**

- 针对单个难题，反复尝试、获取反馈、做小步参数更新。
- 对二值奖励难题，首次正确前，普通 RLVR 收不到正向成败信号；SDPO 仍可用失败反馈修正。
- 极难题达到 **22% 发现概率**时，论文报告所需生成次数约为 best-of-k / 多轮提示的 **1/3**。

**讲法**：这里是 test-time training，模型参数随题目更新；多轮提示只变上下文，best-of-k 固定模型反复采样。这个 3 倍有特定概率门槛与数据集，不能泛化成所有题均快 3 倍。

**配图**：[测试时发现率曲线](https://github.com/lasgroup/SDPO/blob/main/figures/very-hard-questions.png)。

## 第 12 页｜特点、边界与结论

**屏幕文字**

**特点**

1. 借模型已有的“看反馈纠错”能力生成教师信号，不依赖外部强教师。
2. 从整条回答的标量优势细化到每步候选词的概率差。
3. 兼容文本反馈；纯标量环境可借同题成功样本自举。

**边界**

- 基础模型的上下文理解弱时可能不如 GRPO；论文在 Qwen2.5-1.5B 上观察到退化。
- 误导性反馈会影响自教师；全组失败且无文本反馈时，当前实现无法产生蒸馏信号。
- 增加教师前向与 EMA 参数开销；论文主要在 4 张 GH200 上验证。

**收束讲法**：SDPO 的关键创新是“反馈条件下的自教师 + 密集信用分配”。它让模型从“知道错了”推进到“知道哪些选词更可能错”，但效果取决于反馈质量和模型自身理解力。

---

## 备查：公式与配置别混淆

- 论文主公式是 `KL(学生 || 教师)`；代码配置 `alpha=0` 为前向 KL，`alpha=0.5` 为广义 JSD，`alpha=1` 为反向 KL。
- 无丰富反馈实验使用 top-K=100、JSD、EMA 更新率 0.05；丰富反馈实验使用 top-K=20、反向 KL、EMA 更新率 0.01。具体值见论文附录 Experiment Details 与 [实验脚本](https://github.com/lasgroup/SDPO/tree/main/experiments)。
- 代码配置 `algorithm.adv_estimator=grpo` 主要沿用 verl 训练管线；`actor.policy_loss.loss_mode=sdpo` 才决定 actor 策略损失改走自蒸馏。
- 论文与仓库的 `alpha`、`teacher_update_rate` 是两个不同超参：前者选散度，后者控制 EMA / trust-region 教师更新。

## 来源

1. [论文 arXiv 页面](https://arxiv.org/abs/2601.20802)，方法见 §2，标准 RLVR 实验见 §3，丰富反馈实验见 §4，测试时实验见 §5，局限见 §7。
2. [仓库 README](https://github.com/lasgroup/SDPO/blob/main/README.md)，包含官方结果图、运行命令、配置说明。
3. [核心损失代码](https://github.com/lasgroup/SDPO/blob/main/verl/trainer/ppo/core_algos.py#L1085) 与 [教师批次构造](https://github.com/lasgroup/SDPO/blob/main/verl/trainer/ppo/ray_trainer.py#L672)。
