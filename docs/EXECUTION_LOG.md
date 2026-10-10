# 执行日志

## 范围

- 分支：`ex/fuxian`
- RoboTwin2.0`lift_pot`
- 硬件：8个H800，远程主机`h800-1`
- 订单：`val_only`然后原始二进制奖励基线

## 路径迁移

服务器文件夹移动到：

```text
/public/home/s2250018/wsy/VLA/
```

当前路径：

```text
repo:      /public/home/s2250018/wsy/VLA/SimpleVLA-RL
RoboTwin:  /public/home/s2250018/wsy/VLA/RoboTwin
Curobo:    /public/home/s2250018/wsy/VLA/curobo-src
OpenVLA:   /public/home/s2250018/wsy/VLA/openvla-oft
outputs:   /public/home/s2250018/wsy/VLA/outputs
```

RoboTwin 的当前提交：`96c1feab536306b50c26af200044fcdf126e8904`.

Canonical SFT工作检查点：

```text
/public/home/s2250018/wsy/VLA/checkpoints/lift_pot_sft_merged
```

- 它符合旧的`/wsy/checkpoints/lift_pot_sft_work`在所有72个非备份文件中，使用SHA-256。
- 老的原始/工作目录仍然作为外部只读档案存在`VLA`未来的命令只使用标准的VLA路径。

## 完成

- SSH 链路验证：本地`zhongxin`→ 登录节点 →`h800-1`.
- 库存验证：8个GPU，每个约81.5GB。
- Conda 環境驗證成功：`simplevla`, PyTorch`2.4.1+cu124`.
- Curobo 从迁移路径重新安装。
- RoboTwin嵌入式资产/配置路径修复。
- `val_only`完成在256个验证迭代中的端到端训练。

## 结果

```text
robotwin2_lift_pot_train_iid: 0.000
robotwin2_lift_pot_eval_ood:  0.000
all:                          0.000
```

早期二进制基线启动在训练前停止，这并未证明过滤去除了所有样本。

## 重量审核

- 主要检查点：4个safetensor碎片。
- LoRA适配器存在于`lora_adapter/`.
- 主要`lm_head`与适配器不同`base_layer`.
- 主要`lm_head`≈适配器`base_layer + 0.5 × (B @ A)`合并残差均值`2.8e-5`.
- 这直接验证了`lm_head`适配器存储了439个LoRA A/B对，但只有一`base_layer.weight`张量(`lm_head`)其他层仍然缺少一个匹配的基张量来进行完整的合并审计。
- 决策：在没有完整的基础模型比较之前，不要再次合并适配器；当前证据表明`lm_head`已合并，不是每个LoRA层。

## 当前诊断

`val=0`未由LoRA合并引起。下一次审计目标：自定义OpenVLA-OFT动作令牌解码器，特别是切片`[..., -256-64:-64]`，加上提示/动作位置对齐。

## 解码器寻找

- 分词词典`32001`，填充`32000`模型未填充动作词典大小是`31744..31999` (`vocab_size=32000`, 256个桶).
- 解码器切片`[-256-64:-64]`选择logits`31744..31999`偏移和位翻转在内部是自洽的。
- 这段自定义代码是复制行为；没有静默补丁。
- 下一个诊断：检查提示/动作位置对齐和确定性(`do_sample=false`)动作统计在一个固定事件。

## 诊断结果

- 验证路径强制`do_sample=false`已经；不需要单独的确定性设置。
- 两个GPU诊断工人在边缘位置生成了大部分行动ID`31744` / `31999`解码动作范围为非零值(`min≈-2.58`, `max≈2.66`).
- 因此模型调用和解码器发出动作；失败是控制质量或检查点/任务对齐，而不是所有零张量生成。
- 简短的诊断在确认这一点后停止了；完整的值结果仍然是之前的256期。`0.000`.
- 远程临时仪器和数据集重置恢复。

## 官方路径比较

- 暂时称为官方OpenVLA-OFT`predict_action`在相同的两个验证周期。
- 输出形状/范围匹配的RL路径(`1×25×14`，大约`-2.285..2.659`)成功仍然存在`0.0`.
- 决策：没有证据表明`generate_action_verl`单独会导致零成功。所有临时远程编辑恢复。

- `demo_clean + seen`简短的比较也返回了`0.0`在两集。
- 决策：随机/清洁场景配置或指令拆分并不能解释零结果。

## 二进制基线重试

- 8个GPU的原始参数重试在PID下启动`3678888`.
- 将此纯Markdown文本翻译成简体中文。仅输出翻译，不添加说明、标题、示例或Markdown。`/public/home/s2250018/wsy/VLA/outputs/lift_pot_binary_baseline_retry`.
- 初始256集验证完成：所有指标`0.000`.
- 第一轮训练批次完成：`512/512`发布失败；`Accuracy distribution: 0.00:64`过滤批次大小`0`.
- 严格的基准线在这一诊断批次后停止(PID`3678888`)没有生成演员更新/检查点。

## 当前风险

被选中的公共`lift_pot`SFT检查点在该分支的RoboTwin包装器和官方解码器下没有成功验证或首次训练会话。`filter_accuracy=True`，所有零训练批次被丢弃，因此原始二进制基线无法更新。

很可能的不兼容性：当前RoboTwin结账是提交`96c1fea`日期为2026年9月5日，而公共检查点发布于2025年8月12日。仓库设置不锁定RoboTwin提交。需要在获取/使用与基准匹配的RoboTwin 2.0源代码之前，才能声称基准复现。

## A/B 实现

- 公共模型名称包含`aloha`官方任务文档列出了Aloha-AgileX和Piper的实现方式。([任务文档](https://robotwin-platform.github.io/doc/tasks/lift_pot.html)).
- 当前分支配置为双`piper,piper`距离`0.6`添加隔离`demo_randomized_aloha.yml`使用`aloha-agilex`为了比较。
- 全256集的ALOHA A/B在初始化过程中停止：种子`100105002`，对象`066_vinegar`，未能通过RoboTwin稳定性检查；无模型得分。
- 添加选择性`SIMPLEVLA_VAL_TRIALS`诊断覆盖；默认保持不变`128`ID +`128`OOD.
- 两次尝试未能达到推出：第一次有`val_batch_size=256`重试违反了8路数据并行分块。
- 最低合法的ALOHA诊断是8个发作(`4 IID + 4 OOD`)启动PID`3930022`，输出`/public/home/s2250018/wsy/VLA/outputs/lift_pot_aloha_diag8`.
- 阿拉哈 8集诊断也未能通过稳定性检查(`100100007`, `001_bottle`)存储的可行种子在当前源/资产下无法移植到本实施例。

## RoboTwin 版本 A/B

- 添加选择性`SIMPLEVLA_ROBOTWIN2_PATH`默认嵌入路径不变。
- 孤立构建`/public/home/s2250018/wsy/VLA/RoboTwin-20250810`从提交`6be6922`，叠加所需的SimpleVLA修改，重用当前资产。
- 启动了至少8集的Piper诊断，PID`3969545`，输出`/public/home/s2250018/wsy/VLA/outputs/lift_pot_hist20250810_diag8`.
- 历史源码Piper诊断完成`8/8`带有`0.000`这不是一个有效的ALOHA检查点测试，因为配置重写仍然保留了Piper。
- 启动了修正后的1-GPU/1种子历史源+ALOHA配置测试，PID`3995263`，输出`/public/home/s2250018/wsy/VLA/outputs/lift_pot_hist20250810_aloha1`.
- 历史源码 + ALOHA 测试完成所有 1IID 和 1OOD 情节：`0.000/0.000`，没有初始化错误。
- 结论：当前的零值不仅由RoboTwin提交或实现所解释；很可能是检查点代码/处理器/动作合同不匹配。暂不主张论文复制。
- 论文比较：公开报告给出了RoboTwin2.0`lift_pot`OpenVLA-OFT SFT 基线`10.1%`和我们的`64.1%`它声明了8×A800，1000个SFT场景，25个片段，256个动作令牌。([报告](https://arxiv.org/abs/2509.09674))我们的当前SFT是`0%`所以复制目前还不可用。
- 修正解码器A/B：添加了选择性`SIMPLEVLA_USE_PREDICT_ACTION=1`它叫`predict_action`使用加载的 proprio 投影仪。默认的 RL 路径保持不变。
- 发布1-GPU历史源码 + ALOHA + 修正`predict_action`测试，PID`4037215`，输出`/public/home/s2250018/wsy/VLA/outputs/lift_pot_hist20250810_aloha_predict1`.
- 修正`predict_action`测试完成，1个IID + 1个OOD：`0.000/0.000`进程正常退出。
- 最终决定：停止昂贵的训练尝试。当前的公共检查点在任何情况下都无法重现预期的非零SFT，无论是当前还是与源匹配的时间段，Piper还是ALOHA。下一步工作必须确定确切的训练时间处理器/模型修订版，或者获取匹配的检查点/数据。

## 参考RL检查点审计

- 公共RL参考：`Haozhan72/twin2_lift_pot_seed1k_sft_20k_batch64_rl_seed1000_max200_step40`，版本号`ed1d1cd992c3a40821849a58dad82521f9a69160`.
- 这是`modeling_prismatic.py`，处理器配置、预处理器和数据集统计与公开的SFT原始值完全匹配。
- RL配置文件明确存储`use_proprio=true`, `proprio_dim=14`SFT配置缺少这些字段，但SimpleVLA在模型加载前添加了它们，并加载投影文件。
- 服务器无法连接到Hugging Face；服务器下载PID`4087628`退出。
- 将28.16 GB的RL参考本地下载到`/home/skylar/@lab/hf-cache/lift_pot_rl_reference`然后将其转移到服务器下`VLA/checkpoints`只用于验证环境。
- RLL 索引包含所有四个`proprio_projector`张量。加载这个完整的检查点。`actor_rollout_ref.model.resume=True`否则，工人错误地期望有一个单独的投影仪文件。
- 本地下载完成：7/7个碎片，28.16 GB。每个碎片的SHA-256校验码与Hugging Face LFS对象ID匹配。
- 服务器原始路径：`/public/home/s2250018/wsy/VLA/checkpoints/lift_pot_rl_reference_original`.
- 工作路径：`/public/home/s2250018/wsy/VLA/checkpoints/lift_pot_rl_reference_work`模型碎片是原始文件的符号链接，而可变的配置/处理器/代码文件则是独立的副本。仓库检查点兼容性文件仅应用于工作路径。
- 准备好的`/tmp/run_lift_pot_rl_reference_val2.sh`在`h800-1`: 首先运行当前的皮珀，然后仅在需要时运行历史上的Aloha；每个使用一个IID和一个OOD在一个GPU上。

## 历史上的ALOHA验证重试

- RoboTwin提交`6be6922`使用了ALOHA的原始`demo_randomized.yml`当前嵌入的SimpleVLA配置使用了Piper。历史上的ALOHA是更接近2025年的检查点匹配。
- 8个IID和8个OOD在种子偏移量为0的情况下，在推理前停止：训练种子`100100007`不断重复`001_bottle`不稳定。这是环境初始化失败，而不是模型得分。
- 添加选择性`SIMPLEVLA_VAL_SEED_OFFSET`默认`0`保持官方行为不变。
- 重试了8个IID和8个OOD，偏移量为`8`，PID`4166112`它也停止了推理：种子`100088`, `100084`并且`100100036`产生了不稳定且杂乱的物体。
- 决策：种子列表不能移植到历史的ALOHA源/资产。不要将这些未完成的运行报告为模型失败或正式得分。

## 规则

- 保留原始/工作检查点。
- 使用迁移`VLA`路径为仓库、环境和输出。
- 在进行任何长达8个GPU的基准测试之前，先运行简短的诊断。
- 在这里添加决策、命令、结果和障碍；请保持条目简短。

## RL 参考检查点验证

- 本地下载的公共RL检查点已转移至`/public/home/s2250018/wsy/VLA/checkpoints/lift_pot_rl_reference_original`所有7个模型分片的SHA-256值都通过。工作路径使用符号链接来表示不可变的分片，并且有独立的可变兼容性文件。
- 全量RL加载需要`actor_rollout_ref.model.resume=True`8个GPU的加载已完成。
- 默认分支解码器(`generate_action_verl`): 历史上的RoboTwin`6be6922`+ ALOHA 固定重置种子：`0/16`成功。当前RoboTwin + Piper：`0/2`.
- 官方`predict_action`A/B在相同的ALOHA 16种子历史中也正常完成，`0/16`成功。因此，解码器的选择本身并不是原因。
- Ray启动超时在30秒阈值处发生了一次；重试完成。将超时视为基础设施噪音，而非任务失败。
- 决策：不开始100轮二进制基线。公共参考仍然不符合预期的论文行为；确切的训练时间模型/处理器/环境合同尚未解决。

## 附加检查点/评估器检查(2026-09-15)

- SFT检查点文件列表：4个模型碎片 +`proprio_projector--20000_checkpoint.pt`没有连续`action_head` (预期的离散动作令牌SFT).
- SFT 统计`lift_pot_1k`完全匹配Piper 14-D的关节范围；公开型号名称`...sft_aloha...`这不是Aloha体化的证据。
- 参数SFT验证使用官方采样温度`1.6`, 皮珀固定了16颗种子：`0/16`.
- 老RoboTwin资产(2025-08-11)转移到`/public/home/s2250018/wsy/VLA/checkpoints/robotwin2-{embodiments,objects}-20250811.zip`SHA-256 匹配本地快照。ALOHA 资产不匹配不是剩余的解释。
- 尝试在隔离的检查点副本中使用官方RoboTwin OpenVLA-OFT评估器。历史策略包不完整(`prismatic.models`)全包要求不可用`dlimp`没有官方成绩产生；没有修改实时环境。
- 决策：保持基线停止。当前证据是可重复的`0%`诊断，不是纸张复制。下一步有效操作需要精确的作者训练/评估环境或匹配的检查点/数据包；不要启动100个epoch。
