<div align="center">

<img src="figs/logo.png" width="260"/>

## 简单视觉语言行动：面向视觉-语言-动作模型的开放RL框架

[![论文](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.09674) [![GitHub](https://img.shields.io/badge/SimpleVLA--RL-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/PRIME-RL/SimpleVLA-RL) [![Hugging Face 作品集](https://img.shields.io/badge/Models-fcd022?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86) [![推特](https://img.shields.io/badge/Twitter-%23000000.svg?style=for-the-badge&logo=x&logoColor=white)](https://x.com/stingning/status/1927770654385860804) [![微信](https://img.shields.io/badge/WeChat--Group-07C160?style=for-the-badge&logo=wechat&logoColor=white)](figs/wechat-group.png)

</div>

<!-- <div align="center">
<p>
<a href="#news" style="text-decoration: none; font-weight: bold;">🎉 新闻</a> •
<a href="#overview" style="text-decoration: none; font-weight: bold;">📖 概述</a> •
<a href="#main-results" style="text-decoration: none; font-weight: bold;">📃 主要结果</a> •
<a href="#getting-started" style="text-decoration: none; font-weight: bold;">✨ 开始使用</a>
</p>
<p>
<a href="#acknowledgement" style="text-decoration: none; font-weight: bold;">✿ 感谢</a> •
<a href="#contact" style="text-decoration: none; font-weight: bold;">邮件</a> •
<a href="#todo" style="text-decoration: none; font-weight: bold;">📝 任务列表</a> •
<a href="#citation" style="text-decoration: none; font-weight: bold;">🎈 引用</a>
</p>
</div> -->
**简单VLA-RL**是一个高效的RL框架，适用于VLA，能够在数据稀缺的情况下改进长期规划。它利用强化学习，在模拟和现实任务中可以显著优于SFT，揭示出一种“推切”新动作现象，并增强了空间/对象/目标的泛化能力。
</div>

# 🎉新闻
- **[2026-01-01]** **基于{颜色{红色}当然，请提供您需要翻译的Markdown文本。{简单VLA-RL}}我们已经实施了{颜色{红色}{现实世界的强化学习}}在长期的灵巧任务中，我目睹了{颜色{红色}当然，请提供您需要翻译的Markdown文本。{非平凡的}}$$ (大约300%)性能改进超过SFT模型，同时具备令人惊讶的自恢复能力。博客即将上线。**



https://github.com/user-attachments/assets/45fca289-39d4-4a42-8014-1ef7eff2d806



- **[2025-10-01]** **简单VLA-RL**现在支持RoboTwin2.0基准测试。请随意尝试！
- **[2025-09-12]**期待发布**简单VLA-RL**论文！去看看吧：[论文](https://arxiv.org/abs/2509.09674).
- **[2025-05-27]**我们发布了代码。**简单VLA-RL**.

</div>

# 📌亮点

#### 高效有效的VLA强化学习框架
- 基于的端到端VLA RL管道[维拉尔](https://github.com/volcengine/verl)带有VLA特定优化
- 多环境并行渲染显著加速了VLA轨迹采样
- 利用[维拉尔](https://github.com/volcengine/verl)'我们的最先进的基础设施：高效的分布式训练'(FSDP)混合通信模式和优化的内存管理以实现快速训练/推理

#### 模型和环境支持
- **VLA 模型**: [开放虚拟实验室](https://github.com/openvla/openvla), [开放虚拟实验室-光学场技术](https://github.com/moojink/openvla-oft)
- **基准**: [自由](https://github.com/Lifelong-Robot-Learning/LIBERO), [RoboTwin 1.0/2.0](https://github.com/TianxingChen/RoboTwin)
- 模块化架构，便于新VLA模型、基准和RL算法的集成(即将到来)

#### 最小奖励工程和探索策略
- 二进制(0/1)结果奖励 - 不需要复杂的奖励设计
- 探索策略：动态采样，自适应剪裁，温度调整
</div>

# 🔧关键实现

SimplerVLA-RL扩展了veRL的组件，这些组件在以下模块中针对VLA进行了特定化：

**[`verl/trainer/main_ppo.py`](verl/trainer/main_ppo.py)**
- 主入口点与射线初始化
- `RobRewardManager`奖励分配

**[`verl/trainer/ppo/ray_trainer.py`](verl/trainer/ppo/ray_trainer.py)**
- 主要的RL训练循环：数据加载、VLA滚动、模型更新、评估和检查点化
- RL算法特定优势计算

**[`verl/workers/fsdp_workers.py`](verl/workers/fsdp_workers.py)**
- 核心功能调用的来源`ray_trainer.py`
- VLA 模型/优化器初始化，`generate_sequences`, `compute_entropy`, `update_actor`

**[`verl/workers/actor/dp_rob.py`](verl/workers/actor/dp_rob.py)**
- 函数的具体实现`fsdp_workers.py`
- RL损失计算，策略更新，`compute_log_prob`, `compute_entropy`

**[`verl/workers/rollout/rob_rollout.py`](verl/workers/rollout/rob_rollout.py)**
- VLA部署实施：环境创建、多环境并行渲染、VLA动作生成、环境交互、视频保存、轨迹和0/1奖励收集

**[`verl/utils/dataset/rob_dataset.py`](verl/utils/dataset/rob_dataset.py)**
- 数据集构建以训练/测试跨基准

**[`verl/utils/vla_utils/`](verl/utils/vla_utils/)**
- VLA 模型实现(OpenVLA-OFT/OpenVLA 从官方代码)
</div>

# ✨开始使用

#### 1. 设置环境

请看[设置.md](SETUP.md)为了详细了解如何设置conda环境的详细说明。

#### 2. 准备SFT模型

一个**SFT(监督微调)**VLA模型是RL训练所必需的。以下是可用的选项：

* **OpenVLA-OFT SFT 模型**
下载[简单VLA-RL集合](https://huggingface.co/collections/Haozhan72/simplevla-rl-6833311430cd9df52aeb1f86)可选模型包括：
  - `libero-10 traj1/trajall SFT`
  - `libero-goal/object/spatial traj1 SFT`
  - `Robotwin2.0 tasks traj1000 SFT`
* **OpenVLA SFT 模型**
下载[这里](https://huggingface.co/openvla).

* **其他模型**
对于其他模型，您可能需要自己进行微调。

#### 3. 使用SimpleVLA-RL进行训练

在运行训练脚本之前，请确保以下配置已正确设置：

- **设置权重和偏置(W&B)API密钥**
将替换`WANDB_API_KEY`字段`SimpleVLA-RL/align.json`使用您自己的W&B API密钥。

- **修改关键变量**
更新以下变量。`examples/run_openvla_oft_rl_libero/twin2.sh`根据需要：
  - `WANDB_API_KEY`: 你的W&B API密钥。
  - `EXPERIMENT_NAME`实验的名称。你可以选择任何名称。
  - `SFT_MODEL_PATH`模型的路径。
  - `CKPT_PATH`保存检查点的路径。
  - `DATASET_NAME`: 查看详细选项，参见`examples/run_openvla_oft_rl_libero/twin2.sh`.
  - `ALIGN_PATH`路径到`SimpleVLA-RL/align.json`文件。
  - `NUM_GPUS`：每个节点可用的GPU数量(好的，请提供您需要翻译的Markdown文本。`8`).
  - `NUM_NODES`：用于RL训练的节点数量(好的，请提供您需要翻译的Markdown文本。`1`).

> [注意]
> 
- 该脚本已在以下配置上进行测试：
- 单节点设置：`NUM_NODES=1`, `NUM_GPUS=8` (一台节点，配备8个NVIDIA A800 GPU，每个GPU有80GB内存).
- 多节点设置：`NUM_NODES=2`, `NUM_GPUS=8` (两个节点，每个节点配备16个NVIDIA A800 GPU，每个GPU有80GB内存).
- 使用的驱动版本是`470.161.03`并且CUDA版本是`12.4`. *(不需要)*

- **运行RL训练**
使用以下命令在LIBERO或RoboTwin2.0基准上开始OpenVLA-OFT的RL训练：
  
  ```bash
  bash examples/run_openvla_oft_rl_libero.sh
  or
  bash examples/run_openvla_oft_rl_twin2.sh
  ```
  

#### 4. 运行评估

为了评估你的模型性能，请通过设置来启用评估模式。`trainer.val_only=True`在`examples/run_openvla_oft_rl_libero/twin2.sh`然后，执行相同的脚本：

```bash
bash examples/run_openvla_oft_rl_libero.sh
or
bash examples/run_openvla_oft_rl_twin2.sh
```

# 📃 主要结果

我们使用OpenVLA-OFT在LIBERO上评估SimpleVLA-RL。SimpleVLA-RL将OpenVLA-OFT在LIBERO-Long上的性能提升至97.6分，并创造了新的行业标准。值得注意的是，仅使用一个轨迹作为冷启动SFT，SimpleVLA-RL将OpenVLA-OFT的性能从17.3提高到91.7，实现了74.4个百分点的改进。(430.1%).

<div align="center">
<img src="figs/main.png" alt="Main Results of SimpleVLA-RL." width="90%" />
</div>

<div align="center">
<img src="figs/teaser.png" alt="Overview of SimpleVLA-RL." width="90%" />
</div>

# ✿感谢

我们基于此代码的预览版本进行开发。[维拉尔](https://github.com/volcengine/verl), [开放虚拟实验室-光学场技术](https://github.com/moojink/openvla-oft), [RoboTwin2.0](https://github.com/RoboTwin-Platform/RoboTwin.git)并且[Prime](https://github.com/PRIME-RL/PRIME)我们感谢他们的重大贡献！
为了获取更多详情和更新，请参阅相应项目的官方文档和仓库。
</div>


</div>

# 路线图

### 扩展模型支持
- [ ]支持基于扩散的高级强化学习：[p0](https://github.com/Physical-Intelligence/openpi)和[p0.5](https://github.com/Physical-Intelligence/openpi)与流匹配RL
- [ ]支持更多VLA模型：特别是轻量级的(好的，请提供您需要翻译的Markdown文本。[VLA-适配器](https://github.com/OpenHelix-Team/VLA-Adapter), [小VLA](https://huggingface.co/docs/lerobot/smolvla))

### 扩展环境支持
- [ ]支持更多基准测试：例如[更简单的环境](https://github.com/simpler-env/SimplerEnv), [行为](https://github.com/behavior-robot-suite/brs-algo), [卡尔文](https://github.com/mees/calvin)
- [ ]支持现实世界的强化学习。

### 扩展框架
- [ ]额外的在线RL方法和离线RL算法
- [ ]模块化环境和VLA接口，便于适应
- [ ]进一步优化RL框架以实现更高效的训练
</div>

# 邮件

- 李浩然：zhan72426@gmail.com
- 宁丁：dingning@mail.tsinghua.edu.cn

# 🎈引用

如果你发现SimpleVLA-RL很有帮助，请引用我们。

```bibtex
@article{li2025simplevla,
  title={SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning},
  author={Li, Haozhan and Zuo, Yuxin and Yu, Jiale and Zhang, Yuhao and Yang, Zhaohui and Zhang, Kaiyan and Zhu, Xuekai and Zhang, Yuchen and Chen, Tianxing and Cui, Ganqu and others},
  journal={arXiv preprint arXiv:2509.09674},
  year={2025}
}
```

# 🌟星史

[![星历表](https://api.star-history.com/svg?repos=PRIME-RL/SimpleVLA-RL&type=Date)](https://www.star-history.com/#PRIME-RL/SimpleVLA-RL&Date)
