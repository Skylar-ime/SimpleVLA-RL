# 评估员与dlimp说明（2026-09-15）

## Evaluator 是什么

RoboTwin 的评估器不是独立的 pip 包，而是环境仓库中的评测程序：

```text
RoboTwin/script/eval_policy.py        # 标准评测，默认 100 个 test episodes
RoboTwin/script/eval_policy_short.py  # 短评测，支持 ROBOTWIN_EVAL_TEST_NUM
```

它负责：读取任务配置、初始化仿真环境、执行专家稳定性检查、调用策略的`get_model`/`eval`/`reset_model`，统计成功率并保存 `_result.txt` 与视频。

Simplified Chinese: 简单VLA-RL的`trainer.val_only=True` 不调用这两个脚本；它使用 `verl/workers/rollout/rob_rollout.py`自己的环境循环。两条链路都可评测，但输入适配和 checkpoint 加载不同，结果不能直接混称。

## 评估器依赖

服务器历史 RoboTwin 有依赖声明：`script/requirements.txt`，核心包括：

```text
torch==2.4.1, torchvision, sapien==3.0.0b1, mplib==0.2.1,
gymnasium==0.29.1, scipy==1.10.1, trimesh, open3d, zarr,
moviepy, imageio, av, matplotlib, ffmpeg, wandb
```

RoboTwin 安装脚本还会安装 PyTorch3D、Curobo，并修改 SAPIEN/mplib 的兼容代码；资产需另跑`bash script/_download_assets.sh`。官方的 SimpleVLA 安装说明见：
https://github.com/PRIME-RL/SimpleVLA-RL/blob/main/SETUP.md

## Dlimp 是什么

`dlimp`（数据加载器/轨迹包装器）是 OpenVLA-OFT 的 RLDS/TFRecord 数据加载库：以`tf.data.Dataset`底层提供轨迹/帧变换；还包括用于OpenVLA训练的图像处理逻辑。它不是RoboTwin仿真器，也不提供奖励/评估器。

官方 Fork:
https://github.com/moojink/dlimp_openvla

OpenVLA-OFT 的`pyproject.toml` 明确依赖：`dlimp @ git+https://github.com/moojink/dlimp_openvla`，同时要求 `tensorflow==2.15.0`、`tensorflow_datasets==4.9.3`、`tensorflow_graphics==2021.12.3`。来源：
https://github.com/moojink/openvla-oft/blob/main/pyproject.toml

## 服务器当前状态

`simplevla`环境中的 TensorFlow 2.15.0 可以导入；但是`pip show openvla-oft`显示其可编辑路径`/public/home/s2250018/wsy/openvla-oft` 已不存在，且 `import dlimp`当前失败。属于残留元数据，不是完整安装。

## 获取 dlimp（隔离环境）

在 H800 节点执行：

```bash
source /public/home/s2250018/miniconda3/etc/profile.d/conda.sh
conda activate simplevla
export PYTHONNOUSERSITE=1

git clone --depth 1 https://github.com/moojink/dlimp_openvla \
  /public/home/s2250018/wsy/VLA/dlimp_openvla
python -m pip install -e /public/home/s2250018/wsy/VLA/dlimp_openvla

python -c 'import dlimp; print(dlimp.__file__)'
```

如果计算节点没有外网，先在登录节点克隆，再从共享路径安装。也可以直接：

```bash
python -m pip install \
  'dlimp @ git+https://github.com/moojink/dlimp_openvla'
```

## 对本次提升_pot 的必要性

- RoboTwin 评估器本身：不需要 dlimp。
- 简单VLA-RL滚动：主要直接使用TensorFlow图像处理；当前代码未直接`import dlimp`。
- OpenVLA-OFT 完整安装/训练数据链路：建议安装 dlimp，避免导入或相关路径的 dataset 缺包。

因此，dlimp 是“OpenVLA-OFT 数据链路依赖”，而不是当前 0% 成绩的直接证据。首先修复 stale editable 安装，然后再进行离线 replay；不要为了验证结果而盲目重新安装整个 live 环境。
