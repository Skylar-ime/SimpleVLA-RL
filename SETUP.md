# 简易VLA-RL安装指南

这个指南提供了设置SimpleVLA-RL环境的步骤说明。我们的项目建立在[维拉尔](https://github.com/volcengine/verl)并且安装过程涉及三个主要组件：veRL、模拟基准和VLA模型。(开放虚拟实验室-光学场技术).

## 安装选项

### 在LIBERO基准测试上运行RL

#### 步骤1：安装veRL

> **注意：**我们建议使用veRL版本0.2或0.3。最新版本可能会有库冲突。

遵循官方[veRL安装指南](https://verl.readthedocs.io/en/v0.3.x/start/install.html):

```bash
# Create and activate conda environment
conda create -n simplevla python==3.10
conda activate simplevla

# Install PyTorch
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Clone veRL (recommended to place at the same level as simplevla-rl, not inside the simplevla-rl folder)
git clone -b v0.2.x https://github.com/volcengine/verl.git
cd verl
pip3 install -e .
cd ..
```

#### 步骤2：安装LIBERO和OpenVLA-OFT

遵循官方[安装指南](https://github.com/moojink/openvla-oft):

```bash
conda activate simplevla
pip3 install torch torchvision

# Clone OpenVLA-OFT (place at the same level as simplevla-rl, not inside the simplevla-rl folder)
git clone https://github.com/moojink/openvla-oft.git
cd openvla-oft
pip install -e .

# Install Flash Attention 2 for training
# If you encounter issues, try `pip cache remove flash_attn` first
pip install packaging ninja
ninja --version; echo $?  # Should return exit code "0"
pip3 install flash-attn --no-build-isolation

cd ..

# Install LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
pip install -e LIBERO
cd openvla-oft
pip install -r experiments/robot/libero/libero_requirements.txt
```

### 选项2：在RoboTwin 2.0基准上运行RL

#### 步骤1：安装veRL

与选项1的步骤1相同。

#### 步骤2：安装RoboTwin 2.0

遵循官方[RoboTwin 2.0 安装指南](https://robotwin-platform.github.io/doc/usage/robotwin-install.html#1-dependencies):

```bash
# Install system dependencies
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools

conda activate simplevla

# Clone and install RoboTwin
git clone https://github.com/RoboTwin-Platform/RoboTwin.git
cd RoboTwin
bash script/_install.sh

# Download RoboTwin assets
bash script/_download_assets.sh
cd ..
```

#### 步骤3：安装OpenVLA-OFT

```bash
conda activate simplevla
pip3 install torch torchvision

# Clone OpenVLA-OFT (place at the same level as simplevla-rl, not inside the simplevla-rl folder)
git clone https://github.com/moojink/openvla-oft.git
cd openvla-oft
pip install -e .

# Install Flash Attention 2
pip install packaging ninja
ninja --version; echo $?  # Should return exit code "0"
pip3 install flash-attn --no-build-isolation
cd ..
```

#### 步骤4：配置RoboTwin以支持SimpleVLA-RL

对RoboTwin进行必要的修改。

```bash
git clone https://github.com/PRIME-RL/SimpleVLA-RL.git
cd SimpleVLA-RL

# Apply RoboTwin modifications
bash copy_overwrite_robotwin2.sh <your_robotwin_path> <your_simplevlarl_path>
# Example: bash copy_overwrite_robotwin2.sh /mnt/petrelfs/SimpleVLA-RL /mnt/petrelfs/RoboTwin
```

## 故障排除

- 如果你遇到RoboTwin 2.0安装的问题，请参考[罗布双胞胎文档](https://robotwin-platform.github.io/doc/)或者查看他们的GitHub问题
- 所有仓库(维拉，开放虚拟实验室-光学场技术，机器人双胞胎)建议在与SimpleVLA-RL同级目录下克隆

## 目录结构

安装后，你的目录结构应该看起来像这样：
```
your_workspace/
├── SimpleVLA-RL/
├── verl/
├── openvla-oft/
├── LIBERO/          (for Option 1)
└── RoboTwin/        (for Option 2)
```


## 支持RoboTwin 2.0的附加任务(当然可以，请提供您希望翻译的Markdown文本。)

### 步骤1：收集可行的种子

RoboTwin 2.0的任务可能有不可行的种子(超出手臂范围的物体)为了优化RL训练，我们预先收集可行的种子以避免在训练周期中重复验证。

**收集过程：**

1. 更新`DATASET_NAME`在`pre_collect_robotwin2_seed.sh`以您的目标任务名称
2. 运行集合脚本：
   ```bash
   sh pre_collect_robotwin2_seed.sh
   ```
3. 这生成了`robotwin2_train_seeds.json`在SimpleVLA-RL目录中
4. 添加JSON内容到：
   ```
   SimpleVLA-RL/verl/utils/envs/robotwin2/seeds/robotwin2_train_seeds.json
   ```

### 步骤2：注册新任务

1. 添加任务名称`SimpleVLA-RL/verl/utils/dataset/rob_dataset.py`
2. 添加任务名称和对应的最大步骤`SimpleVLA-RL/verl/workers/rollout/rob_rollout.py`

### 步骤3：实现任务特定函数

添加一个`get_info()`在对应任务文件中的函数`SimpleVLA-RL/verl/utils/envs/robotwin2/envs/task_name.py`.

为了实现参考，请参见：
```
SimpleVLA-RL/modified_codes/robotwin2/envs/handover_block.py
```
