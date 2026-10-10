# CUDA 环境级修复记录

修复日期：2026-09-25。范围仅限 `sdpo-vla-pilot`。

## 原因与结果

当前内核加载 NVIDIA 580.173.02，而系统用户态库及磁盘上的 DKMS 为 580.178.04。无密码 sudo 不可用；没有进行系统安装、重启或模块卸载。

从已配置的 NVIDIA 官方 APT 源下载与当前内核相同版本的 libnvidia-compute 包，SHA256 与仓库元数据一致；仅用`dpkg-deb -x`解压到 conda 环境。包脚本未执行。

通过 conda activate/deactivate 事件，为该环境设置和恢复 LD_库_路径：添加环境内的 nvidia-smi 包装器。在加载模块版本不匹配时拒绝继续，优先使用旧驱动库；系统升级/重启后应重新检查 CUDA，不保证此临时修复永远有效。

## 使用

```bash
conda activate sdpo-vla-pilot
nvidia-smi
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

或：

```bash
conda run -n sdpo-vla-pilot nvidia-smi
```

仅使用Python环境的绝对路径，但未激活该环境，不会自动应用此修复。

## 验证边界

`verification.json`记录 NVIDIA-smi、PyTorch 设备发现、BF16 矩阵计算及反向传播。环境退出后的 LD_图书馆_恢复路径已被验证。

该项目现有四项 CPU 测试通过；继承的 ROS launch_测试插件缺少 Lark，因此仅在此次测试中设置了。`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`。没有更改 ROS。四项测试不能证明训练代码完整、实验已完成或模型显存一定够用。

本机 nvidia-smi 测试显示总显存为 49140 MiB；PyTorch 可用总量约为 47.36 GiB。不再根据 RTX 4090 的名称假设 24 GB；硬件来源或改装情况未进行验证。

## 撤销

先 `conda deactivate`，再移走或删除该环境中的以下三个新增文件：

- `/home/skylar/miniconda3/envs/sdpo-vla-pilot/etc/conda/activate.d/sdpo-pilot-nvidia.sh`
- `/home/skylar/miniconda3/envs/sdpo-vla-pilot/etc/conda/deactivate.d/sdpo-pilot-nvidia.sh`
- `/home/skylar/miniconda3/envs/sdpo-vla-pilot/bin/nvidia-smi`

已解压的 `opt/nvidia-driver-580.173.02`不被引用时不会影响运行，可以保留或手动移除。不要删除 conda 环境，也不要删除系统驱动库。

## 系统层面未完成

系统 NVIDIA 包版本混乱及 unpacked 状态未在本次修复中解决。后续系统级维护需要管理员授权、审查 APT 的完整变更计划并协调桌面会话；此前简单的一条安装命令加重启不能保证解决所有包冲突。当前实验不再需要等待系统重启即可使用 CUDA。
