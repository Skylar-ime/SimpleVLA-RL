# 模型
常见的模型库，如huggingface/transformers，在使用Pytorch原生的模型并行时会遇到困难。遵循vLLM的设计原则，我们保持一个简单、可并行化、高度优化且打包输入的verl。
## 添加新Huggingface模型
### 步骤1：将模型文件从HF复制到verl
- 在verl/models/hf下添加一个新文件
- 将模型文件从huggingface/transformers/models复制到verl/models/hf

### 步骤2：修改模型文件以使用打包输入
- 移除所有与推理相关的代码(缓存)
- 修改输入以仅包含
- 输入_id 系列(总计_网络，)
- 请_序列长度(总计_n个数加1，)
- 最大_序列长度_在_批次：整数
- 注意，这需要使用闪存注意力与因果掩码。

### 步骤2.5：添加测试
- 添加一个测试来比较这个版本和HuggingFace的版本
- 按照基础设施并添加测试到tests/models/hf

### 步骤3：添加一个应用张量并行的函数
- 请遵循
- https://pytorch.org/docs/stable/distributed.tensor.parallel.html
- https://pytorch.org/tutorials/intermediate/TP_tutorial.html
- 一般评论
- 在原生 PyTorch 中，张量并行化不是自动并行化。它的工作方式是通过配置来指定模型参数和输入输出的重塑方法。这些配置被注册为钩子，在模型前向或后向执行时进行输入输出的重塑操作。

### 步骤4：添加一个应用数据并行的函数
- 请使用FSDP2 API
- 请参见此处的演示。https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py#L413

### 步骤5：添加一个应用管道并行处理的函数
- 在PyTorch 2.4中
- 目前仅在夜间版本的alpha中可用
- 请查看torchtitan以获取更多详细信息。

