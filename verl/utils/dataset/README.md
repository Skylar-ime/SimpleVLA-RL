# 数据集格式
## RLHF数据集
我们将所有数据源合并到一个Parquet文件中。我们直接将提示组织成聊天格式，以便轻松纳入多轮对话。在提示中，我们可以添加指令跟随文本以指导模型按照特定格式输出答案，以便我们提取答案。

数学问题
```json
{
    "data_source": "openai/gsm8k",
    "prompt": [{"role": "user", "content": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let's think step by step and output the final answer after \"####\""}],
    "ability": "math",
    "reward_model": {
        "style": "rule",
        "ground_truth": ["72"]
    },
}
```
