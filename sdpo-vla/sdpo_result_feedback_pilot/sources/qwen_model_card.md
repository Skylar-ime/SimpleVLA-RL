
---
许可证_名称：qwen研究
许可证_链接：https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE
语言：
- 好的，请提供您需要翻译的Markdown文本。
管道_标签：图像文本转文本
标签：
- 多模态
库_名称：transformers
---

# Qwen2.5-VL-3B-指令
<a href="https://chat.qwenlm.ai/" target="_blank" style="margin: 2px;">
<img alt="Chat" src="https://img.shields.io/badge/%F0%9F%92%9C%EF%B8%8F%20Qwen%20Chat%20-536af5" style="display: inline-block; vertical-align: middle;"/>
</a>

## 介绍

在Qwen2-VL发布后的五个月里，许多开发者基于Qwen2-VL的视觉语言模型构建了新的模型，为我们提供了宝贵的反馈。在此期间，我们专注于构建更多实用的视觉语言模型。今天，我们非常高兴地介绍Qwen家族的最新成员：Qwen2.5-VL。

#### 关键增强：
* **理解事物的视觉**Qwen2.5-VL 不仅擅长识别常见的物体如花朵、鸟类、鱼类和昆虫，而且在分析文本、图表、图标、图形和图像布局方面也表现出色。

* **作为行动者**Qwen2.5-VL 直接作为视觉代理，能够推理和动态地指导工具，具备计算机使用和手机使用的能力。

* **理解长视频和捕捉事件**Qwen2.5-VL 可以理解超过1小时的视频，这次它有了一个新的能力，可以通过精确定位相关视频片段来捕捉事件。

* **具有不同格式的视觉定位能力**Qwen2.5-VL 可以通过生成边界框或点来准确地在图像中定位物体，并且可以提供稳定的数据输出，包括坐标和属性。

* **生成结构化输出**: 用于扫描发票、表格等数据。Qwen2.5-VL 支持其内容的结构化输出，这使我们在金融、商业等领域中受益。


#### 模型架构更新：

* **动态分辨率和帧率训练用于视频理解**:

我们通过采用动态帧率采样来扩展动态分辨率到时间维度，使模型能够理解不同采样率的视频。因此，我们在时间维度上更新mRoPE，使用ID和绝对时间对齐，使模型能够学习时间序列和速度，并最终获得定位特定时刻的能力。

<p align="center">
<img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-VL/qwen2.5vl_arc.jpeg" width="80%"/>
<p>


* **简洁高效的视觉编码器**

我们通过在ViT中战略性地实施窗口注意力来同时提升训练和推理速度。进一步优化ViT架构，使用SwiGLU和RMSNorm，使其与Qwen2.5 LLM的结构相匹配。


我们有三个模型，参数分别为30亿、70亿和720亿。这个仓库包含经过指令调优的3B Qwen2.5-VL模型。更多信息请访问我们的[博客](https://qwenlm.github.io/blog/qwen2.5-vl/)和[GitHub](https://github.com/QwenLM/Qwen2.5-VL).



## 评估

### 图像基准

|基准|InternVL2.5-4B|Qwen2-VL-7B|Qwen2.5-VL-3B|
| :--- | :---:  | :---: | :---: |
|MMMU<sub>val</sub>  | 52.3 | 54.1 | 53.1|
|MMMU-Pro<sub>val</sub>  | **32.7** | 30.5 | 31.6|
|AI2D<sub>测试</sub> | 81.4 | **83.0** | 81.5 |
|文档问答<sub>测试</sub>  | 91.6 | 94.5 | **93.9** |
|信息视觉问答<sub>测试</sub>  | 72.1 | 76.5 | **77.1** |
|文本视觉问答<sub>val</sub>  | 76.8 | **84.3** | 79.3|
|MMBench-V1.1<sub>测试</sub>  | 79.3 | **80.7** | 77.6 |
|MMStar| 58.3 | **60.7** | 55.9 |
|数学视界<sub>测试迷你</sub>  | 60.5 | 58.2 | **62.3** |
|数学视觉<sub>全部</sub>  | 20.9 | 16.3  | **21.2** |


### 视频基准
|基准|InternVL2.5-4B|Qwen2-VL-7B|Qwen2.5-VL-3B|
| :--- | :---:  | :---: | :---: |
|MVBench| 71.6 | 67.0 | 67.0 |
|视频MME| 63.6/62.3 | 69.0/63.3 | 67.6/61.5 |
|MLVU| 48.3 | - | 68.2 |
|LVBench| - | - | 43.3 |
|MMBench-视频| 1.73 | 1.44 | 1.63 |
|自我结构| - | - | 64.8 |
|感知测试| - | - | 66.9 |
|温度指南| - | - | 64.4 |
|长视频评测| 55.2 | 55.6 | 54.2 |
|CharadesSTA/mIoU| - | - | 38.8 |


### 代理基准
|基准|Qwen2.5-VL-3B|
|-------------------------|---------------|
|屏幕焦点|     55.5    |
|屏幕焦点专业|     23.9    |
|AI技术_EM|  	76.9    |
|Android 控制高_EM|    	63.7    |
|Android 控制低_EM|  	22.2    |
|Android世界_SR| 	90.8  	|
|移动迷你摇摆++_SR| 	67.9    |

## 需求
Qwen2.5-VL的代码已经在最新的Hugging Face Transformers中，我们建议您使用以下命令从源代码构建：
```
pip install git+https://github.com/huggingface/transformers accelerate
```
或者你可能会遇到以下错误：
```
KeyError: 'qwen2_5_vl'
```


## 快速入门

以下，我们提供简单的示例来展示如何使用Qwen2.5-VL与ModelScope和Transformers一起使用。

Qwen2.5-VL的代码已经在最新的Hugging Face Transformers中，我们建议您使用以下命令从源代码构建：
```
pip install git+https://github.com/huggingface/transformers accelerate
```
或者你可能会遇到以下错误：
```
KeyError: 'qwen2_5_vl'
```


我们提供一个工具包，帮助您更方便地处理各种类型的视觉输入，就像使用API一样。这包括Base64、URL和交错的图片和视频。您可以使用以下命令安装它：

```bash
# It's highly recommanded to use `[decord]` feature for faster video loading.
pip install qwen-vl-utils[decord]==0.0.8
```

如果你不是使用Linux，你可能无法安装`decord`从PyPI获取。如果是这样，你可以使用`pip install qwen-vl-utils`这将退回到使用torchvision进行视频处理。但是，你仍然可以[安装decord源码](https://github.com/dmlc/decord?tab=readme-ov-file#install-from-source)为了在加载视频时使用Decord。

### 使用 🤗  Transformer 与聊天

这里我们展示一个代码片段，向您展示如何使用聊天模型。`transformers`和`qwen_vl_utils`:

```python
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-3B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```
<details>
<summary>多图推理</summary>

```python
# Messages containing multiple images and a text query
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "file:///path/to/image1.jpg"},
            {"type": "image", "image": "file:///path/to/image2.jpg"},
            {"type": "text", "text": "Identify the similarities between these images."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```
</details>

<details>
<summary>视频推断</summary>

```python
# Messages containing a images list as a video and a text query
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": [
                    "file:///path/to/frame1.jpg",
                    "file:///path/to/frame2.jpg",
                    "file:///path/to/frame3.jpg",
                    "file:///path/to/frame4.jpg",
                ],
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]

# Messages containing a local video path and a text query
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "file:///path/to/video1.mp4",
                "max_pixels": 360 * 420,
                "fps": 1.0,
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]

# Messages containing a video url and a text query
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]

#In Qwen 2.5 VL, frame rate information is also input into the model to align with absolute time.
# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    fps=fps,
    padding=True,
    return_tensors="pt",
    **video_kwargs,
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

视频URL兼容性很大程度上取决于第三方库的版本。详情见下表。更改后端`FORCE_QWENVL_VIDEO_READER=torchvision`或者`FORCE_QWENVL_VIDEO_READER=decord`如果你不介意使用默认的。

|后端|HTTP|HTTPS|
|-------------|------|-------|
|torchvision >= 0.19.0| ✅  | ✅   |
|torchvision < 0.19.0| ❌  | ❌   |
|decord| ✅  | ❌   |
</details>

<details>
<summary>批量推理</summary>

```python
# Sample messages for batch inference
messages1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "file:///path/to/image1.jpg"},
            {"type": "image", "image": "file:///path/to/image2.jpg"},
            {"type": "text", "text": "What are the common elements in these pictures?"},
        ],
    }
]
messages2 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who are you?"},
]
# Combine messages for batch processing
messages = [messages1, messages2]

# Preparation for batch inference
texts = [
    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    for msg in messages
]
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=texts,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Batch Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_texts = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_texts)
```
</details>

### 🤖 模型空间
我们强烈建议用户，尤其是中国大陆的用户，使用ModelScope。`snapshot_download`可以帮助您解决下载检查点的问题。


### 更多使用提示

对于输入的图片，我们支持本地文件、Base64和URL。对于视频，目前仅支持本地文件。

```python
# You can directly insert a local file path, a URL, or a base64-encoded image into the position where you want in the text.
## Local file path
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "file:///path/to/your/image.jpg"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
## Image URL
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "http://path/to/your/image.jpg"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
## Base64 encoded image
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "data:image;base64,/9j/..."},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
```
#### 图像分辨率以提升性能

该模型支持广泛的分辨率输入。默认情况下，它使用原生分辨率作为输入，但更高分辨率可以以牺牲更多计算为代价来提高性能。用户可以设置最小和最大像素数以实现最佳配置，例如将令牌计数范围设置在256-1280之间，以平衡速度和内存使用。

```python
min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels
)
```

此外，我们提供两种方法来对模型输入的图像大小进行精细控制：

1. 定义最小值_像素和最大_像素：图像将被缩放以保持其在[min范围内]的方面比。_像素和最大_像素。
   
2. 指定确切尺寸：直接设置`resized_height`和`resized_width`这些值将被四舍五入到最接近的28倍数。

```python
# min_pixels and max_pixels
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "file:///path/to/your/image.jpg",
                "resized_height": 280,
                "resized_width": 420,
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
# resized_height and resized_width
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "file:///path/to/your/image.jpg",
                "min_pixels": 50176,
                "max_pixels": 50176,
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
```

### 处理长文本

当前`config.json`长度设置为最多32,768个令牌。
为了处理超过32,768个令牌的大量输入，我们使用[雅人](https://arxiv.org/abs/2309.00071)，一种用于增强模型长文本外推的技术，确保在长文本上具有最佳性能。

对于支持的框架，您可以添加以下内容。`config.json`为了启用YaRN：

```
{
	...,
    "type": "yarn",
    "mrope_section": [
        16,
        24,
        24
    ],
    "factor": 4,
    "original_max_position_embeddings": 32768
}
```

然而，需要注意的是，这种方法对时空定位任务的性能有显著影响，因此不建议使用。

同时，对于长视频输入，由于MRoPE本身更经济地使用ID，因此最大_位置_嵌入值可以直接修改为更大的值，例如64K。



## 引用

如果你觉得我们的工作有帮助，请随时给我们引用。

```
@misc{qwen2.5-VL,
    title = {Qwen2.5-VL},
    url = {https://qwenlm.github.io/blog/qwen2.5-vl/},
    author = {Qwen Team},
    month = {January},
    year = {2025}
}

@article{Qwen2VL,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Wang, Peng and Bai, Shuai and Tan, Sinan and Wang, Shijie and Fan, Zhihao and Bai, Jinze and Chen, Keqin and Liu, Xuejing and Wang, Jialin and Ge, Wenbin and Fan, Yang and Dang, Kai and Du, Mengfei and Ren, Xuancheng and Men, Rui and Liu, Dayiheng and Zhou, Chang and Zhou, Jingren and Lin, Junyang},
  journal={arXiv preprint arXiv:2409.12191},
  year={2024}
}

@article{Qwen-VL,
  title={Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond},
  author={Bai, Jinze and Bai, Shuai and Yang, Shusheng and Wang, Shijie and Tan, Sinan and Wang, Peng and Lin, Junyang and Zhou, Chang and Zhou, Jingren},
  journal={arXiv preprint arXiv:2308.12966},
  year={2023}
}
```
