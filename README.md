<div align="center">
    <img alt="MM-Eureka logo" src="./docs/logo.png" style="height: 200px;" />
</div>


<div align="center">

# MM-EUREKA

</div>

<div align="center">
<p align="center">
  📖<a href="https://github.com/ModalMinds/MM-EUREKA/blob/main/MM_Eureka_paper.pdf">Paper</a> |
  📊<a href="https://huggingface.co/datasets/FanqingM/MM-Eureka-Dataset">Datasets</a> |
  🤗<a href="https://huggingface.co/FanqingM/MM-Eureka-8B">MM-Eureka-8B</a> |
  🤗<a href="https://huggingface.co/FanqingM/MM-Eureka-Zero-38B">MM-Eureka-Zero-38B</a>
</p>
</div>

<hr>
<div align="center">
<p style="text-align: center;">MM-EUREKA: Exploring Visual Aha Moment with Rule-based Large-scale Reinforcement Learning<p>
</div>
<hr>
<div align="center">
<a href="https://github.com/ModalMinds/MM-EUREKA/blob/main/MM_Eureka_paper.pdf">[[Paper PDF Link]]</a>
</div>

<div align="center">
    <img alt="Visual Aha Moment" src="./docs/visual_aha_moment.png"/>
</div>


## 🎯Overview

We present **MM-Eureka** and **MM-Eureka-Zero**, a series of multimodal reasoning models that successfully extend large-scale rule-based reinforcement learning (RL) to multimodal reasoning. 

While rule-based RL has shown remarkable success in improving LLMs' reasoning abilities in text domains, its application to multimodal settings has remained challenging. Our work reproduces key characteristics of text-based RL systems like DeepSeek-R1 in the multimodal space for the first time, including steady increases in **accuracy reward** and **response length**, and the emergence of **reflection behaviors**. 

We demonstrate that both instruction-tuned and pre-trained models can develop strong multimodal reasoning capabilities through rule-based RL without supervised fine-tuning, showing superior **data efficiency** compared to alternative approaches. 

🔥We open-source our complete pipeline to foster further research in this area. We release all our codes, models, data, etc. at [MM-EUREKA](https://github.com/ModalMinds/MM-EUREKA)

## 🗞️ News

- **[2025/03/07]** We released `MM-Eureka`.
  - 📖 Paper: [MM-Eureka-paper](https://github.com/ModalMinds/MM-EUREKA/blob/main/MM_Eureka_paper.pdf) 
  - 🤗 Model: [MM-Eureka-8B](https://huggingface.co/FanqingM/MM-Eureka-8B) & [MM-Eureka-Zero-38B](https://huggingface.co/FanqingM/MM-Eureka-Zero-38B)
  - 📊 Dataset: [MM-Eureka-Dataset](https://huggingface.co/datasets/FanqingM/MM-Eureka-Dataset)



## 🚀 Features

This repository is built upon [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), introducing several key enhancements:

- **Multimodal RFT Support**: Extends OpenRLHF to incorporate **vision-language models (VLMs)**, currently supporting **InternVL**, enabling multimodal reasoning capabilities.
  - Currently support **RLOO**, **REINFORCE++**, **GRPO** training using Ray.
  - vLLM integration and distributed training.
  - Support hybrid engine (`--colocate_all_models`, `--vllm_enable_sleep`).
- **Better Rule-based Reward support**: Better training visualization for Rule-based Rewards (i.g. Format Reward, Accuracy Reward, Repetition Penalty)
- **Online Filtering**: Filtering out experiences based on Accuracy Reward during training as in [PRIME](https://github.com/PRIME-RL/PRIME)
  - Use `--enable_accuracy_filter`, `--freezing_filter_steps`, `--accuracy_lower_bound`, `--accuracy_upper_bound` to control the behavior of online accuracy filter.
  - Online Accuracy filter is not currently enabled in our default settings, refer to the Disccusion Section in our [paper](https://github.com/ModalMinds/MM-EUREKA/blob/main/MM_Eureka_paper.pdf) for more details.


## 🤖 Models

<div align="center">
    <img alt="Training Log" src="./docs/training_log.png" width="800"/>
</div>
*Figure 1 | Train Time Scale-up on Accuracy Reward and Response Length of Rule-Based RL. (a) represents the training scenario on InternVL2.5-instruct-8B, while (b) corresponds to the training scenario on InternVL2.5-pretrained-38B. It can be observed that stable improvements in accuracy reward and response length can be achieved regardless of whether the model is based on an instruct model or a pretrained model.*

- 🤗 [MM-Eureka-8B](https://huggingface.co/FanqingM/MM-Eureka-8B)
  
- 🤗 [MM-Eureka-Zero-38B](https://huggingface.co/FanqingM/MM-Eureka-Zero-38B)


## 🏁 Getting Started

### 📦 Installation

```shell
git clone https://github.com/ModalMinds/MM-EUREKA.git
cd MM-EUREKA
pip install -e .[vllm_latest]

# install flash-attn==2.3.6:

pip install flash-attn==2.3.6 --no-build-isolation

# Alternatively you can compile from source:

git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout v2.3.6
python setup.py install
```

### 📂 Data Preparation

You can download our training data from [MM-Eureka-Dataset](https://huggingface.co/datasets/FanqingM/MM-Eureka-Dataset)

Once downloaded, refer to the section below for additional data formation. You may need to update the `image_urls` field to reference your local image paths for proper processing.

#### Custom dataset

For custom dataset, format your data in to a JSONL file,  where each entry is a dictionary organized in the following format. 

```json
{
  "id": "0",
  "conversations": [
      {
          "role": "system",
          "content": "system_prompt"
      },
      {
          "role": "user",
          "content": "user_prompt"
      }
  ],
  "answer": "gt that could be parsed and verified by math_verify",
  "image_urls": ["file:///path/to/image1", "file:///path/to/image2"]
}
```

> [!NOTE]
> For text-only inputs, we follow InternVL's official approach, which requires a dummy image input. 
> Specifically, you should provide a (224, 224) pure white image as a placeholder.
> We have already provided such a blank image at: `examples/blank.png`

### 🌐 Start Training

Before starting your own training, ensure that the paths in the provided training scripts are correctly set and that environment variables like `$MASTER_ADDR` and `$NODE_RANK` are properly configured.

**start MM-Eureka-8B training**

- for single node

  ```shell
  sh examples/scripts/train_mm_eureka_8b_single_node.sh
  ```

- for multiple node

  ```shell
  sh examples/scripts/train_mm_eureka_8b_multi_node.sh
  ```

**start MM-Eureka-Zero-38B training**

```shell
sh examples/scripts/train_mm_eureka_zero_38b_multi_node.sh
```

**Qwen Training Log**

<div align="center">
    <img alt="Qwen training log" src="docs/qwen_log.png" style="height: 200px;"/>
</div>

Although we use the InternVL2.5-8B-Instruct filtered dataset, we can still observe a steady increase in accuracy reward and response length when training Qwen2.5-VL-7B

## ⭐ Starchart

[![Star History Chart](https://api.star-history.com/svg?repos=ModalMinds/MM-EUREKA&type=Date)](https://star-history.com/#ModalMinds/MM-EUREKA&Date)

## 🤝 Contribution

MM-Eureka is stil under active development, if you want to contribute, please feel free to make a pull request or create an issue.

Please refer to `CONTRIBUTING.md` before you dive in！

## 📬 Contact

If you have any questions or would like to engage with our community, feel free to scan the QR code below to join our WeChat group.

<div align="center">
<img alt="MM-Eureka logo" src="https://github.com/user-attachments/assets/a04ebfef-9ac4-44ae-a07b-48586794903a" style="height: 400px;" />
</div>

## 🎓 Acknowledgements

We acknowledge the outstanding open-source contributions from [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), [LMM-R1](https://github.com/TideDra/lmm-r1) and [vLLM](https://github.com/vllm-project/vllm). We also extend our gratitude to [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) and [InternVL](https://github.com/OpenGVLab/InternVL) for their open-source techniques and base models, which have enabled us to further our exploration.

## 📜 Citation
```
@article{meng2025mm,
  title={MM-Eureka: Exploring Visual Aha Moment with Rule-based Large-scale Reinforcement Learning},
  author={Meng, Fanqing and Du, Lingxiao and Liu, Zongkai and Zhou, Zhixiang and Lu, Quanfeng and Fu, Daocheng and Shi, Botian and Wang, Wenhai and He, Junjun and Zhang, Kaipeng and others},
  journal={arXiv preprint arXiv:2503.07365},
  year={2025}
}
```
