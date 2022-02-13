<p align="center">
    <br>
    <img src="assets/logo.png" width="200"/>
    <br>
</p>

<h2 align="center">
<p> NATSpeech: A Non-Autoregressive Text-to-Speech Framework</p>
</h2>

<div align="center">

[![](https://img.shields.io/github/stars/NATSpeech/NATSpeech)](https://github.com/NATSpeech/NATSpeech)
[![](https://img.shields.io/github/forks/NATSpeech/NATSpeech)](https://github.com/NATSpeech/NATSpeech)
[![](https://img.shields.io/github/license/NATSpeech/NATSpeech)](https://github.com/NATSpeech/NATSpeech/blob/main/LICENSE)
[![](https://img.shields.io/github/downloads/NATSpeech/NATSpeech/total?label=pretrained+model+downloads)](https://github.com/NATSpeech/NATSpeech/releases/tag/pretrained_models) | [English README](./README.md)

</div>


本仓库包含了以下工作的官方PyTorch实现：

- [PortaSpeech: Portable and High-Quality Generative Text-to-Speech](https://proceedings.neurips.cc/paper/2021/file/748d6b6ed8e13f857ceaa6cfbdca14b8-Paper.pdf) (NeurIPS 2021)  
[Demo页面](https://portaspeech.github.io/) | [HuggingFace🤗 Demo](https://huggingface.co/spaces/NATSpeech/PortaSpeech)
- [DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism](https://arxiv.org/abs/2105.02446) (DiffSpeech) (AAAI 2022)  
[Demo页面](https://diffsinger.github.io/) | [项目主页](https://github.com/MoonInTheRiver/DiffSinger) | [HuggingFace🤗 Demo](https://huggingface.co/spaces/NATSpeech/DiffSpeech)

## 主要特点 
我们在本框架中实现了以下特点：

- 基于[Montreal Forced Aligner](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner)的非自回归语音合成数据处理流程；
- 便于使用和可扩展的训练和测试框架；
- 简单但有效的随机访问数据集类的实现。

## 安装依赖

```bash
## 在 Linux/Ubuntu 18.04 上通过测试 
## 首先需要安装 Python 3.6+ (推荐使用Anaconda)

export PYTHONPATH=.
# 创建虚拟环境 (推荐).
python -m venv venv
source venv/bin/activate
# 安装依赖
pip install -U pip
pip install Cython numpy==1.19.1
pip install torch==1.9.0 # 推荐 torch >= 1.9.0
pip install -r requirements.txt
sudo apt install -y sox libsox-fmt-mp3
bash mfa_usr/install_mfa.sh # 安装强制对齐工具
```

## 文档

- [关于本框架](./docs/zh/framework.md)
- [运行PortaSpeech](./docs/portaspeech.md)
- [运行DiffSpeech](./docs/diffspeech.md)

## 引用

如果本REPO对你的研究和工作有用，请引用以下论文：

- PortaSpeech

```bib
@article{ren2021portaspeech,
  title={PortaSpeech: Portable and High-Quality Generative Text-to-Speech},
  author={Ren, Yi and Liu, Jinglin and Zhao, Zhou},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

- DiffSpeech

```bib
@article{liu2021diffsinger,
  title={Diffsinger: Singing voice synthesis via shallow diffusion mechanism},
  author={Liu, Jinglin and Li, Chengxi and Ren, Yi and Chen, Feiyang and Liu, Peng and Zhao, Zhou},
  journal={arXiv preprint arXiv:2105.02446},
  volume={2},
  year={2021}
 }
```

## 致谢

我们的代码受以下代码和仓库启发：

- [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
- [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)
- [Hifi-GAN](https://github.com/jik876/hifi-gan)
- [espnet](https://github.com/espnet/espnet)
- [Glow-TTS](https://github.com/jaywalnut310/glow-tts)
- [DiffSpeech](https://github.com/MoonInTheRiver/DiffSinger)
