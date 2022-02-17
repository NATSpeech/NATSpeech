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
[![](https://img.shields.io/github/downloads/NATSpeech/NATSpeech/total?label=pretrained+model+downloads)](https://github.com/NATSpeech/NATSpeech/releases/tag/pretrained_models) | [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue)](https://huggingface.co/NATSpeech) | [中文文档](./README-zh.md) 

</div>


This repo contains official PyTorch implementation of:

- [PortaSpeech: Portable and High-Quality Generative Text-to-Speech](https://proceedings.neurips.cc/paper/2021/file/748d6b6ed8e13f857ceaa6cfbdca14b8-Paper.pdf) (NeurIPS 2021)  
[Demo page](https://portaspeech.github.io/) | [HuggingFace🤗 Demo](https://huggingface.co/spaces/NATSpeech/PortaSpeech)
- [DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism](https://arxiv.org/abs/2105.02446) (DiffSpeech) (AAAI 2022)  
[Demo page](https://diffsinger.github.io/) | [Project page](https://github.com/MoonInTheRiver/DiffSinger) | [HuggingFace🤗 Demo](https://huggingface.co/spaces/NATSpeech/DiffSpeech)

## Key Features 
We implement the following features in this framework:

- Data processing for non-autoregressive Text-to-Speech
  using [Montreal Forced Aligner](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner).
- Convenient and scalable framework for training and inference.
- Simple but efficient random-access dataset implementation.

## Install Dependencies

```bash
## We tested on Linux/Ubuntu 18.04. 
## Install Python 3.6+ first (Anaconda recommended).

export PYTHONPATH=.
# build a virtual env (recommended).
python -m venv venv
source venv/bin/activate
# install requirements.
pip install -U pip
pip install Cython numpy==1.19.1
pip install torch==1.9.0 # torch >= 1.9.0 recommended
pip install -r requirements.txt
sudo apt install -y sox libsox-fmt-mp3
bash mfa_usr/install_mfa.sh # install forced alignment tool
```

## Documents

- [About the framework](./docs/framework.md)
- [Run PortaSpeech](./docs/portaspeech.md)
- [Run DiffSpeech](./docs/diffspeech.md)

## Citation

If you find this useful for your research, please cite the following papers:

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

## Acknowledgments

Our codes are influenced by the following repos:

- [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
- [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)
- [Hifi-GAN](https://github.com/jik876/hifi-gan)
- [espnet](https://github.com/espnet/espnet)
- [Glow-TTS](https://github.com/jaywalnut310/glow-tts)
- [DiffSpeech](https://github.com/MoonInTheRiver/DiffSinger)
