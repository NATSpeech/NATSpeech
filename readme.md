# NATSpeech

A boilerplate for Non-Autoregressive Text-to-Speech. We implement the following features in this boilerplate:

- Data processing for non-autoregressive Text-to-Speech
  using [Montreal Forced Aligner](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner).
- Convenient and scalable framework for training and inference.
- Simple but efficient random-access dataset implementation.

This repo also contains official PyTorch implementation for:

- [PortaSpeech: Portable and High-Quality Generative Text-to-Speech](https://proceedings.neurips.cc/paper/2021/file/748d6b6ed8e13f857ceaa6cfbdca14b8-Paper.pdf) (
  NeurIPS 2021) | [Demo page](https://portaspeech.github.io/)
- [FastSpeech 2: Fast and High-Quality End-to-End Text to Speech](https://openreview.net/pdf?id=piLPYqxtWuA) (ICLR 2020)
  | [Demo page](https://speechresearch.github.io/fastspeech2/)

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
- [Run FastSpeech 2](./docs/fastspeech2.md)

## Citation

If you find this useful for your research, please use the following.

- FastSpeech 2

```
@inproceedings{ren2020fastspeech,
  title={FastSpeech 2: Fast and High-Quality End-to-End Text to Speech},
  author={Ren, Yi and Hu, Chenxu and Tan, Xu and Qin, Tao and Zhao, Sheng and Zhao, Zhou and Liu, Tie-Yan},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
```

- PortaSpeech

```
@article{ren2021portaspeech,
  title={PortaSpeech: Portable and High-Quality Generative Text-to-Speech},
  author={Ren, Yi and Liu, Jinglin and Zhao, Zhou},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
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
