# DiffSpeech

[HuggingFace🤗 Demo](https://huggingface.co/spaces/NATSpeech/DiffSpeech)

## Quick Start

### Install Dependencies

Install dependencies following [readme.md](../readme.md)

### Set Config Path and Experiment Name

```bash
export CONFIG_NAME=egs/datasets/audio/lj/ds.yaml
export MY_EXP_NAME=ds_exp
```

### Preprocess and binary dataset

Prepare dataset following [prepare_data.md](./prepare_data.md)

### Prepare Vocoder

Prepare vocoder following [prepare_vocoder.md](./prepare_vocoder.md)

## Training

First, you need a pre-trained FastSpeech2 checkpoint in `checkpoints/aux_exp`. You can use the [pre-trained model](https://github.com/NATSpeech/NATSpeech/releases/download/pretrained_models/aux_exp.zip), or train FastSpeech2 from scratch, run:

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config egs/datasets/audio/lj/fs2_orig.yaml --exp_name aux_exp --reset
```

Then, to train DiffSpeech, run:

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config $CONFIG_NAME --exp_name $MY_EXP_NAME --reset
```

You can check the training and validation curves open Tensorboard via:

```bash
tensorboard --logdir checkpoints/$MY_EXP_NAME
```

## Inference (Testing)

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config $CONFIG_NAME --exp_name $MY_EXP_NAME --infer
```

## Pre-trained Model

Download checkpoints from https://github.com/NATSpeech/NATSpeech/releases/download/pretrained_models/ds_exp.zip and unzip it to `checkpoints/ds_exp`. Then you can directly run inference command:

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name ds_exp --infer
```

## Citation

If you find this useful for your research, please use the following.

```bib
@article{liu2021diffsinger,
  title={Diffsinger: Singing voice synthesis via shallow diffusion mechanism},
  author={Liu, Jinglin and Li, Chengxi and Ren, Yi and Chen, Feiyang and Liu, Peng and Zhao, Zhou},
  journal={arXiv preprint arXiv:2105.02446},
  volume={2},
  year={2021}
 }
```
