# Run DiffSpeech

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

First, you need a pre-trained FastSpeech2 checkpoint `chckpoints/fs2_exp/model_ckpt_steps_160000.ckpt`. To train a FastSpeech 2 model, run: 

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config egs/datasets/audio/lj/fs2_orig.yaml --exp_name fs2_exp --reset
```

Then, run:

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
