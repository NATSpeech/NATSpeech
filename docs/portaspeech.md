# Run PortaSpeech

## Quick Start

### Install Dependencies

Install dependencies following [readme.md](../readme.md)

### Set Config Path and Experiment Name

#### PortaSpeech (normal)
```bash
export CONFIG_NAME=egs/datasets/audio/lj/ps_flow_nips2021.yaml  
export MY_EXP_NAME=ps_normal_exp
```

#### PortaSpeech (small)
```bash
export CONFIG_NAME=egs/datasets/audio/lj/ps_flow_small_nips2021.yaml
export MY_EXP_NAME=ps_small_exp
```

### Preprocess and binary dataset

Prepare dataset following [prepare_data.md](./prepare_data.md)

### Prepare Vocoder

Prepare vocoder following [prepare_vocoder.md](./prepare_vocoder.md)

## Training

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config $CONFIG_NAME --exp_name $MY_EXP_NAME --reset
```

You can check the training and validation curves open Tensorboard via:

```bash
tensorboard --logdir checkpoints/$MY_EXP_NAME
```

## Inference (Testing)

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config $PS_CONFIG --exp_name $MY_EXP_NAME --infer
```

## Citation

If you find this useful for your research, please use the following.

```
@article{ren2021portaspeech,
  title={PortaSpeech: Portable and High-Quality Generative Text-to-Speech},
  author={Ren, Yi and Liu, Jinglin and Zhao, Zhou},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```
