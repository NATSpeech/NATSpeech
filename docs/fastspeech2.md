# Run FastSpeech 2

## Quick Start

### Install Dependencies

Install dependencies following [readme.md](../readme.md)

### Set Config Path and Experiment Name

```bash
export CONFIG_NAME=egs/datasets/audio/lj/fs2_orig.yaml
export MY_EXP_NAME=fs2_exp
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
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config $CONFIG_NAME --exp_name $MY_EXP_NAME --infer
```

## Citation

If you find this useful for your research, please use the following.

```
@inproceedings{ren2020fastspeech,
  title={FastSpeech 2: Fast and High-Quality End-to-End Text to Speech},
  author={Ren, Yi and Hu, Chenxu and Tan, Xu and Qin, Tao and Zhao, Sheng and Zhao, Zhou and Liu, Tie-Yan},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
```
