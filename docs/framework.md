# Framework of NATSpeech

NATSpeech is a simple framework for Non-Autoregressive Text-to-Speech.

## Directory Structure

- `egs`: configuration files, which will be loaded by `utils/commons/hparams.py`
- `data_gen`: data binarization codes
- `modules`: modules and models
- `tasks`: the training and inference logics
- `utils`: commonly used utils
- `data`: data
    - `raw`: raw data
    - `processed`: data after preprocess
    - `binary`: binary data
- `checkpoints`: model checkpoints, tensorboard logs and generated results for all experiments.

## How to Add New Tasks and Run?

We show the basic steps of adding a new task/model and running the code (LJSpeech dataset as an example).

### Add the model

Add your model to `modules`.

### Add the task

Task classes are used to manage the training and inference procedures.

A new task (e.g., `tasks.tts.fs.FastSpeechTask`) should inherit the base task (`tasks.tts.speech_base.SpeechBaseTask`)
class.

You must implement these methods:

- `build_tts_model`, which builds the model for your task. - `run_model`, indicating how to use the model in training
  and inference.

You can override `test_step` and `save_valid_result` to change the validation/testing logics or add more plots to
tensorboard.

### Add a new config file

Add a new config file in `egs/datasets/audio/lj/YOUR_TASK.yaml`. For example:

```yaml
base_config: ./base_text2mel.yaml
task_cls: tasks.tts.fs.FastSpeechTask

# model configs
hidden_size: 256
dropout: 0.1

# some more configs .....
```

If you use a new dataset `YOUR_DATASET`, you should also add a `YOUR_DATASET_Processor`
in `egs/datasets/audio/YOUR_DATASET/preprocess.py`, inheriting `data_gen.tts.base_preprocess.BasePreprocessor`, which
loads some meta information of the dataset.

### Preprocess and binary dataset

```bash
python data_gen/tts/runs/align_and_binarize.py --config egs/datasets/audio/lj/base_text2mel.yaml
```

### Training

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config YOUR_CONFIG --exp_name YOUR_EXP_NAME --reset
```

You can open Tensorboard via:

```bash
tensorboard --logdir checkpoints/EXP_NAME
```

### Inference (Testing)

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config egs/datasets/audio/lj/YOUR_TASK.yaml --exp_name YOUR_EXP_NAME --reset --infer
```

## Design Philosophy

### Random-Access Binarized Dataset

To address the IO problem when reading small files, we design a `IndexedDataset` class (_utils/commons/indexed_datasets.py_)

### Global Config

We introduce a global config `hparams`, which is load from a `.yaml` config file and can be used in anywhere. However,
we do not recommend using it in some general-purpose modules.

### BaseTrainer Framework

Our [base trainer](utils/commons/trainer.py) and [base task ](utils/commons/base_task.py) classes refer
to [PytorchLightning](https://github.com/PyTorchLightning/pytorch-lightning), which builds some commonly used
training/inference code structure. Our framework supports multi-process GPU training without changing the subclass
codes.

### Checkpoint Saving

All checkpoints and tensorboard logs are saved in `checkpoints/EXP_NAME`, where `EXP_NAME` is set in the running
command: `python tasks/run.py .... --exp_name EXP_NAME`. You can use `tensorboard --logdir checkpoints/EXP_NAME` to open
the tensorboard and check the training loss curves etc.
