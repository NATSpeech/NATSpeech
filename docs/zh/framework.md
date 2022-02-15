# NATSpeech架构细节

NATSpeech是一个简单的非自回归语音合成框架。

## 目录结构

- `egs`: 配置文件，被`utils/commons/hparams.py`读取
- `data_gen`: 数据二进制化代码，便于随机访问，优化IO速度
- `modules`: 模块和模型
- `tasks`: 训练和推理逻辑
- `utils`: 常用工具代码
- `data`: 数据
    - `raw`: 原始数据
    - `processed`: 预处理后的数据
    - `binary`: 二进制化后的数据
- `checkpoints`: 模型的checkpoints，tensorboard日志和生成的结果

## 如何添加新的任务并运行？

我们展示几个添加模型和任务并运行的基本步骤，以LJSpeech数据集为例。

### 添加模型

把模型添加到`modules`。

### 添加任务

任务类用于管理训练和推理的流程。 一个新任务（例如：`tasks.tts.fs.FastSpeechTask`）应继承任务基类（`tasks.tts.speech_base.SpeechBaseTask`）。

你必须实现以下方法：

- `build_tts_model`，用于创建你的模型
- `run_model`，用于描述训练和推理的逻辑

如果你需要修改测试的逻辑，你可以重写`test_step`和`save_valid_result`方法。

### 添加一个新的配置文件

添加一个新的yaml配置文件到`egs/datasets/audio/lj/YOUR_TASK.yaml`。例如：

```yaml
base_config: ./base_text2mel.yaml
task_cls: tasks.tts.fs.FastSpeechTask

# model configs
hidden_size: 256
dropout: 0.1

# some more configs .....
```

如果你使用一个新的数据集`YOUR_DATASET`，你应该在`egs/datasets/audio/YOUR_DATASET/preprocess.py`中添加一个数据预处理类`YOUR_DATASET_Processor`，继承`data_gen.tts.base_preprocess.BasePreprocessor`类。该类用于读取一些数据集的元信息。

### 预处理和二进制化数据集

```bash
python data_gen/tts/runs/align_and_binarize.py --config egs/datasets/audio/lj/base_text2mel.yaml
```

### 训练

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config YOUR_CONFIG --exp_name YOUR_EXP_NAME --reset
```

你可以打开Tensorboard:

```bash
tensorboard --logdir checkpoints/EXP_NAME
```

### 推理和测试

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config egs/datasets/audio/lj/YOUR_TASK.yaml --exp_name YOUR_EXP_NAME --reset --infer
```

## 设计哲学

### 随机读取的二进制化数据集

为了缓解读取大量小文件时的IO问题（应对网络IO等情况），我们设计了一个`IndexedDataset`类(_utils/commons/indexed_datasets.py_)，支持快速随机读取（random access）。

### 全局配置

我们引入了一个全局配置机制`hparams`，可以从`.yaml`配置文件中读取并可被用于任何代码位置。

然而，我们不推荐将它用于一些具有通用使用目的的模块中。

### Trainer

我们的[base trainer](utils/commons/trainer.py)和[base task](utils/commons/base_task.py)类参考了[PytorchLightning](https://github.com/PyTorchLightning/pytorch-lightning) 中的一些实现，它完成了一些常用的训练和推理过程。我们的框架也支持多卡训练和推理，而不需要修改任何子类代码。

### Checkpoint保存机制

所有的checkpoints和tensorboard日志都被保存在`checkpoints/EXP_NAME`，其中`EXP_NAME`是在训练时被指定的：`python tasks/run.py .... --exp_name EXP_NAME`. 你可以使用`tensorboard --logdir checkpoints/EXP_NAME`来打开tensorboard并查看训练、验证曲线以及一些中间生成的结果。
