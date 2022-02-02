# Prepare Vocoder

We use [HiFi-GAN](https://github.com/jik876/hifi-gan) as the default vocoder.

## LJSpeech

### Use Pretrained Model

```bash
wget https://github.com/NATSpeech/NATSpeech/releases/download/pretrained_models/hifi_lj.zip
unzip hifi_lj.zip
mv hifi_lj checkpoints/hifi_lj
```

### Train Your Vocoder

#### Set Config Path and Experiment Name

```bash
export CONFIG_NAME=egs/datasets/audio/lj/hifigan.yaml  
export MY_EXP_NAME=my_hifigan_exp
```

#### Prepare Dataset

Prepare dataset following [prepare_data.md](./prepare_data.md). 

If you have run the `prepare_data` step of the acoustic
model (e.g., PortaSpeech and DiffSpeech), you only need to binarize the dataset for the vocoder training:

```bash
python data_gen/tts/runs/binarize.py --config $CONFIG_NAME
```

#### Training

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config $CONFIG_NAME --exp_name $MY_EXP_NAME --reset
```

#### Inference (Testing)

```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --config $PS_CONFIG --exp_name $MY_EXP_NAME --infer
```

#### Use the trained vocoder
Modify the `vocoder_ckpt` in config files of acoustic models (e.g., `egs/datasets/audio/lj/base_text2mel.yaml`) to $MY_EXP_NAME (e.g., `vocoder_ckpt: checkpoints/my_hifigan_exp`)

