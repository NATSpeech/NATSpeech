# Prepare Dataset

## LJSpeech 

### Download Dataset
```bash
mkdir -p data/raw/ljspeech
cd data/raw
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
bzip2 -d LJSpeech-1.1.tar.bz2
tar -xvf LJSpeech-1.1.tar
cd ../../
```

### Forced Align and Preprocess Dataset
```bash
# Preprocess step: text and unify the file structure.
python data_gen/tts/runs/preprocess.py --config $CONFIG_NAME
# Align step: MFA alignment.
python data_gen/tts/runs/train_mfa_align.py --config $CONFIG_NAME
# Binarization step: Binarize data for fast IO. You only need to rerun this line when running different task if you have `preprocess`ed and `align`ed the dataset before.
python data_gen/tts/runs/binarize.py --config $CONFIG_NAME
```

## More datasets will be supported soon...