from inference.tts.fs import FastSpeechInfer
from modules.tts.fs2_orig import FastSpeech2Orig
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams


class FastSpeech2OrigInfer(FastSpeechInfer):
    def build_model(self):
        dict_size = len(self.ph_encoder)
        model = FastSpeech2Orig(dict_size, self.hparams)
        model.eval()
        load_ckpt(model, hparams['work_dir'], 'model')
        return model


if __name__ == '__main__':
    FastSpeech2OrigInfer.example_run()
