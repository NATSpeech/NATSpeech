import torch
from inference.tts.base_tts_infer import BaseTTSInfer
from modules.tts.fs import FastSpeech
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams


class FastSpeechInfer(BaseTTSInfer):
    def build_model(self):
        dict_size = len(self.ph_encoder)
        model = FastSpeech(dict_size, self.hparams)
        model.eval()
        load_ckpt(model, hparams['work_dir'], 'model')
        return model

    def forward_model(self, inp):
        sample = self.input_to_batch(inp)
        txt_tokens = sample['txt_tokens']  # [B, T_t]
        spk_id = sample.get('spk_ids')
        with torch.no_grad():
            output = self.model(txt_tokens, spk_id=spk_id, infer=True)
            mel_out = output['mel_out']
            wav_out = self.run_vocoder(mel_out)
        wav_out = wav_out.cpu().numpy()
        return wav_out[0]


if __name__ == '__main__':
    FastSpeechInfer.example_run()
