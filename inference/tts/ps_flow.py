import torch
from inference.tts.base_tts_infer import BaseTTSInfer
from modules.tts.portaspeech.portaspeech_flow import PortaSpeechFlow
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import hparams


class PortaSpeechFlowInfer(BaseTTSInfer):
    def build_model(self):
        ph_dict_size = len(self.ph_encoder)
        word_dict_size = len(self.word_encoder)
        model = PortaSpeechFlow(ph_dict_size, word_dict_size, self.hparams)
        load_ckpt(model, hparams['work_dir'], 'model')
        model.to(self.device)
        with torch.no_grad():
            model.store_inverse_all()
        model.eval()
        return model

    def forward_model(self, inp):
        sample = self.input_to_batch(inp)
        with torch.no_grad():
            output = self.model(
                sample['txt_tokens'],
                sample['word_tokens'],
                ph2word=sample['ph2word'],
                word_len=sample['word_lengths'].max(),
                infer=True,
                forward_post_glow=True,
                spk_id=sample.get('spk_ids')
            )
            mel_out = output['mel_out']
            wav_out = self.run_vocoder(mel_out)
        wav_out = wav_out.cpu().numpy()
        return wav_out[0]


if __name__ == '__main__':
    PortaSpeechFlowInfer.example_run()
