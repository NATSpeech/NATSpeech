import os

import torch

from modules.vocoder.hifigan.hifigan import HifiGanGenerator
from tasks.tts.dataset_utils import FastSpeechWordDataset
from tasks.tts.tts_utils import load_data_preprocessor
from utils.commons.ckpt_utils import load_ckpt
from utils.commons.hparams import set_hparams


class BaseTTSInfer:
    def __init__(self, hparams, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hparams = hparams
        self.device = device
        self.data_dir = hparams['binary_data_dir']
        self.preprocessor, self.preprocess_args = load_data_preprocessor()
        self.ph_encoder, self.word_encoder = self.preprocessor.load_dict(self.data_dir)
        self.spk_map = self.preprocessor.load_spk_map(self.data_dir)
        self.ds_cls = FastSpeechWordDataset
        self.model = self.build_model()
        self.model.eval()
        self.model.to(self.device)
        self.vocoder = self.build_vocoder()
        self.vocoder.eval()
        self.vocoder.to(self.device)

    def build_model(self):
        raise NotImplementedError

    def forward_model(self, inp):
        raise NotImplementedError

    def build_vocoder(self):
        base_dir = self.hparams['vocoder_ckpt']
        config_path = f'{base_dir}/config.yaml'
        config = set_hparams(config_path, global_hparams=False)
        vocoder = HifiGanGenerator(config)
        load_ckpt(vocoder, base_dir, 'model_gen')
        return vocoder

    def run_vocoder(self, c):
        c = c.transpose(2, 1)
        y = self.vocoder(c)[:, 0]
        return y

    def preprocess_input(self, inp):
        """

        :param inp: {'text': str, 'item_name': (str, optional), 'spk_name': (str, optional)}
        :return:
        """
        preprocessor, preprocess_args = self.preprocessor, self.preprocess_args
        text_raw = inp['text']
        item_name = inp.get('item_name', '<ITEM_NAME>')
        spk_name = inp.get('spk_name', '<SINGLE_SPK>')
        ph, txt, word, ph2word, ph_gb_word = preprocessor.txt_to_ph(
            preprocessor.txt_processor, text_raw, preprocess_args)
        word_token = self.word_encoder.encode(word)
        ph_token = self.ph_encoder.encode(ph)
        spk_id = self.spk_map[spk_name]
        item = {'item_name': item_name, 'text': txt, 'ph': ph, 'spk_id': spk_id,
                'ph_token': ph_token, 'word_token': word_token, 'ph2word': ph2word}
        item['ph_len'] = len(item['ph_token'])
        return item

    def input_to_batch(self, item):
        item_names = [item['item_name']]
        text = [item['text']]
        ph = [item['ph']]
        txt_tokens = torch.LongTensor(item['ph_token'])[None, :].to(self.device)
        txt_lengths = torch.LongTensor([txt_tokens.shape[1]]).to(self.device)
        word_tokens = torch.LongTensor(item['word_token'])[None, :].to(self.device)
        word_lengths = torch.LongTensor([txt_tokens.shape[1]]).to(self.device)
        ph2word = torch.LongTensor(item['ph2word'])[None, :].to(self.device)
        spk_ids = torch.LongTensor(item['spk_id'])[None, :].to(self.device)
        batch = {
            'item_name': item_names,
            'text': text,
            'ph': ph,
            'txt_tokens': txt_tokens,
            'txt_lengths': txt_lengths,
            'word_tokens': word_tokens,
            'word_lengths': word_lengths,
            'ph2word': ph2word,
            'spk_ids': spk_ids,
        }
        return batch

    def postprocess_output(self, output):
        return output

    def infer_once(self, inp):
        inp = self.preprocess_input(inp)
        output = self.forward_model(inp)
        output = self.postprocess_output(output)
        return output

    @classmethod
    def example_run(cls):
        from utils.commons.hparams import set_hparams
        from utils.commons.hparams import hparams as hp
        from utils.audio.io import save_wav

        set_hparams()
        inp = {
            'text': 'the invention of movable metal letters in the middle of the fifteenth century may justly be considered as the invention of the art of printing.'
        }
        infer_ins = cls(hp)
        out = infer_ins.infer_once(inp)
        os.makedirs('infer_out', exist_ok=True)
        save_wav(out, f'infer_out/example_out.wav', hp['audio_sample_rate'])
