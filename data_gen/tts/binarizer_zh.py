import numpy as np
from data_gen.tts.base_binarizer import BaseBinarizer


class ZhBinarizer(BaseBinarizer):
    @staticmethod
    def process_align(tg_fn, item):
        BaseBinarizer.process_align(tg_fn, item)
        # char-level pitch
        if 'f0' in item:
            ph_list = item['ph'].split(" ")
            item['f0_ph'] = np.array([0 for _ in item['f0']], dtype=float)
            char_start_idx = 0
            f0s_char = []
            for idx, (f0_, ph_idx) in enumerate(zip(item['f0'], item['mel2ph'])):
                is_pinyin = ph_list[ph_idx - 1][0].isalpha()
                if not is_pinyin or ph_idx - item['mel2ph'][idx - 1] > 1:
                    if len(f0s_char) > 0:
                        item['f0_ph'][char_start_idx:idx] = sum(f0s_char) / len(f0s_char)
                    f0s_char = []
                    char_start_idx = idx
                    if not is_pinyin:
                        char_start_idx += 1
                if f0_ > 0:
                    f0s_char.append(f0_)
