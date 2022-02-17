import importlib
import re

import gradio as gr
import yaml
from gradio.inputs import Textbox

from inference.tts.base_tts_infer import BaseTTSInfer
from utils.commons.hparams import set_hparams
from utils.commons.hparams import hparams as hp
import numpy as np

from utils.text.text_encoder import PUNCS


class GradioInfer:
    def __init__(self, exp_name, inference_cls, title, description, article, example_inputs):
        self.exp_name = exp_name
        self.title = title
        self.description = description
        self.article = article
        self.example_inputs = example_inputs
        pkg = ".".join(inference_cls.split(".")[:-1])
        cls_name = inference_cls.split(".")[-1]
        self.inference_cls = getattr(importlib.import_module(pkg), cls_name)

    def greet(self, text):
        sents = re.split(rf'([{PUNCS}])', text.replace('\n', ','))
        if sents[-1] not in list(PUNCS):
            sents = sents + ['.']
        audio_outs = []
        s = ""
        for i in range(0, len(sents), 2):
            if len(sents[i]) > 0:
                s += sents[i] + sents[i + 1]
            if len(s) >= 400 or (i >= len(sents) - 2 and len(s) > 0):
                audio_out = self.infer_ins.infer_once({
                    'text': s
                })
                audio_out = audio_out * 32767
                audio_out = audio_out.astype(np.int16)
                audio_outs.append(audio_out)
                audio_outs.append(np.zeros(int(hp['audio_sample_rate'] * 0.3)).astype(np.int16))
                s = ""
        audio_outs = np.concatenate(audio_outs)
        return hp['audio_sample_rate'], audio_outs

    def run(self):
        set_hparams(exp_name=self.exp_name)
        infer_cls = self.inference_cls
        self.infer_ins: BaseTTSInfer = infer_cls(hp)
        example_inputs = self.example_inputs
        iface = gr.Interface(fn=self.greet,
                             inputs=Textbox(
                                 lines=10, placeholder=None, default=example_inputs[0], label="input text"),
                             outputs="audio",
                             allow_flagging="never",
                             title=self.title,
                             description=self.description,
                             article=self.article,
                             examples=example_inputs,
                             enable_queue=True)
        iface.launch(share=True,cache_examples=True)


if __name__ == '__main__':
    gradio_config = yaml.safe_load(open('inference/tts/gradio/gradio_settings.yaml'))
    g = GradioInfer(**gradio_config)
    g.run()
