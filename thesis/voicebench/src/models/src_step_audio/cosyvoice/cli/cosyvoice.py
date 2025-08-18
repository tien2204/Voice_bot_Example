












import os
import uuid
import time
from tqdm import tqdm
import torch
import torchaudio
from hyperpyyaml import load_hyperpyyaml
from cosyvoice.cli.frontend import CosyVoiceFrontEnd
from cosyvoice.cli.model import CosyVoiceModel


class CosyVoice:

    def __init__(
        self,
        model_dir,
    ):
        self.model_dir = model_dir
        with open("{}/cosyvoice.yaml".format(model_dir), "r") as f:
            configs = load_hyperpyyaml(f)
        self.frontend = CosyVoiceFrontEnd(
            configs["feat_extractor"],
            "{}/campplus.onnx".format(model_dir),
            "{}/speech_tokenizer_v1.onnx".format(model_dir),
        )
        self.model = CosyVoiceModel(configs["flow"], configs["hift"])
        self.model.load(
            "{}/flow.pt".format(model_dir),
            "{}/hift.pt".format(model_dir),
        )
        self.model.flow = self.model.flow.to(torch.bfloat16)
        del configs

    def token_to_wav_offline(
        self,
        speech_token,
        speech_feat,
        speech_feat_len,
        prompt_token,
        prompt_token_len,
        embedding,
    ):
        tts_mel = self.model.flow.inference(
            token=speech_token.to(self.model.device),
            token_len=torch.tensor([speech_token.size(1)], dtype=torch.int32).to(
                self.model.device
            ),
            prompt_token=prompt_token.to(self.model.device),
            prompt_token_len=prompt_token_len.to(self.model.device),
            prompt_feat=speech_feat.to(self.model.device),
            prompt_feat_len=speech_feat_len.to(self.model.device),
            embedding=embedding.to(self.model.device),
        )
        tts_speech = self.model.hift.inference(mel=tts_mel.float())[0].cpu()
        return tts_speech
