import json

import torch
import torch.nn as nn

from src.models.src_freeze_omni.decoder.ticodec.models import Encoder
from src.models.src_freeze_omni.decoder.ticodec.models import Generator
from src.models.src_freeze_omni.decoder.ticodec.models import Quantizer

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class VQVAE(nn.Module):
    def __init__(self,
                 config_path,
                 ckpt_path,
                 with_encoder=False):
        super(VQVAE, self).__init__()
        ckpt = torch.load(ckpt_path)
        with open(config_path) as f:
            data = f.read()
        json_config = json.loads(data)
        self.h = AttrDict(json_config)
        
        
        self.quantizer = Quantizer(self.h)
        self.generator = Generator(self.h)
        self.generator.load_state_dict(ckpt['generator'])
        self.quantizer.load_state_dict(ckpt['quantizer'])
        
        if with_encoder:
            self.encoder = Encoder(self.h)
            self.encoder.load_state_dict(ckpt['encoder'])

    def forward(self, x, global_style_token):
        
        
        quant_emb = self.quantizer.embed(x)
        global_style_quantized_emb = self.quantizer.embed_gst(global_style_token).squeeze(-1)
        return self.generator(quant_emb, global_style_quantized_emb)

    def encode(self, x):
        batch_size = x.size(0)
        if len(x.shape) == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        

        c, global_features = self.encoder(x.unsqueeze(1))
        
        
        q, loss_q, local_token, g, global_style_token = self.quantizer(c, global_features)
        local_token = [code.reshape(batch_size, -1) for code in local_token]
        global_style_token = torch.stack(global_style_token, -1).unsqueeze(1)
        
        return torch.stack(local_token, -1), global_style_token
