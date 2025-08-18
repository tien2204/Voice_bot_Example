import sys
import os

sys.path.append(os.getcwd())

from f5_tts.model import CFM, DiT

import torch
import thop


""" ~155M """


""" ~335M """


transformer = DiT(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)


model = CFM(transformer=transformer)
target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
duration = 20
frame_length = int(duration * target_sample_rate / hop_length)
text_length = 150

flops, params = thop.profile(
    model,
    inputs=(
        torch.randn(1, frame_length, n_mel_channels),
        torch.zeros(1, text_length, dtype=torch.long),
    ),
)
print(f"FLOPs: {flops / 1e9} G")
print(f"Params: {params / 1e6} M")
