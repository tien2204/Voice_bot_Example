




import logging

import torch

from src.models.src_step_audio.funasr_detach.models.transformer.model import Transformer
from src.models.src_step_audio.funasr_detach.register import tables


@tables.register("model_classes", "SANM")
class SANM(Transformer):
    """
    Author: Zhifu Gao, Shiliang Zhang, Ming Lei, Ian McLoughlin
    San-m: Memory equipped self-attention for end-to-end speech recognition
    https://arxiv.org/abs/2006.01713
    """

    def __init__(
        self,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
