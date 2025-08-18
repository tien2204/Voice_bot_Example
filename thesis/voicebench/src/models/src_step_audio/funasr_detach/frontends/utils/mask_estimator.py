from typing import Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch_complex.tensor import ComplexTensor

from src.models.src_step_audio.funasr_detach.models.transformer.utils.nets_utils import make_pad_mask
from src.models.src_step_audio.funasr_detach.models.language_model.rnn.encoders import RNN
from src.models.src_step_audio.funasr_detach.models.language_model.rnn.encoders import RNNP


class MaskEstimator(torch.nn.Module):
    def __init__(self, type, idim, layers, units, projs, dropout, nmask=1):
        super().__init__()
        subsample = np.ones(layers + 1, dtype=np.int32)

        typ = type.lstrip("vgg").rstrip("p")
        if type[-1] == "p":
            self.brnn = RNNP(idim, layers, units, projs, subsample, dropout, typ=typ)
        else:
            self.brnn = RNN(idim, layers, units, projs, dropout, typ=typ)

        self.type = type
        self.nmask = nmask
        self.linears = torch.nn.ModuleList(
            [torch.nn.Linear(projs, idim) for _ in range(nmask)]
        )

    def forward(
        self, xs: ComplexTensor, ilens: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.LongTensor]:
        """The forward function

        Args:
            xs: (B, F, C, T)
            ilens: (B,)
        Returns:
            hs (torch.Tensor): The hidden vector (B, F, C, T)
            masks: A tuple of the masks. (B, F, C, T)
            ilens: (B,)
        """
        assert xs.size(0) == ilens.size(0), (xs.size(0), ilens.size(0))
        _, _, C, input_length = xs.size()
        
        xs = xs.permute(0, 2, 3, 1)

        
        xs = (xs.real**2 + xs.imag**2) ** 0.5
        
        xs = xs.contiguous().view(-1, xs.size(-2), xs.size(-1))
        
        ilens_ = ilens[:, None].expand(-1, C).contiguous().view(-1)

        
        xs, _, _ = self.brnn(xs, ilens_)
        
        xs = xs.view(-1, C, xs.size(-2), xs.size(-1))

        masks = []
        for linear in self.linears:
            
            mask = linear(xs)

            mask = torch.sigmoid(mask)
            
            mask.masked_fill(make_pad_mask(ilens, mask, length_dim=2), 0)

            
            mask = mask.permute(0, 3, 1, 2)

            
            if mask.size(-1) < input_length:
                mask = F.pad(mask, [0, input_length - mask.size(-1)], value=0)
            masks.append(mask)

        return tuple(masks), ilens
