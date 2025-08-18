




from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn




from src.models.src_step_audio.funasr_detach.frontends.abs_frontend import AbsFrontend



from src.models.src_step_audio.funasr_detach.train_utils.device_funcs import force_gatherable

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    
    @contextmanager
    def autocast(enabled=True):
        yield


class Data2VecPretrainModel(nn.Module):
    """Data2Vec Pretrain model"""

    def __init__(
        self,
        frontend=None,
        specaug=None,
        normalize=None,
        encoder=None,
        preencoder=None,
    ):

        super().__init__()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.encoder = encoder
        self.num_updates = 0

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Calc loss
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        
        assert speech.shape[0] == speech_lengths.shape[0], (
            speech.shape,
            speech_lengths.shape,
        )

        self.encoder.set_num_updates(self.num_updates)

        
        encoder_out = self.encode(speech, speech_lengths)

        losses = encoder_out["losses"]
        loss = sum(losses.values())
        sample_size = encoder_out["sample_size"]
        loss = loss.sum() / sample_size

        target_var = float(encoder_out["target_var"])
        pred_var = float(encoder_out["pred_var"])
        ema_decay = float(encoder_out["ema_decay"])

        stats = dict(
            loss=torch.clone(loss.detach()),
            target_var=target_var,
            pred_var=pred_var,
            ema_decay=ema_decay,
        )

        loss, stats, weight = force_gatherable((loss, stats, sample_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
    ):
        """Frontend + Encoder.
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        
        if min(speech_lengths) == max(
            speech_lengths
        ):  
            speech_lengths = None
        encoder_out = self.encoder(
            feats, speech_lengths, mask=True, features_only=False
        )

        return encoder_out

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            
            
            
            
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates

    def get_num_updates(self):
        return self.num_updates
