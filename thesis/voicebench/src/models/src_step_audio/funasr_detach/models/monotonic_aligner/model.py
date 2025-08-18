




import time
import copy
import torch
from torch.cuda.amp import autocast
from typing import Union, Dict, List, Tuple, Optional

from src.models.src_step_audio.funasr_detach.register import tables
from src.models.src_step_audio.funasr_detach.models.ctc.ctc import CTC
from src.models.src_step_audio.funasr_detach.utils import postprocess_utils
from src.models.src_step_audio.funasr_detach.utils.datadir_writer import DatadirWriter
from src.models.src_step_audio.funasr_detach.models.paraformer.cif_predictor import mae_loss
from src.models.src_step_audio.funasr_detach.train_utils.device_funcs import force_gatherable
from src.models.src_step_audio.funasr_detach.models.transformer.utils.add_sos_eos import add_sos_eos
from src.models.src_step_audio.funasr_detach.models.transformer.utils.nets_utils import make_pad_mask
from src.models.src_step_audio.funasr_detach.utils.timestamp_tools import ts_prediction_lfr6_standard
from src.models.src_step_audio.funasr_detach.utils.load_utils import load_audio_text_image_video, extract_fbank


@tables.register("model_classes", "MonotonicAligner")
class MonotonicAligner(torch.nn.Module):
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Achieving timestamp prediction while recognizing with non-autoregressive end-to-end ASR model
    https://arxiv.org/abs/2301.12343
    """

    def __init__(
        self,
        input_size: int = 80,
        specaug: Optional[str] = None,
        specaug_conf: Optional[Dict] = None,
        normalize: str = None,
        normalize_conf: Optional[Dict] = None,
        encoder: str = None,
        encoder_conf: Optional[Dict] = None,
        predictor: str = None,
        predictor_conf: Optional[Dict] = None,
        predictor_bias: int = 0,
        length_normalized_loss: bool = False,
        **kwargs,
    ):
        super().__init__()

        if specaug is not None:
            specaug_class = tables.specaug_classes.get(specaug)
            specaug = specaug_class(**specaug_conf)
        if normalize is not None:
            normalize_class = tables.normalize_classes.get(normalize)
            normalize = normalize_class(**normalize_conf)
        encoder_class = tables.encoder_classes.get(encoder)
        encoder = encoder_class(input_size=input_size, **encoder_conf)
        encoder_output_size = encoder.output_size()
        predictor_class = tables.predictor_classes.get(predictor)
        predictor = predictor_class(**predictor_conf)
        self.specaug = specaug
        self.normalize = normalize
        self.encoder = encoder
        self.predictor = predictor
        self.criterion_pre = mae_loss(normalize_length=length_normalized_loss)
        self.predictor_bias = predictor_bias

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                text: (Batch, Length)
                text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]
        
        text = text[:, : text_lengths.max()]
        speech = speech[:, : speech_lengths.max()]

        
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        encoder_out_mask = (
            ~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]
        ).to(encoder_out.device)
        if self.predictor_bias == 1:
            _, text = add_sos_eos(text, 1, 2, -1)
            text_lengths = text_lengths + self.predictor_bias
        _, _, _, _, pre_token_length2 = self.predictor(
            encoder_out, text, encoder_out_mask, ignore_id=-1
        )

        
        loss_pre = self.criterion_pre(
            text_lengths.type_as(pre_token_length2), pre_token_length2
        )

        loss = loss_pre
        stats = dict()

        
        stats["loss_pre"] = loss_pre.detach().cpu() if loss_pre is not None else None
        stats["loss"] = torch.clone(loss.detach())

        
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def calc_predictor_timestamp(self, encoder_out, encoder_out_lens, token_num):
        encoder_out_mask = (
            ~make_pad_mask(encoder_out_lens, maxlen=encoder_out.size(1))[:, None, :]
        ).to(encoder_out.device)
        ds_alphas, ds_cif_peak, us_alphas, us_peaks = (
            self.predictor.get_upsample_timestamp(
                encoder_out, encoder_out_mask, token_num
            )
        )
        return ds_alphas, ds_cif_peak, us_alphas, us_peaks

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encoder. Note that this method is used by asr_inference.py
        Args:
                speech: (Batch, Length, ...)
                speech_lengths: (Batch, )
                ind: int
        """
        with autocast(False):

            
            if self.specaug is not None and self.training:
                speech, speech_lengths = self.specaug(speech, speech_lengths)

            
            if self.normalize is not None:
                speech, speech_lengths = self.normalize(speech, speech_lengths)

        
        encoder_out, encoder_out_lens, _ = self.encoder(speech, speech_lengths)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        return encoder_out, encoder_out_lens

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        meta_data = {}
        
        time1 = time.perf_counter()
        audio_list, text_token_int_list = load_audio_text_image_video(
            data_in,
            fs=frontend.fs,
            audio_fs=kwargs.get("fs", 16000),
            data_type=kwargs.get("data_type", "sound"),
            tokenizer=tokenizer,
        )
        time2 = time.perf_counter()
        meta_data["load_data"] = f"{time2 - time1:0.3f}"
        speech, speech_lengths = extract_fbank(
            audio_list, data_type=kwargs.get("data_type", "sound"), frontend=frontend
        )
        time3 = time.perf_counter()
        meta_data["extract_feat"] = f"{time3 - time2:0.3f}"
        meta_data["batch_data_time"] = (
            speech_lengths.sum().item() * frontend.frame_shift * frontend.lfr_n / 1000
        )

        speech = speech.to(device=kwargs["device"])
        speech_lengths = speech_lengths.to(device=kwargs["device"])

        
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        
        text_lengths = torch.tensor([len(i) + 1 for i in text_token_int_list]).to(
            encoder_out.device
        )
        _, _, us_alphas, us_peaks = self.calc_predictor_timestamp(
            encoder_out, encoder_out_lens, token_num=text_lengths
        )

        results = []
        ibest_writer = None
        if kwargs.get("output_dir") is not None:
            if not hasattr(self, "writer"):
                self.writer = DatadirWriter(kwargs.get("output_dir"))
            ibest_writer = self.writer["tp_res"]

        for i, (us_alpha, us_peak, token_int) in enumerate(
            zip(us_alphas, us_peaks, text_token_int_list)
        ):
            token = tokenizer.ids2tokens(token_int)
            timestamp_str, timestamp = ts_prediction_lfr6_standard(
                us_alpha[: encoder_out_lens[i] * 3],
                us_peak[: encoder_out_lens[i] * 3],
                copy.copy(token),
            )
            text_postprocessed, time_stamp_postprocessed, _ = (
                postprocess_utils.sentence_postprocess(token, timestamp)
            )
            result_i = {
                "key": key[i],
                "text": text_postprocessed,
                "timestamp": time_stamp_postprocessed,
            }
            results.append(result_i)

            if ibest_writer:
                
                ibest_writer["timestamp_list"][key[i]] = time_stamp_postprocessed
                ibest_writer["timestamp_str"][key[i]] = timestamp_str

        return results, meta_data
