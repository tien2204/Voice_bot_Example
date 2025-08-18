from .wrapper import F5TTSWrapper
import livekit.agents
from livekit.agents import tts, utils, APIConnectionError
import livekit
from dataclasses import dataclass
import asyncio
from .. import logger
import wave
import typing as tp
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
import io
import numpy as np
from contextlib import contextmanager
from underthesea import sent_tokenize
import atexit
import multiprocessing as mp

tts_model: tp.Optional[F5TTSWrapper] = None


@dataclass
class _Conf:
    model_name: str
    ckpt_path: str
    vocab_file: str
    vocoder_name: str = "vocos"
    use_ema: bool = False
    target_sample_rate: int = 24000
    use_duration_predictor: bool = False


@dataclass
class GenerateConf:
    nfe_step: int = 32
    cfg_strength: float = 2.0
    speed: float = 1.0
    cross_fade_duration: float = 0.15
    sway_sampling_coef: float = -1.0


@dataclass
class ReferenceInput:
    ref_id: str
    ref_audio_path: str
    ref_text: str = ""


import torch


@dataclass
class ReferenceItem:
    cached_mel: torch.Tensor
    cached_mel_len: int
    cached_text: str


def _create_wave_header(sample_rate, num_channels=1, bits_per_sample=16, data_size=0):
    """Create a wave header for streaming. data_size=0 means unknown size."""

    if data_size > 0:

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(num_channels)
            wf.setsampwidth(bits_per_sample // 8)
            wf.setframerate(sample_rate)
            wf.setnframes(data_size // (num_channels * (bits_per_sample // 8)))
            wf.writeframes(b"")
        header_bytes = buffer.getvalue()
    else:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(num_channels)
            wf.setsampwidth(bits_per_sample // 8)
            wf.setframerate(sample_rate)

            wf.writeframes(b"")

        header_bytes = buffer.getvalue()

    return header_bytes


import re

preprocess_text_REPEATED_DOTS = re.compile(r"\.+", flags=re.MULTILINE)
preprocess_text_SPECIAL_CHARACTERS = re.compile(
    r"[^\w\s\.\,\?\!\d]+", flags=re.MULTILINE | re.UNICODE
)


def _preprocess_input_text(text: str):
    if not isinstance(text, str) or not text:
        return ""
    text = text.strip()
    text = preprocess_text_SPECIAL_CHARACTERS.sub("", text)
    return text


def _process_chunk(
    chunk_text: str,
    model: F5TTSWrapper,
    nfe_step: int = 32,
    cfg_strength: float = 2.0,
    speed: float = 1.0,
    cross_fade_duration: float = 0.15,
    sway_sampling_coef: float = -1.0,
) -> tp.Optional[bytes]:
    """Process a single text chunk and return raw audio bytes (int16)"""
    if not isinstance(chunk_text, str) or not chunk_text:
        return None
    chunk_text = _preprocess_input_text(chunk_text)

    logger.debug(f"Synthesizing chunk: '{chunk_text}'")

    try:

        audio_array, sample_rate = model.generate(
            text=chunk_text,
            return_numpy=True,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            speed=speed,
            cross_fade_duration=cross_fade_duration,
            sway_sampling_coef=sway_sampling_coef,
            use_duration_predictor=model.use_duration_predictor,
        )

        if audio_array is None or audio_array.size == 0:
            logger.warning(f"Model generated empty audio for chunk: '{chunk_text}'")
            return None

        audio_int16 = (audio_array * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        return audio_bytes
    except Exception as e:
        logger.warning(f"Error generating audio for chunk '{chunk_text}': {e}")
        return None


@contextmanager
def log_time(title: str = ""):
    global logger
    if not logger:
        import logging

        logger = logging.getLogger(__name__)
    if not isinstance(title, str):
        title = ""
    start = time.time()

    yield

    stop = time.time()
    logger.debug(f"{title} Duration: " + str(stop - start))


async def _stream_audio_generator(
    text: str,
    model: F5TTSWrapper,
    nfe_step: int = 32,
    cfg_strength: float = 2.0,
    speed: float = 1.0,
    cross_fade_duration: float = 0.15,
    sway_sampling_coef: float = -1.0,
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=True,
        separators=[r"[\n\.\?\!\;\,]+", re.escape(" "), re.escape("")],
        keep_separator=True,
    ),
) -> tp.AsyncGenerator[bytes, None]:
    with log_time("[stream_audio_generator]"):

        if model is None:
            logger.warning("Error: F5TTS model is not initialized.")
            raise Exception("TTS model is not ready. Please try again later.")

        try:
            if model.ref_audio_processed.device != model.device:
                logger.debug(
                    f"Cached mel tensor device ({model.ref_audio_processed.device}) differs from model device ({model.device}). Moving tensor."
                )
                model.ref_audio_processed = model.ref_audio_processed.to(model.device)

            logger.debug(
                f"[stream_audio_generator] Model reference state set from cache."
            )
            logger.debug(
                f"  Cached Mel Shape: {model.ref_audio_processed.shape}, Len: {model.ref_audio_len}"
            )

        except Exception as e:
            raise

        if not text or not text.strip():
            logger.warning("Error: No text provided in the request.")
            raise Exception("Input text cannot be empty.")

        logger.debug(
            f"[stream_audio_generator] Normalizing input text: '{text[:100]}...'"
        )
        try:
            normalized_text = _preprocess_input_text(text)
            logger.warning(
                f"[stream_audio_generator] Normalized text (first 100 chars): '{normalized_text[:100]}...'"
            )
        except Exception as e:
            logger.warning(
                f"Text normalization failed: {e}. Proceeding with original text."
            )
            normalized_text = text.strip()

        text_chunks = text_splitter.split_text(normalized_text)
        num_chunks = len(text_chunks)
        logger.debug(f"[stream_audio_generator] Text split into {num_chunks} chunks.")

        if num_chunks == 0:
            logger.warning("Text resulted in zero chunks after splitting.")
            yield _create_wave_header(model.target_sample_rate)
            return
        sample_rate = model.target_sample_rate
        logger.debug(
            f"[stream_audio_generator] Starting audio stream generation at {sample_rate} Hz..."
        )
        yield _create_wave_header(sample_rate=sample_rate, data_size=0)
        for chunk_num, chunk_text in enumerate(text_chunks):
            chunk_start_time = time.time()
            logger.debug(
                f"[stream_audio_generator] Processing chunk {chunk_num}/{num_chunks}..."
            )
            audio_bytes = _process_chunk(
                chunk_text,
                model,
                nfe_step=nfe_step,
                cfg_strength=cfg_strength,
                speed=speed,
                cross_fade_duration=cross_fade_duration,
                sway_sampling_coef=sway_sampling_coef,
            )
            if audio_bytes and len(audio_bytes) > 0:
                try:
                    yield audio_bytes
                    bytes_yielded = len(audio_bytes)
                    chunk_duration = time.time() - chunk_start_time
                    logger.debug(
                        f"  [Chunk {chunk_num}] Yielded {bytes_yielded} bytes. Time: {chunk_duration:.3f}s"
                    )
                except Exception as yield_e:
                    logger.debug(
                        f"Error yielding audio bytes for chunk {chunk_num}: {yield_e}"
                    )
                    break
            else:
                logger.debug(
                    f"[Chunk {chunk_num}] Skipped yielding (no audio data generated or error in process_chunk)."
                )


import threading

T = tp.TypeVar("T")


class RoundrobinWorkerPool(tp.Generic[T]):
    def __init__(self, num_workers: int, factory_func: tp.Callable[[], T]):
        self._num_workers = num_workers
        self.factory_func = factory_func
        self.workers = [factory_func() for _ in range(num_workers)]
        self.cur_id = 0
        self._lock = threading.RLock()
        atexit.register(self.close)

    @property
    def num_workers(self):
        with self._lock:
            return self._num_workers

    @num_workers.setter
    def num_workers(self, value):
        with self._lock:
            if value < self.num_workers:
                for item in self.workers[value:]:
                    del item
                self.workers = self.workers[:value]
            else:
                self.workers.extend(
                    self.factory_func() for _ in range(self.num_workers, value)
                )
            assert len(self.workers) == value, "Code ngu"
            self._num_workers = value

    def get(self) -> T:
        self.cur_id = (self.cur_id + 1) % self.num_workers
        return self.workers[self.cur_id]

    def close(self):
        for worker in self.workers:
            try:
                del worker
            except Exception as e:
                logger.warning(str(e))


import os
from cached_path import cached_path


class TTS(tts.TTS):
    def __init__(
        self,
        model_name: str = "F5TTS_v1_Pruned_14",
        ckpt_path: str = str(cached_path(f"hf://Darejkal/kakaka/model.safetensors")),
        vocab_file: str = str(cached_path(f"hf://Darejkal/kakaka/vocab.txt")),
        ref_id: str = "woman",
        ref_audio_path: str = os.path.join(os.path.dirname(__file__), "ref.wav"),
        ref_text: str = "",
        use_ema: bool = False,
        vocoder_name: str = "vocos",
        use_duration_predictor: bool = False,
        nfe_step: int = 32,
        cfg_strength: float = 2.0,
        speed: float = 1.0,
        cross_fade_duration: float = 0.15,
        sway_sampling_coef: float = -1.0,
        target_sample_rate: int = 24000,
        num_workers: int = 1,
    ):

        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=target_sample_rate,
            num_channels=1,
        )

        self._opts = _Conf(
            model_name=model_name,
            vocoder_name=vocoder_name,
            ckpt_path=ckpt_path,
            vocab_file=vocab_file,
            use_ema=use_ema,
            target_sample_rate=target_sample_rate,
            use_duration_predictor=use_duration_predictor,
        )
        self._generate_opts = GenerateConf(
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            speed=speed,
            cross_fade_duration=cross_fade_duration,
            sway_sampling_coef=sway_sampling_coef,
        )
        self._ref_opts = ReferenceInput(
            ref_audio_path=ref_audio_path, ref_id=ref_id, ref_text=ref_text
        )
        self.worker_pool = RoundrobinWorkerPool(
            num_workers, lambda: F5TTSWrapper(**self._opts.__dict__)
        )

    def synthesize(
        self,
        text: str,
        *,
        conn_options: livekit.agents.APIConnectOptions | None = None,
    ) -> "ChunkedStream":
        return ChunkedStream(
            tts=self,
            input_text=text,
            generate_opts=self._generate_opts,
            conn_options=conn_options,
            ref_opts=self._ref_opts,
            opts=self._opts,
            worker_pool=self.worker_pool,
        )


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        generate_opts: GenerateConf,
        conn_options: livekit.agents.APIConnectOptions,
        ref_opts: ReferenceInput,
        opts: _Conf,
        worker_pool: tp.Optional[RoundrobinWorkerPool[F5TTSWrapper]] = None,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts
        self._generate_opts = generate_opts
        if not worker_pool:
            self.worker_pool = RoundrobinWorkerPool(
                num_workers=1, factory_func=lambda: F5TTSWrapper(**self._opts.__dict__)
            )
        else:
            self.worker_pool = worker_pool
        self._ref_opts = ref_opts
        _, processed_ref_text = self.worker_pool.get().preprocess_reference(
            ref_audio_path=self._ref_opts.ref_audio_path,
            ref_text=(
                _preprocess_input_text(self._ref_opts.ref_text).strip()
                if self._ref_opts.ref_text
                else ""
            ),
            clip_short=False,
        )
        if not self._ref_opts.ref_text:
            self._ref_opts.ref_text = processed_ref_text

    async def _run(self):
        request_id = utils.shortuuid()
        decoder = utils.codecs.AudioStreamDecoder(
            sample_rate=self._opts.target_sample_rate, num_channels=1, format="wav"
        )

        @utils.log_exceptions(logger=logger)
        async def _decode_loop():
            try:
                async for chunk in _stream_audio_generator(
                    self.input_text,
                    model=self.worker_pool.get(),
                    **self._generate_opts.__dict__,
                ):
                    decoder.push(chunk)
            finally:
                decoder.end_input()

        decode_task = asyncio.create_task(_decode_loop())
        try:
            emitter = tts.SynthesizedAudioEmitter(
                event_ch=self._event_ch,
                request_id=request_id,
            )
            async for frame in decoder:
                emitter.push(frame)
            emitter.flush()
            await decode_task
        except Exception as e:
            import traceback

            logger.warning(traceback.format_exc())
            raise APIConnectionError() from e
        finally:
            await utils.aio.cancel_and_wait(decode_task)
            await decoder.aclose()
