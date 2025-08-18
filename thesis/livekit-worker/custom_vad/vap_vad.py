# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import logging
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal, Tuple, Set, Optional, List, Any, Awaitable

import numpy as np
import torch
from cached_path import cached_path
import soundfile as sf

from livekit import agents, rtc
from livekit.agents import utils
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given
from pathlib import Path
from vap.model import VapGPT, VapConfig
logger = logging.getLogger("livekit.plugins.custom_vad")

NORM_FACTOR: float = 1 / (2 ** 15)
SLOW_INFERENCE_THRESHOLD = 0.2  # late by 200ms


@dataclass
class _VADOptions:
    min_speech_duration: float
    min_silence_duration: float
    prefix_padding_duration: float
    max_buffered_speech: float
    sample_rate: int
    context_duration: float
    frame_length: float
    activation_threshold: float = 0.5
    eot_threshold: float = 0.7  # End of turn threshold
    sot_threshold: float = 0.3  # Start of turn threshold


class VAPModel:
    def __init__(self, state_dict_path: str, context_duration: float = 5.0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model(state_dict_path)
        self.context_duration = context_duration
        self.sample_rate = 16000 
        self.channels = 2  
        self.buffer_samples = int(self.context_duration * self.sample_rate)
        self.audio_buffer = torch.zeros((1, self.channels, self.buffer_samples), dtype=torch.float32, device=self.device)

    def _load_model(self, state_dict_path: str) -> VapGPT:
        model_conf = VapConfig()
        model = VapGPT(model_conf)
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model = model.to(self.device)  # Ensure model is on the same device as buffer
        return model.eval()

    def add_audio_frame(self, audio_frame: np.ndarray) -> None:
        """Add stereo audio frame to rolling buffer"""
        audio_tensor = torch.from_numpy(audio_frame).float() * NORM_FACTOR

        if audio_tensor.ndim == 1:
            a = audio_tensor
            b = torch.zeros_like(audio_tensor)
        elif audio_tensor.ndim == 2 and audio_tensor.shape[0] == 2:
            a = audio_tensor[0]
            b = audio_tensor[1]
        elif audio_tensor.ndim == 2 and audio_tensor.shape[1] == 2:
            a = audio_tensor[:, 0]
            b = audio_tensor[:, 1]
        else:
            raise ValueError("Audio tensor must be mono or stereo.")
        chunk_size = a.shape[0]
        print(f"[DEBUG] add_audio_bytes_to_tensor: chunk_size={chunk_size}, a.max={a.max().item():.4f}, b.max={b.max().item():.4f}")

        # Roll buffer and add new audio
        self.audio_buffer = self.audio_buffer.roll(-chunk_size, -1)
        self.audio_buffer[0, 0, -chunk_size:] = a.to(self.device)
        self.audio_buffer[0, 1, -chunk_size:] = b.to(self.device)

    @torch.no_grad()
    def predict(self) -> Tuple[float, float]:
        """Get VAD predictions for both speakers using model.probs()"""
        try:
            out = self.model.probs(self.audio_buffer)
            # Take the mean of the last 10 frames for smoothing, adjust as needed
            p_now = out["p_now"][0, -10:, :]  # shape: (10, 2)
            vad_probs = p_now.mean(dim=0).cpu().numpy()  # shape: (2,)
            return float(vad_probs[0]), float(vad_probs[1])
        except Exception as e:
            logger.error(f"VAP prediction error: {e}")
            return 0.0, 0.0

    @torch.no_grad()
    def predict_eot(self, threshold: float = 0.5, speaker: int = 0) -> float:
        """
        Predict end-of-turn (EOT) probability for the given speaker.
        Returns the probability of the other speaker (i.e., the likelihood that the turn should switch).
        Args:
            threshold (float): Probability threshold for EOT (default 0.5)
            speaker (int): Speaker index (0 or 1, default 0)
        Returns:
            float: EOT probability (probability of the other speaker)
        """
        probs = self.predict()
        other_speaker = 1 - speaker
        eot_prob = float(probs[other_speaker])
        return eot_prob


class VAD(agents.vad.VAD):
    """
    VAP Voice Activity Detection (VAD) class.

    This class provides functionality to detect speech segments within audio data using the VAP model.
    """
    _model: VAPModel
    _opts: _VADOptions
    _streams: weakref.WeakSet["VADStream"]

    @classmethod
    def load(
        cls,
        *,
        min_speech_duration: float = 0.05,
        min_silence_duration: float = 0.55,
        prefix_padding_duration: float = 0.5,
        max_buffered_speech: float = 60.0,
        activation_threshold: float = 0.5,
        sample_rate: Literal[8000, 16000] = 16000,
        force_cpu: bool = False,
        context_duration: float = 10.0,
        frame_length: float = 5.0,
        state_dict_path: str = "",
        eot_threshold: float = 0.7,
        sot_threshold: float = 0.3,
        # deprecated
        padding_duration: NotGivenOr[float] = NOT_GIVEN,
    ) -> "VAD":
        if is_given(padding_duration):
            prefix_padding_duration = padding_duration

        if not state_dict_path:
            state_dict_path = str(cached_path("hf://Darejkal/hmmmmmm/hmmmm.pt"))

        opts = _VADOptions(
            min_speech_duration=min_speech_duration,
            min_silence_duration=min_silence_duration,
            prefix_padding_duration=prefix_padding_duration,
            max_buffered_speech=max_buffered_speech,
            activation_threshold=activation_threshold,
            sample_rate=sample_rate,
            context_duration=context_duration,
            frame_length=frame_length,
            eot_threshold=eot_threshold,
            sot_threshold=sot_threshold,
        )

        model = VAPModel(state_dict_path, context_duration)
        return cls(model=model, opts=opts)

    def __init__(
        self,
        *,
        model: VAPModel,
        opts: _VADOptions,
    ) -> None:
        super().__init__(capabilities=agents.vad.VADCapabilities(update_interval=opts.frame_length))
        self._model = model
        self._opts = opts
        self._streams = weakref.WeakSet[VADStream]()

    def stream(self) -> "VADStream":
        """
        Create a new VADStream for processing audio data.

        Returns:
            VADStream: A stream object for processing audio input and detecting speech.
        """
        stream = VADStream(self, self._opts, self._model)
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        min_speech_duration: NotGivenOr[float] = NOT_GIVEN,
        min_silence_duration: NotGivenOr[float] = NOT_GIVEN,
        prefix_padding_duration: NotGivenOr[float] = NOT_GIVEN,
        max_buffered_speech: NotGivenOr[float] = NOT_GIVEN,
        activation_threshold: NotGivenOr[float] = NOT_GIVEN,
        eot_threshold: NotGivenOr[float] = NOT_GIVEN,
        sot_threshold: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(min_speech_duration):
            self._opts.min_speech_duration = min_speech_duration
        if is_given(min_silence_duration):
            self._opts.min_silence_duration = min_silence_duration
        if is_given(prefix_padding_duration):
            self._opts.prefix_padding_duration = prefix_padding_duration
        if is_given(max_buffered_speech):
            self._opts.max_buffered_speech = max_buffered_speech
        if is_given(activation_threshold):
            self._opts.activation_threshold = activation_threshold

        for stream in self._streams:
            stream.update_options(
                min_speech_duration=min_speech_duration,
                min_silence_duration=min_silence_duration,
                prefix_padding_duration=prefix_padding_duration,
                max_buffered_speech=max_buffered_speech,
                activation_threshold=activation_threshold,
                eot_threshold=eot_threshold,
                sot_threshold=sot_threshold,
            )

    def update_options(
        self,
        *,
        min_speech_duration: NotGivenOr[float] = NOT_GIVEN,
        min_silence_duration: NotGivenOr[float] = NOT_GIVEN,
        prefix_padding_duration: NotGivenOr[float] = NOT_GIVEN,
        max_buffered_speech: NotGivenOr[float] = NOT_GIVEN,
        activation_threshold: NotGivenOr[float] = NOT_GIVEN,
        eot_threshold: NotGivenOr[float] = NOT_GIVEN,
        sot_threshold: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        if is_given(min_speech_duration):
            self._opts.min_speech_duration = min_speech_duration
        if is_given(min_silence_duration):
            self._opts.min_silence_duration = min_silence_duration
        if is_given(prefix_padding_duration):
            self._opts.prefix_padding_duration = prefix_padding_duration
        if is_given(max_buffered_speech):
            self._opts.max_buffered_speech = max_buffered_speech
        if is_given(activation_threshold):
            self._opts.activation_threshold = activation_threshold
        if is_given(eot_threshold):
            self._opts.eot_threshold = eot_threshold
        if is_given(sot_threshold):
            self._opts.sot_threshold = sot_threshold

        for stream in self._streams:
            stream.update_options(
                min_speech_duration=min_speech_duration,
                min_silence_duration=min_silence_duration,
                prefix_padding_duration=prefix_padding_duration,
                max_buffered_speech=max_buffered_speech,
                activation_threshold=activation_threshold,
                eot_threshold=eot_threshold,
                sot_threshold=sot_threshold,
            )


class VADStream(agents.vad.VADStream):
    _opts: _VADOptions
    _model: VAPModel
    _loop: asyncio.AbstractEventLoop
    _executor: ThreadPoolExecutor
    _exp_filter: utils.ExpFilter
    _input_sample_rate: int
    _speech_buffer: Optional[np.ndarray]
    _speech_buffer_max_reached: bool
    _prefix_padding_samples: int
    _last_inference_time: float
    _resampler: Optional[rtc.AudioResampler]

    def __init__(self, vad: VAD, opts: _VADOptions, model: VAPModel) -> None:
        super().__init__(vad)
        self._opts, self._model = opts, model
        self._loop = asyncio.get_event_loop()

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._task.add_done_callback(lambda _: self._executor.shutdown(wait=False))
        self._exp_filter = utils.ExpFilter(alpha=0.35)

        self._input_sample_rate = 0
        self._speech_buffer: np.ndarray | None = None
        self._speech_buffer_max_reached = False
        self._prefix_padding_samples = 0

        # VAP-specific state
        self._last_inference_time = 0.0

        # Add resampler state
        self._resampler: rtc.AudioResampler | None = None

    def update_options(
        self,
        *,
        min_speech_duration: NotGivenOr[float] = NOT_GIVEN,
        min_silence_duration: NotGivenOr[float] = NOT_GIVEN,
        prefix_padding_duration: NotGivenOr[float] = NOT_GIVEN,
        max_buffered_speech: NotGivenOr[float] = NOT_GIVEN,
        activation_threshold: NotGivenOr[float] = NOT_GIVEN,
        eot_threshold: NotGivenOr[float] = NOT_GIVEN,
        sot_threshold: NotGivenOr[float] = NOT_GIVEN,
    ) -> None:
        old_max_buffered_speech = self._opts.max_buffered_speech

        if is_given(min_speech_duration):
            self._opts.min_speech_duration = min_speech_duration
        if is_given(min_silence_duration):
            self._opts.min_silence_duration = min_silence_duration
        if is_given(prefix_padding_duration):
            self._opts.prefix_padding_duration = prefix_padding_duration
        if is_given(max_buffered_speech):
            self._opts.max_buffered_speech = max_buffered_speech
        if is_given(activation_threshold):
            self._opts.activation_threshold = activation_threshold
        if is_given(eot_threshold):
            self._opts.eot_threshold = eot_threshold
        if is_given(sot_threshold):
            self._opts.sot_threshold = sot_threshold

        if self._input_sample_rate:
            self._prefix_padding_samples = round(
                self._opts.prefix_padding_duration * self._input_sample_rate
            )

    @agents.utils.log_exceptions(logger=logger)
    async def _main_task(self) -> None:
        async for frame in self._input_ch:
            start_time = time.perf_counter()

            if not isinstance(frame, rtc.AudioFrame):
                continue

            # Initialize sample rate and resampler if needed
            if self._input_sample_rate == 0:
                self._input_sample_rate = frame.sample_rate
                self._prefix_padding_samples = round(
                    self._opts.prefix_padding_duration * self._input_sample_rate
                )
                if self._input_sample_rate != 16000:
                    self._resampler = rtc.AudioResampler(
                        input_rate=self._input_sample_rate,
                        output_rate=16000,
                        quality=rtc.AudioResamplerQuality.QUICK,
                    )

            # Resample to 16kHz if needed (VAP expects 16kHz)
            frames_for_vap = []
            if self._resampler is not None:
                frames_for_vap.extend(self._resampler.push(frame))
            else:
                frames_for_vap.append(frame)

            for vap_frame in frames_for_vap:
                # Convert frame to numpy array
                audio_data = np.frombuffer(vap_frame.data, dtype=np.int16).astype(np.float32) / 32768.0

                # Add to VAP model buffer
                self._model.add_audio_frame(audio_data)

                # Run inference
                inference_fut = self._loop.run_in_executor(
                    self._executor, self._model.predict
                )
                speaker1_prob, speaker2_prob = await inference_fut
                print(speaker1_prob,speaker2_prob)
                # Use the maximum probability as overall speech probability
                # speech_prob = max(speaker1_prob, speaker2_prob)
                # audio is mono only, use speak1 for now
                speech_prob=speaker1_prob
                # Apply exponential filter
                speech_prob = self._exp_filter.apply(exp=1.0, sample=speech_prob)

                inference_duration = time.perf_counter() - start_time
                if inference_duration > SLOW_INFERENCE_THRESHOLD:
                    logger.warning(
                        f"VAP inference took {inference_duration:.2f}s, "
                        f"threshold is {SLOW_INFERENCE_THRESHOLD}s"
                    )

                self._process_audio_frame(vap_frame, speech_prob, speaker1_prob)

    def _process_audio_frame(self, frame: rtc.AudioFrame, speech_prob: float, speaker1_prob: float) -> None:
        """Process audio frame with VAD probability"""
        is_speech = speech_prob >= self._opts.activation_threshold

        # Buffer management similar to Silero VAD
        if self._speech_buffer is None:
            max_samples = round(self._opts.max_buffered_speech * frame.sample_rate)
            self._speech_buffer = np.empty(
                (frame.num_channels, max_samples), dtype=np.int16
            )

        frame_samples = frame.samples_per_channel

        # Add frame to speech buffer
        if not self._speech_buffer_max_reached:
            current_samples = getattr(self, '_current_samples', 0)
            if current_samples + frame_samples <= self._speech_buffer.shape[1]:
                self._speech_buffer[:, current_samples:current_samples + frame_samples] = frame.data
                self._current_samples = current_samples + frame_samples
            else:
                self._speech_buffer_max_reached = True

        # Handle speech detection logic using turn-taking probabilities
        self._handle_speech_detection(frame, is_speech, speech_prob, speaker1_prob)

    def _handle_speech_detection(self, frame: rtc.AudioFrame, is_speech: bool, speech_prob: float, speaker1_prob: float) -> None:
        """Handle speech start/end detection using turn-taking probabilities"""
        
        # Check for END_OF_SPEECH: when speaker1_prob > eot_threshold
        if speaker1_prob > self._opts.eot_threshold:
            if hasattr(self, '_in_speech') and self._in_speech:
                # End speech immediately based on turn-taking probability
                self._in_speech = False
                
                # Create speech chunk with buffered data
                if hasattr(self, '_current_samples') and self._current_samples > 0:
                    speech_data = self._speech_buffer[:, :self._current_samples]
                    
                    speech_frame = rtc.AudioFrame(
                        data=speech_data,
                        sample_rate=frame.sample_rate,
                        num_channels=frame.num_channels,
                        samples_per_channel=speech_data.shape[1]
                    )
                    
                    event = agents.vad.VADEvent(
                        type=agents.vad.VADEventType.END_OF_SPEECH,
                        samples_index=0,
                        timestamp=time.time(),
                        silence_duration=0.0,
                        speech_duration=time.time() - getattr(self, '_speech_start_time', time.time()),
                        raw_inference_prob=speaker1_prob,
                        frames=[speech_frame]
                    )
                    self._event_ch.send_nowait(event)
                    
                    # Reset buffer
                    self._current_samples = 0
        
        # Check for START_OF_SPEECH: when speaker1_prob < sot_threshold
        elif speaker1_prob < self._opts.sot_threshold:
            if not hasattr(self, '_in_speech') or not self._in_speech:
                # Start speech based on turn-taking probability
                self._in_speech = True
                self._speech_start_time = time.time()
                
                # Create speech chunk with prefix padding if available
                if hasattr(self, '_current_samples'):
                    start_idx = max(0, self._current_samples - self._prefix_padding_samples)
                    speech_data = self._speech_buffer[:, start_idx:self._current_samples]
                    
                    speech_frame = rtc.AudioFrame(
                        data=speech_data,
                        sample_rate=frame.sample_rate,
                        num_channels=frame.num_channels,
                        samples_per_channel=speech_data.shape[1]
                    )
                    
                    event = agents.vad.VADEvent(
                        type=agents.vad.VADEventType.START_OF_SPEECH,
                        samples_index=0,
                        timestamp=self._speech_start_time,
                        silence_duration=0.0,
                        speech_duration=0.0,
                        raw_inference_prob=speaker1_prob,
                        frames=[speech_frame]
                    )
                    self._event_ch.send_nowait(event)

if __name__ == "__main__":
    import numpy as np
    import soundfile as sf
    from pathlib import Path
    audio_path = str(Path(__file__).parent/"ref.wav")
    audio_data, sr = sf.read(audio_path, dtype="int16")
    if audio_data.ndim == 1:
        audio_data = np.stack([audio_data, audio_data], axis=0)
    elif audio_data.shape[1] == 2:
        audio_data = audio_data.T
    elif audio_data.shape[0] == 2:
        pass
    else:
        raise ValueError("Audio file must be mono or stereo.")

    vad = VAD.load()
    model = vad._model
    n_samples = audio_data.shape[-1]
    duration = round(n_samples / model.sample_rate, 2)
    context_time = 20
    step_time = 0.5 
    chunk_time = context_time + step_time
    chunk_size = int(chunk_time * model.sample_rate)
    step_size = int(step_time * model.sample_rate)
    print(f"Audio duration: {duration}s, chunk_time: {chunk_time}s, chunk_size: {chunk_size}, step_size: {step_size}")

    for start in range(0, max(n_samples - chunk_size + 1,1), step_size):
        end = min(audio_data.shape[1],start + chunk_size)
        chunk = audio_data[:, start:end]
        model.add_audio_frame(chunk)
        prob1, prob2 = model.predict()
        print(f"Chunk {start}-{end}: Speaker 1: {prob1:.3f}, Speaker 2: {prob2:.3f}")
    # Optionally process the last chunk if not covered
    if n_samples % step_size != 0 and n_samples > chunk_size:
        chunk = audio_data[:, -chunk_size:]
        model.add_audio_frame(chunk)
        prob1, prob2 = model.predict()
        print(f"Chunk (last) {n_samples-chunk_size}-{n_samples}: Speaker 1: {prob1:.3f}, Speaker 2: {prob2:.3f}")


