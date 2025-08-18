import io
import wave
from livekit.agents import stt, APIConnectOptions

from livekit.agents.utils import AudioBuffer
from livekit import rtc
from pydub import AudioSegment
from faster_whisper import WhisperModel
import typing as tp
from transformers import pipeline
from scipy.io.wavfile import read as read_wave
import numpy as np


class STT(stt.STT):
    def __init__(
        self,
        model: str = "vinai/PhoWhisper-large",
        device: str = "auto",
        compute_type: str = "default",
        language: str = "vi",
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self._model = model
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.transcriber = pipeline(
            "automatic-speech-recognition", model=self._model, chunk_length_s=30
        )

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        lang = language if language else self.language
        transcriber = self.transcriber
        frame = rtc.combine_audio_frames(buffer)
        wav_io = io.BytesIO()

        with wave.open(wav_io, "wb") as wav:
            wav.setparams((frame.num_channels, 2, frame.sample_rate, 0, "NONE", "NONE"))
            wav.writeframes(buffer.data)
        wav_io.seek(0)
        sr, au = read_wave(wav_io)
        if au.ndim > 1:
            au = au.mean(axis=1)

        au = au.astype(np.float32)
        au /= np.max(np.abs(au))

        text = transcriber(
            inputs={"sampling_rate": sr, "raw": au},
        )["text"]
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text, language=language or "")],
        )
