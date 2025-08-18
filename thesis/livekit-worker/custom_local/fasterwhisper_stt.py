import io
import wave
from livekit.agents import stt, APIConnectOptions

from livekit.agents.utils import AudioBuffer
from livekit import rtc
from pydub import AudioSegment
from faster_whisper import WhisperModel
import typing as tp
from custom_denoiser import DemucsWrapper


class STT(stt.STT):
    def __init__(
        self,
        model_size_or_path: str = "Darejkal/vv-pp-ww-mm-cc-22",
        device: str = "auto",
        compute_type: str = "default",
        language: str = "vi",
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self.model_size_or_path = model_size_or_path
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.model = WhisperModel(
            model_size_or_path=self.model_size_or_path,
            device=self.device,
            compute_type=self.compute_type,
        )
        self.demucs = DemucsWrapper(device="cuda" if "cuda" in device else "cpu")

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        lang = language if language else self.language
        model = self.model
        frame = rtc.combine_audio_frames(buffer)
        frame = self.demucs.denoise(frame)
        wav_io = io.BytesIO()

        with wave.open(wav_io, "wb") as wav:
            wav.setparams((frame.num_channels, 2, frame.sample_rate, 0, "NONE", "NONE"))
            wav.writeframes(buffer.data)
        wav_io.seek(0)
        segments, info = model.transcribe(language=lang, audio=wav_io, beam_size=5)
        text = "".join(seg.text for seg in segments) or ""
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text, language=language or "")],
        )
