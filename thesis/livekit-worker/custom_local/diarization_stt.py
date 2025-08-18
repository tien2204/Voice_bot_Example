import io
import wave
from livekit.agents import stt, APIConnectOptions

from livekit.agents.utils import AudioBuffer
from livekit import rtc
from pydub import AudioSegment
from faster_whisper import WhisperModel
import typing as tp
import whisperx.diarize as diarize
import whisperx.transcribe as transcribe
import whisperx.alignment as alignment
import whisperx.audio as waudio
import os
import tempfile
from whisperx.types import (
    TypedDict,
    SingleAlignedSegment,
    List,
    SingleWordSegment,
)
import asyncio


class SingleDiarizedSegment(SingleWordSegment):
    speaker: str


class SingleDiarizedSegment(SingleAlignedSegment):
    speaker: str


class DiarizationResult(TypedDict):
    """
    A list of segments and word segments of a speech.
    """

    segments: List[SingleDiarizedSegment]


class STT(stt.STT):
    def __init__(
        self,
        model_size_or_path: str = "Darejkal/vv-pp-ww-mm-cc-22",
        device: str = "cpu",
        compute_type: str = "default",
        language: str = "vi",
        use_auth_token=None,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        if use_auth_token:
            os.environ["HF_TOKEN"] = use_auth_token
        if device == "auto":
            device == "cpu"
        self.model_size_or_path = model_size_or_path
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.alignment, self.alignment_metadata = alignment.load_align_model(
            language_code=language, device=device
        )
        self.diarization = diarize.DiarizationPipeline(
            use_auth_token=use_auth_token, device=device
        )
        self.whisper = transcribe.load_model(
            model_size_or_path, device, compute_type=compute_type
        )

    async def _transcription_task(self, language, audio):
        result = self.whisper.transcribe(language=language, audio=audio)
        result = alignment.align(
            result["segments"],
            model=self.alignment,
            align_model_metadata=self.alignment_metadata,
            audio=audio,
            device=self.device,
            return_char_alignments=False,
        )
        return result

    async def _diarization_task(self, audio):
        return self.diarization(audio=audio)

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        lang = language if language else self.language
        frame = rtc.combine_audio_frames(buffer)

        fi = tempfile.NamedTemporaryFile(
            "w", suffix=".wav", dir="/tmp/audio", delete=False
        )
        with wave.open(fi.name, "wb") as wav:
            wav.setparams((frame.num_channels, 2, frame.sample_rate, 0, "NONE", "NONE"))
            wav.writeframes(buffer.data)
        audio = waudio.load_audio(fi.name, frame.sample_rate)
        transcription_result, diarization_result = await asyncio.gather(
            asyncio.create_task(self._transcription_task(language=lang, audio=audio)),
            asyncio.create_task(self._diarization_task(audio=audio)),
        )
        result: DiarizationResult = diarize.assign_word_speakers(
            diarization_result, transcription_result
        )
        text = (
            "".join(
                x
                for seg in result["segments"]
                for x in (
                    [
                        str(seg["speaker"]),
                        ": ",
                        seg["text"],
                        "\n",
                    ]
                    if seg.get("speaker")
                    else []
                )
            )
            or ""
        )
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text, language=language or "")],
        )
