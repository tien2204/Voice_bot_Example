import base64
import io
import wave
import logging
import httpx
import asyncio
import numpy as np
import torch
import soundfile as sf
import json
from livekit.agents import stt, APIConnectOptions
from livekit.agents.utils import AudioBuffer
from livekit import rtc
from livekit.agents.inference_runner import _InferenceRunner
from custom_vad.vap_vad import VAPModel
from livekit.agents.job import get_job_context
from cached_path import cached_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SEC = 5
FASTERWHISPER_API_URL = "http://localhost:8000/predict"
VAP_MODEL_PATH = str(cached_path("hf://Darejkal/hmmmmmm/hmmmm.pt"))  # Update as needed

class VAPFasterWhisperSTT(stt.STT):
    def __init__(
        self,
        api_url: str = FASTERWHISPER_API_URL,
        language: str = "vi",
        connect_timeout: float = 5.0,
        read_timeout: float = 30.0,
        vap_model_path: str = VAP_MODEL_PATH,
        buffer_size_seconds: float = 0.5,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self.api_url = api_url
        self.language = language
        self.buffer_size_seconds = buffer_size_seconds
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=connect_timeout, read=read_timeout, write=5.0, pool=5.0
            )
        )
        self._vap_model = VAPModel(state_dict_path=vap_model_path)
        
    def stream(
        self,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions = APIConnectOptions(),
    ) -> "SpeechStream":
        from dataclasses import dataclass
        
        @dataclass
        class STTOptions:
            sample_rate: int = 16000
            buffer_size_seconds: float = self.buffer_size_seconds
            
        opts = STTOptions()
        return SpeechStream(
            stt=self,
            opts=opts,
            conn_options=conn_options,
        )

    async def close(self):
        await self._client.aclose()

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: str | None = None,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        current_language = language if language else self.language
        frame = rtc.combine_audio_frames(buffer)

        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wf:
            wf.setparams((frame.num_channels, 2, frame.sample_rate, 0, "NONE", "NONE"))
            wf.writeframes(buffer.data)
        wav_io.seek(0)
        audio_bytes = wav_io.read()
        audio_bytes_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        payload = {
            "data": audio_bytes_b64,
            "sample_rate": frame.sample_rate,
            "num_channels": frame.num_channels,
            "samples_per_channel": 0,
            "language": current_language,
        }

        async def stt_task():
            logger.info(f"Sending request to {self.api_url} with language {current_language}")
            response = await self._client.post(self.api_url, json=payload)
            response.raise_for_status()
            response_data = response.json()
            transcription = response_data.get("transcription", "")
            logger.info(f"Transcription received: {transcription}")
            return transcription

        async def vap_task():
            # Run VAPModel directly
            audio, sr = sf.read(io.BytesIO(audio_bytes))
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)
            audio_tensor = torch.from_numpy(audio).float()
            self._vap_model.add_audio_frame(np.stack([audio_tensor, np.zeros_like(audio_tensor)], axis=0))
            eot_prob = self._vap_model.predict_eot()
            return eot_prob

        transcription, eou_prob = await asyncio.gather(stt_task(), vap_task())

        logger.info(f"VAP EOU probability: {eou_prob}")

        final_event = stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[
                stt.SpeechData(text=transcription, language=current_language)
            ],
        )

        eou_threshold = 0.5  # or make this configurable
        events = [final_event]
        if eou_prob > eou_threshold:
            events.append(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.END_OF_SPEECH,
                    alternatives=[],
                )
            )

        return events[-1]

class SpeechStream(stt.SpeechStream):
    _CLOSE_MSG: str = json.dumps({"terminate_session": True})

    def __init__(
        self,
        *,
        stt: VAPFasterWhisperSTT,
        opts: object,  # You can define a dataclass for options if needed
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=opts.sample_rate)
        self._opts = opts
        self._speech_duration: float = 0
        self._reconnect_event = asyncio.Event()
        self._client = stt._client
        self._vap_model = stt._vap_model
        self._api_url = stt.api_url
        self._language = stt.language

    def update_options(self, **kwargs) -> None:
        self._reconnect_event.set()

    async def _run(self) -> None:
        buffer = bytearray()
        sample_rate = getattr(self._opts, 'sample_rate', 16000)
        chunk_size = int(sample_rate * getattr(self._opts, 'buffer_size_seconds', 0.5) * 2)  # 2 bytes per sample

        async def process_and_emit(audio_bytes: bytes):
            try:
                async def stt_task():
                    wav_io = io.BytesIO()
                    with wave.open(wav_io, "wb") as wf:
                        wf.setparams((1, 2, sample_rate, 0, "NONE", "NONE"))
                        wf.writeframes(audio_bytes)
                    wav_io.seek(0)
                    audio_bytes_wav = wav_io.read()
                    audio_bytes_b64 = base64.b64encode(audio_bytes_wav).decode("utf-8")
                    payload = {
                        "data": audio_bytes_b64,
                        "sample_rate": sample_rate,
                        "num_channels": 1,
                        "samples_per_channel": 0,
                        "language": self._language,
                    }
                    response = await self._client.post(self._api_url, json=payload)
                    response.raise_for_status()
                    response_data = response.json()
                    transcription = response_data.get("transcription", "")
                    return transcription

                async def vap_task():
                    wav_io = io.BytesIO()
                    with wave.open(wav_io, "wb") as wf:
                        wf.setparams((1, 2, sample_rate, 0, "NONE", "NONE"))
                        wf.writeframes(audio_bytes)
                    wav_io.seek(0)
                    audio, sr = sf.read(wav_io)
                    if len(audio.shape) > 1:
                        audio = np.mean(audio, axis=1)
                    audio_tensor = torch.from_numpy(audio).float()
                    self._vap_model.add_audio_frame(np.stack([audio_tensor, np.zeros_like(audio_tensor)], axis=0))
                    eot_prob = self._vap_model.predict_eot()
                    return eot_prob

                transcription, eou_prob = await asyncio.gather(stt_task(), vap_task())
                
                if transcription.strip():
                    final_event = stt.SpeechEvent(
                        type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                        alternatives=[stt.SpeechData(text=transcription, language=self._language)],
                    )
                    self._event_ch.send_nowait(final_event)
                
                eou_threshold = 0.5
                if eou_prob > eou_threshold:
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH, alternatives=[])
                    )
            except Exception as e:
                logger.exception(f"Error processing audio chunk: {e}")

        async for data in self._input_ch:
            try:
                if isinstance(data, self._FlushSentinel):
                    if buffer:
                        await process_and_emit(bytes(buffer))
                        buffer.clear()
                    break
                
                buffer.extend(data.data.tobytes())
                if len(buffer) >= chunk_size:
                    await process_and_emit(bytes(buffer))
                    buffer.clear()
            except Exception as e:
                logger.exception(f"Error in SpeechStream: {e}")
                break
