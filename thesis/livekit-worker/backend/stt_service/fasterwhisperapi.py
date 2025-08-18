import litserve as ls
from faster_whisper import WhisperModel
from pydantic import BaseModel
import numpy as np
import torch
import torchaudio.transforms as T
import typing as tp
import os
import logging

logging.getLogger("litserve.server").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

from custom_denoiser import DemucsWrapper
from livekit import rtc


class TranscriptionResponse(BaseModel):
    transcription: str


class TranscriptionRequest(BaseModel):
    data: str
    sample_rate: int
    num_channels: int
    samples_per_channel: int
    language: tp.Optional[str] = "vi"


class TransformedTranscriptionRequest(BaseModel):
    data: bytes
    sample_rate: int
    num_channels: int
    samples_per_channel: int
    language: tp.Optional[str] = "vi"


class WhisperLitAPI(ls.LitAPI):
    def setup(self, device: str):

        if ":" in str(device):
            device_index = int(str(device).rsplit(":", 1)[1])
        else:
            device_index = None

        self.model = WhisperModel(
            "Darejkal/vv-pp-ww-mm-cc-22",
            device="cuda" if "cuda" in str(device) else "cpu",
            device_index=device_index,
        )
        logger.info(f"Whisper model loaded on {device}")

        self.demucs = DemucsWrapper(device="cuda" if "cuda" in str(device) else "cpu")
        logger.info(f"Demucs denoiser loaded on {device}")

    def decode_request(self, request: TranscriptionRequest):
        import base64

        raw_bytes = base64.b64decode(request.data)

        return TransformedTranscriptionRequest(
            data=raw_bytes,
            sample_rate=request.sample_rate,
            num_channels=request.num_channels,
            samples_per_channel=request.samples_per_channel,
            language=request.language,
        )

    def predict(self, audio: TransformedTranscriptionRequest):
        MODEL_EXPECTED_SR = 16000

        audio_frame = rtc.AudioFrame(
            data=audio.data,
            sample_rate=audio.sample_rate,
            num_channels=audio.num_channels,
            samples_per_channel=audio.samples_per_channel,
        )

        denoised_audio_frame = self.demucs.denoise(audio_frame)

        import io
        import wave

        with io.BytesIO(denoised_audio_frame.to_wav_bytes()) as wav_io:
            with wave.open(wav_io, "rb") as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                pcm_data = wf.readframes(n_frames)

        audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

        if n_channels > 1:
            audio_np = audio_np.reshape(-1, n_channels)
            audio_np = audio_np.mean(axis=1)

        if sample_rate != MODEL_EXPECTED_SR:
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
            resampler = T.Resample(orig_freq=sample_rate, new_freq=MODEL_EXPECTED_SR)
            audio_np = resampler(audio_tensor).squeeze(0).numpy()

        max_val = np.max(np.abs(audio_np))
        if max_val > 0:
            audio_np /= max_val

        segments, _ = self.model.transcribe(audio_np, language=audio.language)
        transcription = " ".join(segment.text for segment in segments)

        return transcription

    def encode_response(self, output: str):
        return TranscriptionResponse(transcription=output)


if __name__ == "__main__":
    PORT = int(os.getenv("PORT", "8000"))
    api = WhisperLitAPI()
    server = ls.LitServer(api, accelerator="auto", timeout=1000, workers_per_device=2)
    logger.info(f"Starting WhisperLitAPI server on port {PORT}...")
    server.run(port=PORT)
