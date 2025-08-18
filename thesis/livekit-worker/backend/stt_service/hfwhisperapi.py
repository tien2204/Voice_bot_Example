import litserve as ls
from transformers import pipeline
from pydantic import BaseModel
import numpy as np
import torch
import torchaudio.transforms as T
import typing as tp
import os
import logging
import base64
import io
import wave

logging.getLogger("litserve.server").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

from custom_denoiser import DemucsWrapper
from livekit import rtc  # Assuming rtc.AudioFrame is available


class TranscriptionResponse(BaseModel):
    transcription: str


class TranscriptionRequest(BaseModel):
    data: str  # base64 encoded audio bytes
    sample_rate: int
    num_channels: int
    samples_per_channel: int
    language: tp.Optional[str] = "vi"


class TransformedTranscriptionRequest(BaseModel):
    data: bytes  # decoded raw audio bytes
    sample_rate: int
    num_channels: int
    samples_per_channel: int
    language: tp.Optional[str] = "vi"


class WhisperLitAPI(ls.LitAPI):
    def setup(self, device: str):
        # Setup the Hugging Face pipeline for ASR
        # device: "cpu" or "cuda" or "cuda:0"
        device_id = 0 if "cuda" in device else -1

        logger.info(f"Loading Hugging Face ASR pipeline on device {device}...")
        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model="vinai/PhoWhisper-large",
            device=device_id,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            chunk_length_s=30,  # optional: process audio in chunks
        )
        logger.info("ASR pipeline loaded successfully.")

        # Setup denoiser
        self.demucs = DemucsWrapper(device="cuda" if "cuda" in device else "cpu")
        logger.info("Demucs denoiser loaded successfully.")

    def decode_request(self, request: TranscriptionRequest):
        # Decode base64 audio to bytes
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

        # Wrap raw bytes into rtc.AudioFrame for denoising
        audio_frame = rtc.AudioFrame(
            data=audio.data,
            sample_rate=audio.sample_rate,
            num_channels=audio.num_channels,
            samples_per_channel=audio.samples_per_channel,
        )

        # Denoise audio
        denoised_audio_frame = self.demucs.denoise(audio_frame)

        # Extract raw PCM from wav bytes
        with io.BytesIO(denoised_audio_frame.to_wav_bytes()) as wav_io:
            with wave.open(wav_io, "rb") as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                pcm_data = wf.readframes(n_frames)

        # Convert PCM bytes to numpy float32 normalized [-1, 1]
        audio_np = np.frombuffer(pcm_data, dtype=np.int16).astype(np.float32) / 32768.0

        # Convert multi-channel to mono by averaging
        if n_channels > 1:
            audio_np = audio_np.reshape(-1, n_channels)
            audio_np = audio_np.mean(axis=1)

        # Resample if necessary
        if sample_rate != MODEL_EXPECTED_SR:
            audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)  # (1, time)
            resampler = T.Resample(orig_freq=sample_rate, new_freq=MODEL_EXPECTED_SR)
            audio_np = resampler(audio_tensor).squeeze(0).numpy()

        # Normalize audio again
        max_val = np.max(np.abs(audio_np))
        if max_val > 0:
            audio_np /= max_val

        # Run ASR pipeline
        output = self.asr_pipeline(audio_np)

        transcription = output.get("text", "")

        return transcription

    def encode_response(self, output: str):
        return TranscriptionResponse(transcription=output)


if __name__ == "__main__":
    PORT = int(os.getenv("PORT", "8000"))
    api = WhisperLitAPI()
    server = ls.LitServer(api, accelerator="auto", timeout=1000, workers_per_device=2)
    logger.info(f"Starting WhisperLitAPI server on port {PORT}...")
    server.run(port=PORT)