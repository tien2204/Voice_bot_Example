import torch
import torchaudio
from denoiser.demucs import DemucsStreamer
from denoiser.pretrained import get_model, Demucs
from denoiser.utils import bold
from livekit import rtc
import numpy as np
import io
import multiprocessing as mp
from types import SimpleNamespace
import asyncio
import typing as tp


class DefaultNamespace(SimpleNamespace):
    def __getattr__(self, name):
        return False


class DemucsWrapper:
    def __init__(self, model_name="htdemucs", dry=0.04, num_frames=1, device="cpu"):
        self.device = device
        self.model: Demucs = get_model(
            DefaultNamespace(type="demucs", name=model_name)
        ).to(self.device)
        self.model.eval()
        self.streamer: DemucsStreamer = DemucsStreamer(
            self.model, dry=dry, num_frames=num_frames
        )
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()

    def denoise(self, audio: tp.Union[rtc.AudioFrame, tp.Dict]):
        if isinstance(audio, dict):
            audio = rtc.AudioFrame(
                data=audio["data"],
                sample_rate=audio["sample_rate"],
                num_channels=audio["num_channels"],
                samples_per_channel=audio["samples_per_channel"],
            )
        return self.run_live_denoising(
            streamer=self.streamer, device=self.device, audio=audio
        )

    @staticmethod
    def run_live_denoising(
        streamer: DemucsStreamer,
        device: str,
        audio: rtc.AudioFrame,
        compressor=True,
        num_threads=None,
    ):
        if num_threads:
            torch.set_num_threads(num_threads)

        frame = (
            np.frombuffer(audio.to_wav_bytes(), dtype=np.int16).astype(np.float32)
            / 32768.0
        )

        frame = frame.reshape(-1, audio.num_channels)
        frame = frame.mean(axis=1)

        frame = torch.from_numpy(frame).to(device)

        with torch.no_grad():
            out = streamer.feed(frame[None])[0]

        if not out.numel():
            return audio

        if compressor:
            out = 0.99 * torch.tanh(out)

        out = out.clamp(-1, 1).squeeze(0).cpu().numpy()

        int16_audio = (out * 32768.0).astype(np.int16).tobytes()

        streamer.flush()

        return rtc.AudioFrame(
            data=int16_audio,
            sample_rate=audio.sample_rate,
            num_channels=1,
            samples_per_channel=len(out),
        )
