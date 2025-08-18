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
    def __init__(
        self,
        model_path=None,  # path to model checkpoint if needed
        dry=0.04,
        device="cpu",
        sample_rate=16000,
    ):
        self.device = device
        self.dry = dry
        self.sample_rate = sample_rate

        # Init your Demucs model
        self.model = Demucs(
            chin=1,
            chout=1,
            hidden=48,
            depth=5,
            kernel_size=8,
            stride=4,
            causal=True,
            resample=4,
            growth=2,
            max_hidden=10_000,
            normalize=True,
            glu=True,
            rescale=0.1,
            floor=1e-3,
            sample_rate=sample_rate,
        ).to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=device))

        self.model.eval()

    def denoise(self, audio: tp.Union[rtc.AudioFrame, tp.Dict]) -> rtc.AudioFrame:
        if isinstance(audio, dict):
            audio = rtc.AudioFrame(
                data=audio["data"],
                sample_rate=audio["sample_rate"],
                num_channels=audio["num_channels"],
                samples_per_channel=audio["samples_per_channel"],
            )

        # Decode raw bytes to float32 waveform
        frame = np.frombuffer(audio.data, dtype=np.int16).astype(np.float32) / 32768.0
        frame = frame.reshape(-1, audio.num_channels)
        frame = frame.mean(axis=1)  # convert to mono
        frame_tensor = torch.tensor(frame, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, T)

        with torch.no_grad():
            out_tensor = self.model(frame_tensor).squeeze(0)

        # Post-process output
        out_tensor = (1 - self.dry) * out_tensor + self.dry * frame_tensor.squeeze(0)
        out_tensor = 0.99 * torch.tanh(out_tensor)  # simple compressor
        out_tensor = torch.clamp(out_tensor, -1.0, 1.0)

        # Convert back to int16 bytes
        out_int16 = (out_tensor.cpu().numpy() * 32768.0).astype(np.int16)
        out_bytes = out_int16.tobytes()

        return rtc.AudioFrame(
            data=out_bytes,
            sample_rate=audio.sample_rate,
            num_channels=1,
            samples_per_channel=audio.samples_per_channel,
        )

class OldDemucsWrapper:
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
        streamer = streamer
        model: Demucs = streamer.demucs
        if num_threads:
            torch.set_num_threads(num_threads)

        first = True
        current_time = 0
        length = streamer.total_length if first else streamer.stride
        first = False
        current_time += length / model.sample_rate
        frame = (
            np.frombuffer(audio.to_wav_bytes(), dtype=np.int16).astype(np.float32)
            / 32768.0
        )
        frame = np.tile(frame[:, None], (1, 2))
        frame = torch.from_numpy(frame).mean(dim=1).to(device)
        with torch.no_grad():
            out = streamer.feed(frame[None])[0]
        if not out.numel():
            return audio
        if compressor:
            out = 0.99 * torch.tanh(out)
        out = out[:, None].repeat(1, 1)
        out.clamp_(-1, 1)
        out = out.cpu().numpy()
        print(streamer.flush())
        return rtc.AudioFrame(
            data=((out* 32768.0).astype(np.int16)).tobytes(),
            sample_rate=audio.sample_rate,
            num_channels=audio.num_channels,
            samples_per_channel=audio.samples_per_channel,
        )