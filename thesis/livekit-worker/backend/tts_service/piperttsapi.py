import io, base64
import torch
import litserve as ls
from piper import PiperVoice
from cached_path import cached_path
from pydantic import BaseModel
import typing as tp
import logging
import os
import wave

VOICE_MODEL_PATH = cached_path(
    f"hf://Darejkal/letsbetogether/cahaibenhaycogangobennhau.onnx"
)
CONFIG_PATH = cached_path(
    f"hf://rhasspy/piper-voices/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx.json"
)


class TTSRequest(BaseModel):
    text: str
    length_scale: tp.Optional[float] = 0.7


class TTSResponse(BaseModel):
    audio_content: str
    content_type: str = "audio/wav"


class PiperTextToWavLitAPI(ls.LitAPI):
    def setup(self, device: str) -> None:
        use_cuda = device == "cuda" and torch.cuda.is_available()
        self.voice = PiperVoice.load(
            model_path=VOICE_MODEL_PATH, config_path=CONFIG_PATH, use_cuda=use_cuda
        )
        logging.info("Piper model loaded successfully.")

    def decode_request(self, request: TTSRequest) -> TTSRequest:
        return request

    def predict(self, request_data: TTSRequest) -> bytes:
        text = request_data.text
        length_scale = request_data.length_scale

        with io.BytesIO() as audio_buffer:
            with wave.open(audio_buffer, "w") as fiwrite:
                self.voice.synthesize(text, fiwrite, length_scale=length_scale)
            audio_buffer.seek(0)
            audio_data = audio_buffer.getvalue()
        return audio_data

    def encode_response(self, audio_data: bytes) -> TTSResponse:
        audio_content_base64 = base64.b64encode(audio_data).decode("utf-8")
        return TTSResponse(audio_content=audio_content_base64)


if __name__ == "__main__":
    PORT = int(os.getenv("PORT", "8001"))
    api = PiperTextToWavLitAPI()

    server = ls.LitServer(api, accelerator="auto")
    server.run(port=PORT)
