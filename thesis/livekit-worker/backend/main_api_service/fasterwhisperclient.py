import base64
import os
import wave
import logging
import io
import httpx
import asyncio
from dotenv import load_dotenv

from livekit.agents import stt, APIConnectOptions
from livekit.agents.utils import AudioBuffer
from livekit import rtc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
)
logger = logging.getLogger(__name__)

load_dotenv()

API_URL = os.getenv("FASTERWHISPER_API_URL", "http://localhost:8000/predict")
DEFAULT_API_URL = os.getenv("FASTERWHISPER_API_URL", "http://localhost:8000/predict")


class FasterWhisperClientSTT(stt.STT):
    """
    Sends an audio file to the FasterWhisper API for transcription.
    STT class that uses a remote FasterWhisper API service for transcription.
    """

    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        language: str = "vi",
        connect_timeout: float = 5.0,
        read_timeout: float = 30.0,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        self.api_url = api_url
        self.language = language
        if not connect_timeout:
            connect_timeout = 10.0
        if not read_timeout:
            connect_timeout = 60.0
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=connect_timeout, read=read_timeout, write=5.0, pool=5.0
            )
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
        """
        Sends an audio buffer to the FasterWhisper API for transcription.
        """
        current_language = language if language else self.language
        frame = rtc.combine_audio_frames(buffer)
        audio_bytes_b64 = base64.b64encode(frame.data.tobytes()).decode("utf-8")
        payload = {
            "data": audio_bytes_b64,
            "sample_rate": frame.sample_rate,
            "num_channels": frame.num_channels,
            "samples_per_channel": 0,
            "language": current_language,
        }

        logger.info(
            f"Sending request to {self.api_url} with language {current_language}"
        )

        try:
            response = await self._client.post(self.api_url, json=payload)
            response.raise_for_status()
            response_data = response.json()

            if "transcription" in response_data:
                transcription = response_data["transcription"]
                logger.info(f"Transcription received: {transcription}")
                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[
                        stt.SpeechData(text=transcription, language=current_language)
                    ],
                )
            else:
                logger.error(f"Error: Unexpected response format: {response_data}")

                return stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    alternatives=[stt.SpeechData(text="", language=current_language)],
                )

        except httpx.RequestError as e:
            logger.error(
                f"Error connecting to Transcription API at {self.api_url}: {e}"
            )
            raise ConnectionError(f"API request failed: {e}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            raise RuntimeError(f"Unexpected error during transcription: {e}") from e


async def main_test():

    dummy_file_path = "dummy_audio_for_transcription_client.wav"

    logger.info("This test requires a running FasterWhisper API service.")
    logger.info(f"And a dummy audio file: {dummy_file_path}")


if __name__ == "__main__":

    asyncio.run(main_test())
    logger.info(
        f"Please manually remove the dummy audio file if not needed: dummy_audio_for_transcription_client.wav"
    )
