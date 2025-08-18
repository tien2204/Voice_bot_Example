import base64
import io
import os
import typing as tp
import wave
import asyncio
import httpx
import logging

from dotenv import load_dotenv
from livekit.agents import tts, utils, APIConnectionError, APIConnectOptions
from livekit.agents.utils import codecs


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()
DEFAULT_API_URL = os.getenv("PIPER_TTS_API_URL", "http://localhost:8001/predict")


class PiperTTSClient(tts.TTS):
    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        length_scale: float = 0.7,
        connect_timeout: float = 5.0,
        read_timeout: float = 60.0,
        sample_rate: int = 16000,
        num_channels: int = 1,
    ):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=num_channels,
        )
        self.api_url = api_url
        self.length_scale = length_scale
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

    def synthesize(
        self,
        text: str,
        *,
        conn_options: tp.Optional[APIConnectOptions] = None,
    ) -> "PiperTTSClientStream":
        return PiperTTSClientStream(
            tts_client=self,
            input_text=text,
            api_url=self.api_url,
            length_scale=self.length_scale,
            http_client=self._client,
        )


class PiperTTSClientStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts_client: PiperTTSClient,
        input_text: str,
        api_url: str,
        length_scale: float,
        http_client: httpx.AsyncClient,
    ):
        super().__init__(tts=tts_client, input_text=input_text, conn_options=None)
        self._api_url = api_url
        self._length_scale = length_scale
        self._http_client = http_client

    async def _run(self):
        request_id = utils.shortuuid()
        payload = {"text": self.input_text, "length_scale": self._length_scale}
        decoder = None
        try:
            logger.info(f"Sending request to {self._api_url} with payload: {payload}")
            response = await self._http_client.post(self._api_url, json=payload)
            response.raise_for_status()
            response_data = response.json()

            if (
                "audio_content" in response_data
                and response_data.get("content_type") == "audio/wav"
            ):
                audio_base64 = response_data["audio_content"]
                audio_bytes = base64.b64decode(audio_base64)

                with io.BytesIO(audio_bytes) as bio:
                    with wave.open(bio, "rb") as wf:
                        sample_rate = wf.getframerate()
                        num_channels = wf.getnchannels()

                decoder = codecs.AudioStreamDecoder(
                    sample_rate=sample_rate, num_channels=num_channels, format="wav"
                )
                decoder.push(audio_bytes)
                decoder.end_input()

                emitter = tts.SynthesizedAudioEmitter(
                    event_ch=self._event_ch, request_id=request_id
                )
                async for frame in decoder:
                    emitter.push(frame)
                emitter.flush()
                logger.info("TTS request ok!")
            else:
                logger.error(f"Unexpected response format: {response_data}")
                raise APIConnectionError(
                    f"Unexpected response format from TTS API: {response_data}"
                )

        except httpx.RequestError as e:
            logger.error(f"Error connecting to TTS API at {self._api_url}: {e}")
            raise APIConnectionError(f"API request failed: {e}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during TTS: {e}")
            raise APIConnectionError(f"Unexpected error during TTS: {e}") from e
        finally:
            if decoder:
                await decoder.aclose()


async def main_test():
    sample_text = "Chào bạn, đây là một thử nghiệm tổng hợp giọng nói."
    output_filename = "output_audio_client_class.wav"

    tts_service = PiperTTSClient()
    try:
        logger.info(f"Synthesizing text: '{sample_text}'")
        stream = tts_service.synthesize(sample_text)

        audio_data = bytearray()
        async for chunk_event in stream:
            if chunk_event.type == tts.SynthesizedAudioEventType.AUDIO_CHUNK:
                for frame in chunk_event.audio.frames:
                    audio_data.extend(frame.data)
            elif chunk_event.type == tts.SynthesizedAudioEventType.SYNTHESIS_COMPLETED:
                logger.info("Synthesis completed.")

        if audio_data:

            logger.info(f"Received {len(audio_data)} bytes of audio data.")
            logger.info(
                f"To properly save, you'd reconstruct a WAV file from these PCM frames or modify the stream to yield raw WAV bytes."
            )

        else:
            logger.error("Speech synthesis failed to produce audio data.")

    except Exception as e:
        logger.error(f"Error during TTS test: {e}")
    finally:
        await tts_service.close()


if __name__ == "__main__":
    asyncio.run(main_test())
