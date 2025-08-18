from __future__ import annotations

import asyncio
import dataclasses
import time
import weakref
from dataclasses import dataclass
from typing import Callable, Union

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import DeadlineExceeded, GoogleAPICallError
from google.auth import default as gauth_default
from google.auth.exceptions import DefaultCredentialsError
from google.cloud.speech_v2 import SpeechAsyncClient
from google.cloud.speech_v2.types import cloud_speech
from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    stt,
    utils,
)
from livekit.agents.types import (
    NOT_GIVEN,
    NotGivenOr,
)
import io
import wave
import pyaudio
import google.genai as genai
import google.genai.types
from livekit.agents.utils import is_given
from livekit.plugins.google.log import logger
from livekit.plugins.google.models import SpeechLanguages, SpeechModels
from livekit.plugins.google.beta.realtime.api_proto import LiveAPIModels, Voice
import os

LgType = Union[SpeechLanguages, str]
LanguageCode = Union[LgType, list[LgType]]
_max_session_duration = 24000


class STT(stt.STT):
    def __init__(
        self,
        *,
        model: LiveAPIModels | str = "gemini-2.0-flash-live-001",
        api_key: NotGivenOr[str] = NOT_GIVEN,
        voice: Voice | str = "Puck",
        language: NotGivenOr[str] = NOT_GIVEN,
        location: NotGivenOr[str] = NOT_GIVEN,
        vertexai: bool = False,
        project: NotGivenOr[str] = NOT_GIVEN,
    ):
        """
        Create a new instance of Google STT.

        Credentials must be provided, either by using the ``credentials_info`` dict, or reading
        from the file specified in ``credentials_file`` or via Application Default Credentials as
        described in https://cloud.google.com/docs/authentication/application-default-credentials

        args:
            languages(LanguageCode): list of language codes to recognize (default: "en-US")
            detect_language(bool): whether to detect the language of the audio (default: True)
            interim_results(bool): whether to return interim results (default: True)
            punctuate(bool): whether to punctuate the audio (default: True)
            spoken_punctuation(bool): whether to use spoken punctuation (default: False)
            model(SpeechModels): the model to use for recognition default: "latest_long"
            location(str): the location to use for recognition default: "global"
            sample_rate(int): the sample rate of the audio default: 16000
            min_confidence_threshold(float): minimum confidence threshold for recognition
            (default: 0.65)
            credentials_info(dict): the credentials info to use for recognition (default: None)
            credentials_file(str): the credentials file to use for recognition (default: None)
            keywords(List[tuple[str, float]]): list of keywords to recognize (default: None)
        """
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )

        gemini_api_key = (
            api_key if is_given(api_key) else os.environ.get("GOOGLE_API_KEY")
        )
        gcp_project = (
            project if is_given(project) else os.environ.get("GOOGLE_CLOUD_PROJECT")
        )
        gcp_location = (
            location if is_given(location) else os.environ.get("GOOGLE_CLOUD_LOCATION")
        )

        if vertexai:
            if not gcp_project or not gcp_location:
                raise ValueError(
                    "Project and location are required for VertexAI either via project and location or GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables"
                )
            gemini_api_key = None
        else:
            gcp_project = None
            gcp_location = None
            if not gemini_api_key:
                raise ValueError(
                    "API key is required for Google API either via api_key or GOOGLE_API_KEY environment variable"
                )
        self.gemini_api_key = gemini_api_key
        self.vertexai = vertexai
        self.project = gcp_project
        self.location = gcp_location
        self.client = genai.Client(
            api_key=self.gemini_api_key,
            vertexai=self.vertexai,
            project=self.project,
            location=self.location,
        )

        self.language = language

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[SpeechLanguages | str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        frame = rtc.combine_audio_frames(buffer)
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wf:
            wf.setparams(
                (
                    frame.num_channels,
                    pyaudio.get_sample_size(pyaudio.paInt16),
                    frame.sample_rate,
                    0,
                    "NONE",
                    "NONE",
                )
            )
            wf.writeframes(frame.data.tobytes())
            wf.close()
        wav_io.seek(0)
        wav_file_bytes = wav_io.read()
        try:
            resp = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    f"Transcribe audio. No comments. If there is no recognizable sound, returns empty string. Target language: {self.language}",
                    google.genai.types.Part.from_bytes(
                        data=wav_file_bytes,
                        mime_type="audio/wav",
                    ),
                ],
            )

            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[
                    stt.SpeechData(
                        language=language,
                        text=resp.text,
                    )
                ],
            )
        except DeadlineExceeded:
            raise APITimeoutError() from None
        except GoogleAPICallError as e:
            raise APIStatusError(e.message, status_code=e.code or -1) from None
        except Exception as e:
            raise APIConnectionError() from e
