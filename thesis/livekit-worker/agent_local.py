import enum
from livekit.agents.llm import function_tool
import logging
import os
import weaviate
import pandas as pd
import re
import pandas as pd
from livekit.agents import Agent, AgentSession, ModelSettings, stt, utils
import asyncio
from collections.abc import AsyncGenerator, AsyncIterable, Coroutine
from livekit import rtc
from typing import Any
import duckdb
from unidecode import unidecode

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class HustAgent(Agent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            **kwargs,
            instructions=(
                """\
You are a Vietnamese voice assisstant. Follow the rules:
- Do not output nonword characters, no abbreviation, no markdown.
- You answer questions related to Đại học Bách Khoa Hà Nội.
- Transcription inputs might be wrong.
- Be concise, reject questions unrelated to Đại học Bách Khoa Hà Nội.\
"""
            ),
        )

    def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> (
        AsyncIterable[stt.SpeechEvent | str]
        | Coroutine[Any, Any, AsyncIterable[stt.SpeechEvent | str]]
        | Coroutine[Any, Any, None]
    ):
        async def _stt_node(
            agent: Agent,
            audio: AsyncIterable[rtc.AudioFrame],
            model_settings: ModelSettings,
        ) -> AsyncGenerator[stt.SpeechEvent, None]:
            activity = agent._get_activity_or_raise()
            assert (
                activity.stt is not None
            ), "stt_node called but no STT node is available"

            wrapped_stt = activity.stt

            if not activity.stt.capabilities.streaming:
                if not activity.vad:
                    raise RuntimeError(
                        f"The STT ({activity.stt.label}) does not support streaming, add a VAD to the AgentTask/VoiceAgent to enable streaming"
                        "Or manually wrap your STT in a stt.StreamAdapter"
                    )

                wrapped_stt = stt.StreamAdapter(stt=wrapped_stt, vad=activity.vad)

            async with wrapped_stt.stream() as stream:

                @utils.log_exceptions(logger=logger)
                async def _forward_input():
                    async for frame in audio:
                        stream.push_frame(frame)

                forward_task = asyncio.create_task(_forward_input())
                try:
                    async for event in stream:
                        yield event
                finally:
                    await utils.aio.cancel_and_wait(forward_task)

        return _stt_node(agent=self, audio=audio, model_settings=model_settings)


class SimpleAgent(Agent):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(
            *args,
            **kwargs,
            instructions=(
                """\
You are a helpful assistant.
"""
            ),
        )

    def stt_node(
        self, audio: AsyncIterable[rtc.AudioFrame], model_settings: ModelSettings
    ) -> (
        AsyncIterable[stt.SpeechEvent | str]
        | Coroutine[Any, Any, AsyncIterable[stt.SpeechEvent | str]]
        | Coroutine[Any, Any, None]
    ):
        async def _stt_node(
            agent: Agent,
            audio: AsyncIterable[rtc.AudioFrame],
            model_settings: ModelSettings,
        ) -> AsyncGenerator[stt.SpeechEvent, None]:
            activity = agent._get_activity_or_raise()
            assert (
                activity.stt is not None
            ), "stt_node called but no STT node is available"

            wrapped_stt = activity.stt

            if not activity.stt.capabilities.streaming:
                if not activity.vad:
                    raise RuntimeError(
                        f"The STT ({activity.stt.label}) does not support streaming, add a VAD to the AgentTask/VoiceAgent to enable streaming"
                        "Or manually wrap your STT in a stt.StreamAdapter"
                    )

                wrapped_stt = stt.StreamAdapter(stt=wrapped_stt, vad=activity.vad)

            async with wrapped_stt.stream() as stream:

                @utils.log_exceptions(logger=logger)
                async def _forward_input():
                    async for frame in audio:
                        stream.push_frame(frame)

                forward_task = asyncio.create_task(_forward_input())
                try:
                    async for event in stream:
                        yield event
                finally:
                    await utils.aio.cancel_and_wait(forward_task)

        return _stt_node(agent=self, audio=audio, model_settings=model_settings)
