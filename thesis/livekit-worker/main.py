import logging

logger = logging.getLogger(__name__)

logging.getLogger("websockets.client").setLevel(logging.INFO)

import asyncio

from dotenv import load_dotenv
from pathlib import Path

load_dotenv((Path(__file__).parent / ".vetc.env").__str__())

from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm, stt
from livekit.agents import metrics, MetricsCollectedEvent, RoomOutputOptions, JobProcess
from livekit.agents.voice.agent_session import AgentSession
from livekit.plugins import silero

from agent import CarPartAgent
from custom_gemini.realtime import (
    RealtimeModel as CustomRealtimeModel,
    AudioTranscriptionConfig,
)
from custom_gemini.geministt import STT as CustomGeminiSTT
import livekit.rtc.data_stream
import asyncio


_active_tasks = set()


def get_text_handler(session: AgentSession):
    def handle_text_stream(
        reader: livekit.rtc.data_stream.TextStreamReader, participant_identity: str
    ):
        async def async_handle_text_stream(
            reader: livekit.rtc.data_stream.TextStreamReader, participant_identity: str
        ):
            info = reader.info

            logger.info(
                f"Text stream received from {participant_identity}\n"
                f"  Topic: {info.topic}\n"
                f"  Timestamp: {info.timestamp}\n"
                f"  ID: {info.stream_id}\n"
                f"  Size: {info.size}"
            )
            text = await reader.read_all()
            await session.generate_reply(user_input=text)
            print(f"Received text: {text}")

        task = asyncio.create_task(
            async_handle_text_stream(reader, participant_identity)
        )
        _active_tasks.add(task)
        task.add_done_callback(lambda t: _active_tasks.remove(t))

    return handle_text_stream


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    await ctx.connect()
    session = AgentSession(
        llm=CustomRealtimeModel(
            model="gemini-2.0-flash-exp",
            voice="Aoede",
            temperature=0.8,
            instructions="You are a helpful assistant",
            language="vi-VN",
        ),
    )

    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await ctx.wait_for_participant()
    await session.start(
        room=ctx.room,
        agent=CarPartAgent(
            stt=CustomGeminiSTT(language="vi-VN"),
            vad=ctx.proc.userdata["vad"],
        ),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )
    ctx.room.register_text_stream_handler(
        "custom-agent-text-input", get_text_handler(session)
    )
    await session.generate_reply(
        instructions="Greet the user and offer your assistance in Vietnamese."
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm, port=12312)
    )
