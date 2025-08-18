import logging

logger = logging.getLogger(__name__)
import asyncio

from dotenv import load_dotenv

load_dotenv()

from livekit.agents import (
    AutoSubscribe,
    JobContext,
    WorkerOptions,
    WorkerType,
    cli,
    llm,
    stt,
)
from livekit.agents import metrics, MetricsCollectedEvent, RoomOutputOptions, JobProcess
from livekit.agents.voice.agent_session import AgentSession
from livekit.plugins import silero, openai
from custom_local import pipertts, fasterwhisper_stt, hf_stt
from agent_local import HustAgent
import livekit.rtc.data_stream
import httpx

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

            await session.say(text)

        task = asyncio.create_task(
            async_handle_text_stream(reader, participant_identity)
        )
        _active_tasks.add(task)
        task.add_done_callback(lambda t: _active_tasks.remove(t))

    return handle_text_stream


import json


def ensure_number(value):
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def get_config_handler(session: AgentSession):
    def handle_text_stream(
        reader: livekit.rtc.data_stream.TextStreamReader, participant_identity: str
    ):
        async def async_handle_config_stream(
            reader: livekit.rtc.data_stream.TextStreamReader, participant_identity: str
        ):
            info = reader.info

            logger.info(
                f"Config stream received from {participant_identity}\n"
                f"  Topic: {info.topic}\n"
                f"  Timestamp: {info.timestamp}\n"
                f"  ID: {info.stream_id}\n"
                f"  Size: {info.size}"
            )
            text = await reader.read_all()
            for line in text.splitlines():
                config_update = json.loads(line)
                if (
                    (_speed := ensure_number(config_update.get("speed")))
                    and session.tts
                    and hasattr(session.tts, "update_speed")
                ):
                    try:
                        session.tts.update_speed(_speed)
                    except Exception:
                        pass

        task = asyncio.create_task(
            async_handle_config_stream(reader, participant_identity)
        )
        _active_tasks.add(task)
        task.add_done_callback(lambda t: _active_tasks.remove(t))

    return handle_text_stream


import random
from multiprocessing import Manager

manager = Manager()
stt_instances = manager.list([hf_stt.STT() for _ in range(3)])


def _get_stt():
    global stt_instances
    return random.choice(stt_instances)


def _get_tts():
    return pipertts.TTS(generation_kwargs={"length_scale": 0.6})


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
    try:

        proc.userdata["stt"] = _get_stt()

        proc.userdata["tts"] = _get_tts()
    except Exception:
        import traceback

        traceback.print_exc()
        raise


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    await ctx.connect()
    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=ctx.proc.userdata["stt"],
        llm=openai.LLM(
            model="gemma3:27b-it-qat",
            api_key="nokey",
            base_url="http://0.0.0.0:12131/v1",
            max_completion_tokens=1024,
            timeout=httpx.Timeout(connect=60.0, read=60.0, write=60.0, pool=5.0),
        ),
        tts=ctx.proc.userdata["tts"],
        turn_detection="vad",
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
        agent=HustAgent(),
        room_output_options=RoomOutputOptions(transcription_enabled=True),
    )
    ctx.room.register_text_stream_handler(
        "custom-agent-text-input", get_text_handler(session)
    )
    ctx.room.register_text_stream_handler("config", get_config_handler(session))


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            job_memory_warn_mb=8000000,
            initialize_process_timeout=3600000,
            num_idle_processes=1,
            worker_type=WorkerType.PUBLISHER,
        )
    )
