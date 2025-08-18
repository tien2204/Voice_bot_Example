import asyncio
import os
from dotenv import load_dotenv
from livekit import rtc
from livekit.api import AccessToken, VideoGrants
from typing import List
from contextlib import asynccontextmanager
from .base import VoiceAssistant
import logging
import numpy as np
from datasets import Audio
import io
from pydub import AudioSegment
import torchaudio
from scipy.signal import resample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def require_env(var):
    value = os.getenv(var)
    if not value:
        raise EnvironmentError(f"Missing required environment variable: {var}")
    return value

LIVEKIT_URL = require_env("LIVEKIT_URL")
LIVEKIT_API_KEY = require_env("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = require_env("LIVEKIT_API_SECRET")
ROOM_NAME = os.getenv("LIVEKIT_ROOM", "test_room2")
PARTICIPANT_NAME = os.getenv("PARTICIPANT_NAME", "python-client")

def log_audio_info(audio, context="Audio Info"):
    logger.info(f"{context}: shape={audio['array'].shape}, dtype={audio['array'].dtype}, sampling_rate={audio['sampling_rate']}")

async def stream_datasets_audio_to_room(audio: Audio, source: rtc.AudioSource, frame_duration: float = 1.0):
    logger.info("stream_datasets_audio_to_room called")
    log_audio_info(audio, context="Input audio before processing")

    audio_array = audio["array"]
    original_sample_rate = audio["sampling_rate"]
    target_sample_rate = source.sample_rate
    target_channels = source.num_channels

    if audio_array.dtype != np.int16:
        logger.info("Converting audio to int16 format.")
        audio_array = np.clip(audio_array, -1.0, 1.0)
        audio_array = (audio_array * 32767).astype(np.int16)

    if audio_array.ndim == 1:
        logger.info("Expanding audio array to 2D.")
        audio_array = np.expand_dims(audio_array, axis=1)

    original_channels = audio_array.shape[1]

    if original_sample_rate != target_sample_rate:
        logger.info(f"Resampling from {original_sample_rate} to {target_sample_rate}.")
        num_samples = int(audio_array.shape[0] * target_sample_rate / original_sample_rate)
        audio_array = resample(audio_array, num_samples, axis=0).astype(np.int16)

    if original_channels != target_channels:
        logger.info(f"Adjusting channels from {original_channels} to {target_channels}.")
        if target_channels == 1:
            audio_array = audio_array.mean(axis=1, dtype=np.int16, keepdims=True)
        elif original_channels == 1 and target_channels > 1:
            audio_array = np.tile(audio_array, (1, target_channels))
        else:
            audio_array = audio_array[:, :target_channels] if original_channels > target_channels else \
                          np.pad(audio_array, ((0, 0), (0, target_channels - original_channels)), mode='constant')

    interleaved = audio_array.flatten()
    samples_per_channel = int(target_sample_rate * frame_duration)
    total_samples = samples_per_channel * target_channels

    frames = []
    for i in range(0, len(interleaved), total_samples):
        chunk = interleaved[i:i + total_samples]
        if len(chunk) < total_samples:
            chunk = np.pad(chunk, (0, total_samples - len(chunk)), mode='constant')

        frame = rtc.AudioFrame.create(
            sample_rate=target_sample_rate,
            num_channels=target_channels,
            samples_per_channel=samples_per_channel
        )

        np.copyto(np.frombuffer(frame.data, dtype=np.int16), chunk)
        frames.append(frame)

    logger.info(f"Streaming {len(frames)} audio frames to room.")
    for idx, frame in enumerate(frames):
        logger.debug(f"Capturing frame {idx+1}/{len(frames)}")
        await source.capture_frame(frame)
        await asyncio.sleep(frame_duration)

def concat_wav_bytes(wav_bytes_list: list[bytes], target_sample_rate=16000, target_channels=1, target_sample_width=2) -> bytes:
    logger.info(f"Concatenating {len(wav_bytes_list)} wav byte segments.")
    combined = AudioSegment.silent(duration=0, frame_rate=target_sample_rate)
    for wav_bytes in wav_bytes_list:
        segment: AudioSegment = AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav")
        segment = segment.set_frame_rate(target_sample_rate)\
                         .set_channels(target_channels)\
                         .set_sample_width(target_sample_width)
        combined += segment
    out_io = io.BytesIO()
    combined.export(out_io, format="wav")
    return out_io.getvalue()

def wav_bytes_to_np(wav_bytes: bytes, target_sample_rate=16000) -> dict:
    logger.info(f"Converting wav bytes to numpy array at {target_sample_rate} Hz.")
    waveform, sample_rate = torchaudio.load(io.BytesIO(wav_bytes))
    if waveform.shape[0] > 1:
        logger.info("Averaging multi-channel waveform to mono.")
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        logger.info(f"Resampling from {sample_rate} to {target_sample_rate}.")
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=target_sample_rate)
    return {
        "array": waveform.squeeze(0).numpy(),
        "sampling_rate": target_sample_rate
    }

class LiveKitAssistant(VoiceAssistant):
    def __init__(self):
        self.response_text = []
        self.agent_state = "disconnected"
        self.token = self._create_token()
        self.room = rtc.Room()
        self.source = rtc.AudioSource(16000, 1)
        self._register_event_handlers()
        logger.info("LiveKitAssistant initialized.")

    def _register_event_handlers(self):
        @self.room.on("active_speakers_changed")
        def _(*args, **kwargs):
            logger.info(f"active_speakers_changed: {args}, {kwargs}")
        # Add other handlers as needed

    async def ensure_disconnected(self):
        """Ensure the room is properly disconnected."""
        if self.room and self.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
            logger.info("Disconnecting existing room connection.")
            try:
                await self.room.disconnect()
                logger.info("Successfully disconnected.")
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")
        
        # Always create a fresh room instance
        self.room = rtc.Room()
        self._register_event_handlers()
        logger.info("Fresh room instance created.")

    @asynccontextmanager
    async def connect(self):
        # Ensure clean state before connecting
        await self.ensure_disconnected()
        
        ready = 0
        track = rtc.LocalAudioTrack.create_audio_track("audio", self.source)
        
        @self.room.on("track_subscribed")
        def on_track(remote_track: rtc.Track, pub: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            nonlocal ready
            if remote_track.kind == rtc.TrackKind.KIND_AUDIO:
                if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
                    ready += 1
                    logger.info(f"Subscribed to audio track from AGENT {participant.identity}")
            
            # Publish our track when we see a remote track
            asyncio.get_running_loop().create_task(
                self.room.local_participant.publish_track(track, rtc.TrackPublishOptions(
                    source=rtc.TrackSource.SOURCE_MICROPHONE
                ))
            )
        
        @self.room.on("local_track_subscribed")
        def track_sub(_track: rtc.Track):
            nonlocal ready
            if track.sid == _track.sid:
                ready += 1
        
        logger.info(f"Connecting to LiveKit room: {ROOM_NAME} as {PARTICIPANT_NAME}")
        await self.room.connect(LIVEKIT_URL, self.token)
        
        try:
            # Wait for agent to be ready
            while ready < 2:
                logger.info(f"Waiting for agent readiness, ready={ready}")
                await asyncio.sleep(1)
            
            logger.info("Room connection established and ready.")
            yield self.room
            
        finally:
            logger.info("Cleaning up room connection.")
            await self.ensure_disconnected()

    def generate_audio(self, audio: Audio, max_new_tokens=2048, return_all=False, max_retries=3):
        async def run():
            logger.info("Starting audio streaming and response reception.")
            _, result = await asyncio.gather(
                stream_datasets_audio_to_room(source=self.source, audio=audio),
                self._receive_response(self.room)
            )
            logger.info(f"Received result: {result}")
            return result
        retries = 0
        while retries < max_retries:
            try:
                import nest_asyncio
                nest_asyncio.apply()
                loop = asyncio.get_event_loop()
                async def main_async():
                    async with self.connect():
                        return await run()
                result = loop.run_until_complete(main_async())
                if return_all:
                    return result
                return result["transcription"]
            except Exception as e:
                logger.error(f"Error in generate_audio: {e}")
                retries += 1
        raise RuntimeError("generate_audio failed after maximum retries")

    def _create_token(self):
        logger.info("Creating LiveKit access token.")
        token = AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET).with_identity(
            PARTICIPANT_NAME
        ).with_grants(
            VideoGrants(room_join=True, room=ROOM_NAME)
        ).with_name(
            PARTICIPANT_NAME
        )
        return token.to_jwt()

    async def _receive_response(self, room: rtc.Room):
        logger.info("Waiting for audio response or 'listening' state...")
        audio_done = asyncio.Event()
        listening_state_reached = asyncio.Event()
        final_text = ""
        final_question_text = ""
        @room.on("transcription_received")
        def on_transcription_received(
            segments: List[rtc.TranscriptionSegment],
            participant: rtc.Participant,
            publication: rtc.TrackPublication
        ) -> None:
            nonlocal final_question_text, final_text
            for seg in segments:
                logger.info(f"Transcription segment: {seg}")
                if seg.final:
                    if participant.sid == self.room.local_participant.sid:
                        final_question_text = seg.text
                    elif seg.text.strip():
                        final_text = seg.text
                        listening_state_reached.set()
        try:
            done, pending = await asyncio.wait(
                [audio_done.wait(), listening_state_reached.wait()],
                return_when=asyncio.FIRST_COMPLETED,
                timeout=300
            )
            if not done and pending:
                logger.warning("Timeout waiting for audio stream or 'listening' state.")
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
            logger.info(f"Final transcription: {final_text}, Question transcription: {final_question_text}")
            return {
                "transcription": final_text,
                "question_transcription": final_question_text,
            }
        except asyncio.TimeoutError:
            logger.error("Overall timeout during reception, this should be caught by asyncio.wait.")
        finally:
            logger.info("_receive_response finished.")

if __name__ == "__main__":
    assistant = LiveKitAssistant()
    logger.info("Starting voice assistant interaction using a WAV file...")
    audio_loader = Audio(sampling_rate=None)
    audio_data = audio_loader.decode_example({
        "path": "/home/rejk/Documents/temppp/livekittest/voicebench/ref.wav",
        "bytes": None
    })
    log_audio_info(audio_data, context="Loaded audio for main")
    logger.info(assistant.generate_audio(audio_data, return_all=True))
    logger.info("Voice assistant interaction finished (datasets.Audio input).")
