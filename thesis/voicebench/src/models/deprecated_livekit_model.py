import asyncio
import logging
import datetime 
from livekit.api import AccessToken, VideoGrants 
from livekit import rtc
from livekit.protocol import agent as proto_agent 
import json
import uuid
import base64
import numpy as np 
from .base import VoiceAssistant
from urllib.parse import urlparse, urlunparse
import os
from dotenv import load_dotenv


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveKitAssistant(VoiceAssistant):
    def __init__(self,
                 ws_url: str | None = None,
                 token: str | None = None,
                 api_key: str | None = None,
                 api_secret: str | None = None,
                 room_name: str | None = None,
                 participant_identity: str | None = None,
                 participant_name: str | None = "PythonClientDefaultName",
                 ttl_seconds: int = 3600):
        """
        Initializes the LiveKitAssistant.
        ... (docstring remains the same)
        """
        load_dotenv() 

        self.ws_url = self._initialize_ws_url(ws_url)
        self.token = self._initialize_token(
            token, api_key, api_secret, room_name,
            participant_identity, participant_name, ttl_seconds
        )

        self.room: rtc.Room | None = None
        
        self._event_loop = asyncio.get_event_loop()
        self._pending_stt_requests = {}  

    def _initialize_ws_url(self, ws_url_arg: str | None) -> str:
        """Initializes and validates the WebSocket URL."""
        raw_ws_url = ws_url_arg if ws_url_arg is not None else os.getenv('LIVEKIT_URL')
        if not raw_ws_url:
            raise ValueError(
                "LiveKit WebSocket URL must be provided either as the 'ws_url' argument "
                "or as the LIVEKIT_URL environment variable."
            )

        parsed_url = urlparse(raw_ws_url)
        if parsed_url.scheme == 'http':
            final_ws_url = urlunparse(parsed_url._replace(scheme='ws'))
            logger.info(f"Converted http URL to ws: {final_ws_url}")
        elif parsed_url.scheme == 'https':
            final_ws_url = urlunparse(parsed_url._replace(scheme='wss'))
            logger.info(f"Converted https URL to wss: {final_ws_url}")
        elif parsed_url.scheme in ('ws', 'wss'):
            final_ws_url = raw_ws_url
        else:
            raise ValueError(f"Invalid URL scheme: {parsed_url.scheme}. Must be http, https, ws, or wss.")
        return final_ws_url

    def _initialize_token(self,
                          token_arg: str | None,
                          api_key_arg: str | None,
                          api_secret_arg: str | None,
                          room_name_arg: str | None,
                          participant_identity_arg: str | None,
                          participant_name: str, 
                          ttl_seconds: int         
                          ) -> str:
        """Initializes the access token, either by direct value, env var, or generation."""
        _token = token_arg if token_arg is not None else os.getenv('LIVEKIT_TOKEN')

        if _token:
            logger.info("Using provided token (from argument or LIVEKIT_TOKEN env var).")
            return _token
        else:
            logger.info("Token not directly provided, attempting to generate from API credentials.")
            _api_key = api_key_arg if api_key_arg is not None else os.getenv('LIVEKIT_API_KEY')
            _api_secret = api_secret_arg if api_secret_arg is not None else os.getenv('LIVEKIT_API_SECRET')

            _room_name = room_name_arg
            if _room_name is None:
                _room_name = os.getenv('LIVEKIT_ROOM_NAME')
            if _room_name is None:
                _room_name = str(uuid.uuid4())
                logger.info(f"No room_name provided, generated UUID: {_room_name}")

            _participant_identity = participant_identity_arg
            if _participant_identity is None:
                _participant_identity = os.getenv('LIVEKIT_PARTICIPANT_IDENTITY')
            if _participant_identity is None:
                _participant_identity = f"py-sdk-client-{str(uuid.uuid4())[:8]}"
                logger.info(f"No participant_identity provided, generated: {_participant_identity}")

            if _api_key and _api_secret and _room_name and _participant_identity:
                logger.info(f"Generating token with: room='{_room_name}', identity='{_participant_identity}'")
                return self.generate_access_token(
                    api_key=_api_key,
                    api_secret=_api_secret,
                    room_name=_room_name,
                    participant_identity=_participant_identity,
                    participant_name=participant_name,
                    ttl_seconds=ttl_seconds
                )
            else:
                missing_parts = []
                if not _api_key: missing_parts.append("API key ('api_key' arg or LIVEKIT_API_KEY env var)")
                if not _api_secret: missing_parts.append("API secret ('api_secret' arg or LIVEKIT_API_SECRET env var)")
                if not _room_name: missing_parts.append("Room name ('room_name' arg or LIVEKIT_ROOM_NAME env var)")
                if not _participant_identity: missing_parts.append("Participant identity ('participant_identity' arg or LIVEKIT_PARTICIPANT_IDENTITY env var)")
                
                raise ValueError(
                    "LiveKit access token must be provided (as 'token' argument or LIVEKIT_TOKEN env var), "
                    "or all components for token generation must be available. "
                    f"Missing for token generation: {', '.join(missing_parts)}."
                )

    @staticmethod
    def generate_access_token(api_key: str, api_secret: str, room_name: str, participant_identity: str, participant_name: str | None = None, ttl_seconds: int = 3600) -> str:
        """
        Generates a LiveKit access token.

        Args:
            api_key: Your LiveKit API key.
            api_secret: Your LiveKit API secret.
            room_name: The name of the room the token is for.
            participant_identity: The identity of the participant.
            participant_name: The display name of the participant (optional).
            ttl_seconds: Time-to-live for the token in seconds (default is 1 hour).

        Returns:
            A JWT token string.
        """
        grants = VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True, 
            can_subscribe=True, 
        )
        access_token_builder = AccessToken(api_key, api_secret)
        access_token_builder = access_token_builder.with_identity(participant_identity)
        access_token_builder = access_token_builder.with_grants(grants)
        access_token_builder = access_token_builder.with_ttl(datetime.timedelta(seconds=ttl_seconds))

        if participant_name:
            access_token_builder = access_token_builder.with_name(participant_name)
        return access_token_builder.to_jwt()

    async def connect(self, room_name: str, participant_identity: str) -> bool:
        """
        Connects to a LiveKit room.

        Args:
            room_name: The name of the room to join.
            participant_identity: The identity of this participant.

        Returns:
            True if connection was successful, False otherwise.
        """
        if self.room and self.room.is_connected:
            logger.info("Already connected to a room.")
            return True

        try:
            logger.info(f"Attempting to connect to LiveKit room: {room_name} as {participant_identity}")
            self.room = rtc.Room(loop=self._event_loop)

            @self.room.on("data_received")
            async def on_data_received(payload: bytes, participant: rtc.RemoteParticipant | None, topic: str | None, **kwargs):
                logger.info(f"[LiveKit] Data received on topic '{topic}' from participant '{participant.identity if participant else 'N/A'}'. Raw payload (first 200 bytes): {payload[:200]!r}")
                
                if topic and topic == f"stt_response_{self.room.local_participant.identity if self.room and self.room.local_participant else ''}": 
                    try:
                        data_str = payload.decode('utf-8', errors='replace')
                        logger.info(f"[LiveKit] Decoded payload: {data_str}")
                        response_data = json.loads(data_str)
                        request_id = response_data.get("request_id")
                        text = response_data.get("text", "")
                        error = response_data.get("error")
                        state = response_data.get("state")

                        logger.info(f"[LiveKit] STT response for request_id: {request_id}, text: '{text}', error: {error}, state: {state}")
                        logger.info(f"[LiveKit] Current pending requests: {list(self._pending_stt_requests.keys())}")

                        future = self._pending_stt_requests.get(request_id)
                        if future and not future.done():
                            if error:
                                logger.error(f"[LiveKit] STT request {request_id} failed with error: {error}")
                                future.set_exception(RuntimeError(f"STT agent error: {error}"))
                            elif state == "listening":
                                logger.info(f"[LiveKit] Received 'listening' state for request_id: {request_id}, not completing future yet.")
                                # Do not complete the future, just log
                            else:
                                logger.info(f"[LiveKit] STT request {request_id} completed successfully with text: '{text}'")
                                future.set_result(text)
                        elif future and future.done():
                            logger.warning(f"[LiveKit] Received STT response for already completed request_id: {request_id}")
                        else:
                            logger.warning(f"[LiveKit] Received STT response for unknown request_id: {request_id}")
                        
                        # Always clean up the request from pending list if not 'listening'
                        if request_id in self._pending_stt_requests and state != "listening":
                            del self._pending_stt_requests[request_id]
                            logger.debug(f"[LiveKit] Cleaned up request_id {request_id} from pending requests")
                        
                    except Exception as e:
                        logger.error(f"[LiveKit] Error processing STT response on topic {topic}: {e}")
                        # Try to clean up any pending requests that might be affected
                        for req_id, future in list(self._pending_stt_requests.items()):
                            if not future.done():
                                logger.warning(f"[LiveKit] Setting exception for potentially stuck request {req_id} due to processing error")
                                future.set_exception(Exception(f"STT processing error: {e}"))
                                del self._pending_stt_requests[req_id]
                

            @self.room.on("connected")
            async def on_connected():
                logger.info(f"Successfully connected to room: {self.room.name}") 

            @self.room.on("disconnected")
            async def on_disconnected():
                logger.info(f"Disconnected from room: {self.room.name}") 
                self.room = None 

            @self.room.on("participant_connected")
            def on_participant_connected(participant: rtc.RemoteParticipant):
                logger.info(f"Participant connected: {participant.identity}")

            @self.room.on("participant_disconnected")
            def on_participant_disconnected(participant: rtc.RemoteParticipant):
                logger.info(f"Participant disconnected: {participant.identity}")

            @self.room.on("track_subscribed")
            def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
                logger.info(f"Track subscribed: {track.sid} from participant {participant.identity}")
                
                
                
                

            await self.room.connect(self.ws_url, self.token, rtc.RoomOptions(
                auto_subscribe=True, 
                
            ))
            
            
            
            

            return True
        except rtc.ConnectError as e:
            logger.error(f"Failed to connect to LiveKit room: {e}")
            self.room = None
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred during connection: {e}")
            if self.room and self.room.is_connected:
                await self.room.disconnect()
            self.room = None
            return False

    async def disconnect(self):
        """
        Disconnects from the LiveKit room if connected.
        """
        if self.room and self.room.is_connected:
            logger.info(f"Disconnecting from room: {self.room.name}") 
            
            # Cancel any pending STT requests before disconnecting
            if self._pending_stt_requests:
                logger.info(f"Cancelling {len(self._pending_stt_requests)} pending STT requests before disconnect")
                for request_id, future in self._pending_stt_requests.items():
                    if not future.done():
                        future.set_exception(Exception("Connection being closed"))
                self._pending_stt_requests.clear()
            
            await self.room.disconnect()
            self.room = None
        else:
            logger.info("Not connected to any room.")

    async def publish_audio_track(self, track_name: str = "my-audio-track"):
        """
        Creates and publishes a local audio track.
        This is a placeholder and would typically involve capturing audio from a microphone.
        For simplicity, this example won't actually capture audio.
        """
        if self.room and self.room.is_connected and self.room.local_participant:
            try:
                
                
                
                
                
                logger.info(f"Placeholder for publishing audio track: {track_name}. Real implementation needed.")
                
                
                
            except Exception as e:
                logger.error(f"Failed to publish audio track: {e}")
        else:
            logger.warning("Cannot publish audio track: Not connected to a room or no local participant.")

    async def _async_generate_audio(
        self,
        audio: dict, 
        max_new_tokens=2048, 
        stt_request_topic: str = "stt_request", 
        timeout_seconds: float = 30.0
    ) -> str | None:
        """
        Sends audio data to an agent for Speech-to-Text and returns the transcription.
        This requires the agent to be listening on `stt_request_topic` and
        responding on a topic like "stt_response_YOUR_PARTICIPANT_ID" with the request_id.
        """
        if not self.room or not self.room.is_connected or not self.room.local_participant:
            logger.error("Cannot request STT: Not connected or no local participant.")
            return None

        audio_array = audio.get('array')
        sample_rate = audio.get('sampling_rate')
        channels = audio.get('channels', 1)

        if not isinstance(audio_array, np.ndarray) or not isinstance(sample_rate, int):
            logger.error(f"Invalid audio data format for STT request. Got array type: {type(audio_array)}, sample_rate type: {type(sample_rate)}")
            return None

        request_id = str(uuid.uuid4())
        logger.info(f"Starting STT request with ID: {request_id}")
        
        try:
            future = self._event_loop.create_future()
            self._pending_stt_requests[request_id] = future
            logger.debug(f"Added request {request_id} to pending requests. Total pending: {len(self._pending_stt_requests)}")

            if audio_array.dtype != np.int16:
                if np.issubdtype(audio_array.dtype, np.floating):
                    logger.debug(f"Converting float audio array to int16. Original min: {audio_array.min()}, max: {audio_array.max()}")
                    audio_array = np.clip(audio_array, -1.0, 1.0) 
                    audio_array = (audio_array * 32767).astype(np.int16)
                else:
                    logger.error(f"Unsupported audio array dtype: {audio_array.dtype}. Expected int16 or float.")
                    self._pending_stt_requests.pop(request_id, None) 
                    return None
            
            audio_bytes = audio_array.tobytes()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

            request_payload = {
                "request_id": request_id,
                "audio_base64": audio_base64,
                "sample_rate": sample_rate,
                "channels": channels,
                "encoding": "pcm_s16le" 
            }
            payload_bytes = json.dumps(request_payload).encode('utf-8')
            
            logger.info(f"Sending STT request (ID: {request_id}) to agent on topic '{stt_request_topic}' with audio size: {len(audio_bytes)} bytes")
            await self.room.local_participant.publish_data(
                payload=payload_bytes,
                topic=stt_request_topic,
                reliable=True
            )
            
            logger.info(f"STT request {request_id} sent, waiting for response (timeout: {timeout_seconds}s)...")
            transcribed_text = await asyncio.wait_for(future, timeout=timeout_seconds)
            logger.info(f"STT request {request_id} completed successfully: '{transcribed_text}'")
            return transcribed_text
        except asyncio.TimeoutError:
            logger.error(f"STT request {request_id} timed out after {timeout_seconds}s. Cleaning up...")
            # Clean up the pending request
            if request_id in self._pending_stt_requests:
                del self._pending_stt_requests[request_id]
            logger.debug(f"Remaining pending requests after timeout cleanup: {len(self._pending_stt_requests)}")
            return None
        except Exception as e:
            logger.error(f"Failed to process STT request {request_id}: {e}")
            # Clean up the pending request
            if request_id in self._pending_stt_requests:
                del self._pending_stt_requests[request_id]
            logger.debug(f"Remaining pending requests after error cleanup: {len(self._pending_stt_requests)}")
            return None


    def generate_audio(
        self,
        audio: dict,
        max_new_tokens=2048,
        return_all=False,
        stt_request_topic: str = "stt_request",
        timeout_seconds: float = 30.0
    ) -> str | dict | None:
        """
        Synchronous wrapper for _async_generate_audio.
        
        Args:
            audio: Dictionary with 'array', 'sampling_rate', and optionally 'channels'
            max_new_tokens: Not used in this implementation but kept for interface compatibility
            return_all: If True, return a dict with additional metadata
            stt_request_topic: Topic to send STT requests on
            timeout_seconds: Timeout for the STT request
            
        Returns:
            If return_all=False: transcribed text string or None
            If return_all=True: dict with 'transcription' and other metadata
        """
        result = self._event_loop.run_until_complete(
            self._async_generate_audio(audio, max_new_tokens, stt_request_topic, timeout_seconds)
        )
        
        if return_all:
            return {
                "transcription": result if result is not None else "",
                "success": result is not None,
                "pending_requests": len(self._pending_stt_requests)
            }
        else:
            return result

    async def _async_generate_text(self, text: str, tts_request_topic: str = "custom-agent-text-input") -> str:
        """
        Sends text to an agent in the room to be spoken (Text-to-Speech).
        This method triggers TTS by an agent; it does not return audio data directly.
        The return value from VoiceAssistant.generate_text is usually audio data/path.
        This implementation returns a status string.
        """
        if not self.room or not self.room.is_connected or not self.room.local_participant:
            logger.error("Cannot send text for TTS: Not connected or no local participant.")
            return "Error: Not connected"

        try:
            logger.info(f"Sending text to agent on topic '{tts_request_topic}': {text}")
            payload = text.encode('utf-8')
            await self.room.local_participant.publish_data(payload=payload, topic=tts_request_topic, reliable=True)
            logger.info("Text sent for TTS.")
            return "TTS request sent successfully"
        except Exception as e:
            logger.error(f"Failed to send text for TTS: {e}")
            return f"Error sending TTS request: {e}"


    def generate_text(self, text: str, tts_request_topic: str = "custom-agent-text-input") -> str:
        """Synchronous wrapper for _async_generate_text."""
        return self._event_loop.run_until_complete(
            self._async_generate_text(text, tts_request_topic)
        )

    def cleanup_stale_requests(self, max_age_seconds: float = 60.0):
        """
        Clean up any pending STT requests that are older than max_age_seconds.
        This helps prevent memory leaks from stuck requests.
        """
        current_time = asyncio.get_event_loop().time()
        stale_requests = []
        
        for request_id, future in self._pending_stt_requests.items():
            # Check if future has been waiting too long (this is a simple heuristic)
            if not future.done():
                # We don't have creation time, so we'll use a different approach
                # For now, just check if we have too many pending requests
                if len(self._pending_stt_requests) > 10:  # Arbitrary threshold
                    stale_requests.append(request_id)
        
        for request_id in stale_requests:
            future = self._pending_stt_requests.get(request_id)
            if future and not future.done():
                logger.warning(f"Cleaning up stale request {request_id}")
                future.set_exception(Exception("Request cleaned up due to staleness"))
                del self._pending_stt_requests[request_id]
        
        logger.debug(f"Cleaned up {len(stale_requests)} stale requests. Remaining: {len(self._pending_stt_requests)}")

    def get_pending_requests_count(self) -> int:
        """Return the number of pending STT requests."""
        return len(self._pending_stt_requests)

    def is_healthy(self) -> bool:
        """Check if the assistant is in a healthy state."""
        if not self.room or not self.room.is_connected:
            return False
        
        # If too many pending requests, something might be wrong
        if len(self._pending_stt_requests) > 5:
            logger.warning(f"Many pending requests detected: {len(self._pending_stt_requests)}")
            return False
            
        return True

async def main():
    
    load_dotenv()

    
    
    
    
    
    

    
    
    
    

    LIVEKIT_API_KEY = os.getenv('LIVEKIT_API_KEY')
    LIVEKIT_API_SECRET = os.getenv('LIVEKIT_API_SECRET')
    LIVEKIT_URL = os.getenv('LIVEKIT_URL')

    if not all([LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_URL]):
        logger.error("Missing one or more environment variables: LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_URL")
        return

    
    ROOM_NAME = str(uuid.uuid4()) 
    PARTICIPANT_IDENTITY = "python-client-generated-token"
    PARTICIPANT_NAME = "Python Client"

    
    LIVEKIT_TOKEN = LiveKitAssistant.generate_access_token(LIVEKIT_API_KEY, LIVEKIT_API_SECRET, ROOM_NAME, PARTICIPANT_IDENTITY, PARTICIPANT_NAME)
    logger.info(f"Generated LiveKit Token for room '{ROOM_NAME}': {LIVEKIT_TOKEN}")

    model = LiveKitAssistant(ws_url=LIVEKIT_URL, token=LIVEKIT_TOKEN)

    connected = await model.connect(room_name=ROOM_NAME, participant_identity=PARTICIPANT_IDENTITY)

    if connected:
        logger.info("Connection successful. Model is ready.")
        try:
            
            
            
            
            
            
            
            tts_status = model.generate_text("Hello from Python client, please say this.")
            logger.info(f"TTS request status: {tts_status}")

            
            
            sample_rate = 16000
            duration = 1
            frequency = 440
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            dummy_audio_array = (0.5 * np.sin(2 * np.pi * frequency * t)).astype(np.float32) 
            
            
            transcription = model.generate_audio({'array': dummy_audio_array, 'sampling_rate': sample_rate, 'channels': 1})
            logger.info(f"STT result for dummy audio: {transcription}")
            await asyncio.sleep(5) 
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            await model.disconnect()
    else:
        logger.error("Failed to connect to LiveKit.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application shut down.")