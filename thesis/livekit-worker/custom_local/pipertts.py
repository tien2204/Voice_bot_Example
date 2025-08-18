import livekit.agents
from livekit.agents import tts, utils, APIConnectionError
import livekit
from piper.voice import PiperVoice
from dataclasses import dataclass
import asyncio
from . import logger
import tempfile
import subprocess
import wave
import typing as tp
import io
from cached_path import cached_path
import torch


@dataclass
class _Conf:
    config_file: str
    model_file: str
    piper_path: str
    generation_kwargs: tp.Dict[str, tp.Any]


import json
from piper.voice import Path, PiperConfig, onnxruntime


def load_piper(
    model_path: tp.Union[str, Path],
    config_path: tp.Optional[tp.Union[str, Path]] = None,
    use_cuda: bool = False,
    **kwargs,
) -> "PiperVoice":
    """Load an ONNX model and config."""
    if config_path is None:
        config_path = f"{model_path}.json"

    with open(config_path, "r", encoding="utf-8") as config_file:
        config_dict = json.load(config_file)

    return PiperVoice(
        config=PiperConfig.from_dict({**config_dict, **kwargs}),
        session=onnxruntime.InferenceSession(
            str(model_path),
            sess_options=onnxruntime.SessionOptions(),
            providers=(
                ["CPUExecutionProvider"] if not use_cuda else ["CUDAExecutionProvider"]
            ),
        ),
    )


class TTS(tts.TTS):
    def __init__(
        self,
        config_file=str(
            cached_path(
                f"hf://rhasspy/piper-voices/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx.json"
            )
        ),
        model_file=str(
            cached_path(
                f"hf://rhasspy/piper-voices/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx"
            )
        ),
        piper_path: str = "piper",
        generation_kwargs: tp.Dict[str, tp.Any] = {},
    ):
        self._opts = _Conf(
            config_file=config_file,
            model_file=model_file,
            piper_path=piper_path,
            generation_kwargs=generation_kwargs,
        )
        self._model = load_piper(
            model_path=model_file,
            config_path=config_file,
            use_cuda=torch.cuda.is_available(),
            **self._opts.generation_kwargs,
        )
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=self._model.config.sample_rate,
            num_channels=1,
        )

    def update_speed(self, speed):
        self._opts.generation_kwargs["length_scale"] = round(1 / speed, 1)
        self._model = load_piper(
            model_path=self._opts.model_file,
            config_path=self._opts.config_file,
            use_cuda=torch.cuda.is_available(),
            **self._opts.generation_kwargs,
        )

    def synthesize(
        self,
        text: str,
        *,
        conn_options: livekit.agents.APIConnectOptions | None = None,
    ) -> "ChunkedStream":
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            opts=self._opts,
            model=self._model,
        )


from vi_cleaner.vi_cleaner import ViCleaner
import re

preprocess_text_SPECIAL_CHARACTERS = re.compile(
    r"[^\w\s\.\,\?\!\d]+", flags=re.MULTILINE | re.UNICODE
)


def _preprocess_input_text(text: str):
    text = ViCleaner(text).clean()
    if not isinstance(text, str) or not text:
        return ""
    text = text.strip()
    text = preprocess_text_SPECIAL_CHARACTERS.sub("", text)
    return text


class ChunkedStream(tts.ChunkedStream):
    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        conn_options: livekit.agents.APIConnectOptions,
        opts: _Conf,
        model: tp.Optional[PiperVoice],
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts
        self._model = model

    async def _run(self):
        return await (self._run_piper if self._model else self._run_cli)()

    async def _run_piper(self):
        request_id = utils.shortuuid()
        decoder = utils.codecs.AudioStreamDecoder(
            sample_rate=16000, num_channels=1, format="wav"
        )
        try:

            audiobytes = io.BytesIO()
            text = self.input_text
            text = _preprocess_input_text(text)
            try:
                with wave.open(audiobytes, "w") as fi:
                    self._model.synthesize(text, wav_file=fi, sentence_silence=0.2)
                decoder.push(audiobytes.getvalue())
                decoder.end_input()
            except FileNotFoundError:
                raise
            except subprocess.CalledProcessError as e:
                raise Exception(
                    f"Error running Piper: {e.stderr.decode('utf-8', errors='replace')}"
                )
            emitter = tts.SynthesizedAudioEmitter(
                event_ch=self._event_ch,
                request_id=request_id,
            )
            async for frame in decoder:
                emitter.push(frame)
            emitter.flush()
            logger.info("TTS request ok!")
        except Exception as e:
            import traceback

            logger.warning(traceback.format_exc())
            raise APIConnectionError() from e
        finally:

            await decoder.aclose()

    async def _run_cli(self):
        request_id = utils.shortuuid()
        decoder = utils.codecs.AudioStreamDecoder(
            sample_rate=16000, num_channels=1, format="wav"
        )
        try:

            with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_wav_file:
                output_wav_path = tmp_wav_file.name

                cmd_list = [
                    self._opts.piper_path,
                    "-m",
                    self._opts.model_file,
                    "-f",
                    output_wav_path,
                ]
                if self._opts.config_file:
                    cmd_list.extend(["-c", self._opts.config_file])

                logger.debug(
                    f"Running Piper with args: {cmd_list}. Text: {self.input_text}"
                )
                text = self.input_text
                text = _preprocess_input_text(text)
                try:

                    result = subprocess.run(
                        cmd_list,
                        input=text.encode("utf-8"),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=True,
                        shell=False,
                    )
                    with open(output_wav_path, "rb") as wf:
                        decoder.push(wf.read())
                        decoder.end_input()
                except FileNotFoundError:
                    raise
                except subprocess.CalledProcessError as e:
                    raise Exception(
                        f"Error running Piper: {e.stderr.decode('utf-8', errors='replace')}"
                    )
            emitter = tts.SynthesizedAudioEmitter(
                event_ch=self._event_ch,
                request_id=request_id,
            )
            async for frame in decoder:
                emitter.push(frame)
            emitter.flush()
            logger.info("TTS request ok!")
        except Exception as e:
            import traceback

            logger.warning(traceback.format_exc())
            raise APIConnectionError() from e
        finally:

            await decoder.aclose()
