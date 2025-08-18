import json
import pyaudio
import torch
import time
import wave
import zmq
from dataclasses import dataclass
from pathlib import Path
from vap.model import VapGPT, VapConfig
import logging 
logger=logging.getLogger(__name__)

"""
* Listen to stereo audio input
* Add audio into a queue/deque/np.array
    * of a certain size (20s)
* Loop that process the audio input with the model
* Send actions/probs throuch zmk to target path

TODO:
* [ ] Audio + tensor update working
* [ ] Model forward
* [ ] ZMK working

Later Todos:
* [ ] torch.compile
"""

NORM_FACTOR: float = 1 / (2 ** 15)


@dataclass
class SDSConfig:
    # audio
    frame_length: float = 0.02  # time (seconds) of each frame of audio
    sample_width: int = 2
    sample_rate: int = 16_000

    # Model
    context: int = 20  # The size of the audio processed by the model
    state_dict: str = str(
        Path(__file__).parent.parent/"example/VAP_3mmz3t0u_50Hz_ad20s_134-epoch9-val_2.56.pt"
    )

    # TurnTaking
    tt_time: float = 0.5

    # ZMK
    port: int = 5578
    topic: str = "tt_probs"


# TODO: dynamic conf
def load_model(state_dict_path):
    model_conf = VapConfig()
    model = VapGPT(model_conf)
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    return model.eval()

class TurnTakingSDS:
    def __init__(
        self,
        conf:SDSConfig,
        probs_file: str = str(Path(__file__).parent/"data/probs.txt"),
    ):
        self.conf = conf
        # logging
        self.probs_txt_file = open(probs_file, "w")
        self.model = load_model(conf.state_dict)
        n_samples = round(conf.context * conf.sample_rate)
        self.x = torch.zeros((1, 2, n_samples))
        self.device = "cpu"
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")
            self.x = self.x.to("cuda")
            self.device = "cuda"
            logger.info("Moved to CUDA")

        # The number of frames to average the turn-shift probabiltites in
        self.tt_frames = round(conf.tt_time * self.model.frame_hz)

    def add_audio_bytes_to_tensor(
        self, audio_bytes: bytes, norm_factor: float = NORM_FACTOR
    ) -> None:

        chunk = torch.frombuffer(audio_bytes, dtype=torch.int16).float() * norm_factor

        # Split stereo audio
        a = chunk[::2]
        b = chunk[1::2]
        chunk_size = a.shape[0]

        # Move values back
        self.x = self.x.roll(-chunk_size, -1)
        self.x[0, 0, -chunk_size:] = a.to(self.device)
        self.x[0, 1, -chunk_size:] = b.to(self.device)

    @torch.no_grad()
    def run(self):
        start_time = time.time()
        try:
            while True:
                # Get new data from stream
                audio_bytes = self.audio_in.get_audio_buffer()
                if len(audio_bytes) == 0:
                    continue

                # update tensor X
                self.add_audio_bytes_to_tensor(audio_bytes)
                a = (self.x[0, 0, -4000:].pow(2).sqrt().max() * 100).long().item()
                b = (self.x[0, 1, -4000:].pow(2).sqrt().max() * 100).long().item()

                # feed through model
                out = self.model.probs(self.x)
                p = out["p_now"][0, -self.tt_frames :, 0].mean().item()
                # p = out["p_future"][0, -self.tt_frames :, 0].mean().item()
                logger.info(f"FURHAT: {round(100*p)} | Audio   A: ", a, "B: ", b)
                # logger.info(f"FURHAT: {round(100*p)}")
                # d = {"now": p}

                cur_time = time.time() - start_time
                self.probs_txt_file.write(f"{cur_time} {p}\n")

        except KeyboardInterrupt:
            logger.info("Abort by user KEYBOARD")
        self.audio_in.stop_stream()
        self.socket.close()
        self.probs_txt_file.close()
        logger.info("Closed socket")

