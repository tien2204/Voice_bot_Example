from __future__ import annotations

import gc
import math
import os

import torch
import torchaudio
import wandb
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from ema_pytorch import EMA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

from f5_tts.model import CFM
from f5_tts.model.dataset import DynamicBatchSampler, collate_fn
from f5_tts.model.utils import default, exists
from f5_tts.model import alignment_utils
from f5_tts.model.alignment_utils import (
    text_to_phonemes,
    phoneme_to_indices,
    monotonic_alignment_search,
    get_durations_from_alignment,
)


import re

import math
import logging


class DurationWeightScheduler:
    def __init__(
        self,
        total_epochs,
        initial_weight=1.5,
        min_weight=0.1,
        ema_alpha=0.05,
        slope=8.0,
        center=0.75,
        decay_rate=3.0,
        decay_start_frac=0.1,
        max_delta=0.01,
        logger=None,
    ):
        self.total_epochs = total_epochs
        self.initial_weight = initial_weight
        self.min_weight = min_weight
        self.ema_alpha = ema_alpha
        self.slope = slope
        self.center = center
        self.decay_rate = decay_rate
        self.decay_start_epoch = max(1, int(total_epochs * decay_start_frac))
        self.max_delta = max_delta
        self.ema_cov = 0.0
        self.ema_diag = 0.0
        self.prev_weight = initial_weight
        self.logger = logger or logging.getLogger(__name__)
        self.phase = 1

    def step(self, update, coverage, diagonal, epoch, steps_per_epoch):

        self.ema_cov = self.ema_alpha * coverage + (1 - self.ema_alpha) * self.ema_cov
        self.ema_diag = self.ema_alpha * diagonal + (1 - self.ema_alpha) * self.ema_diag

        quality = 0.4 * self.ema_cov + 0.6 * self.ema_diag
        quality = max(0.0, min(1.0, quality))

        if self.phase == 1:
            return self.initial_weight

        sig = 1.0 / (1.0 + math.exp(-self.slope * (quality - self.center)))
        target = self.min_weight + (1 - sig) * (self.initial_weight - self.min_weight)

        if epoch > self.decay_start_epoch:
            prog = (epoch - self.decay_start_epoch) / (
                self.total_epochs - self.decay_start_epoch
            )
            decay = math.exp(-self.decay_rate * prog)
            target = self.min_weight + (target - self.min_weight) * decay

        delta = target - self.prev_weight
        delta = max(-self.max_delta, min(delta, self.max_delta))
        weight = self.prev_weight + delta
        self.prev_weight = weight

        if update % steps_per_epoch < 1:
            self.logger.info(
                f"[Epoch {epoch}] dur_weight={weight:.4f} "
                f"cov={self.ema_cov:.4f} diag={self.ema_diag:.4f}"
            )

        return weight


class Trainer:
    def __init__(
        self,
        model: CFM,
        epochs,
        learning_rate,
        weight_decay=0.1,
        num_warmup_updates=20000,
        save_per_updates=1000,
        keep_last_n_checkpoints: int = -1,
        checkpoint_path=None,
        batch_size_per_gpu=32,
        batch_size_type: str = "sample",
        max_samples=32,
        grad_accumulation_steps=1,
        max_grad_norm=1.0,
        noise_scheduler: str | None = None,
        duration_predictor: torch.nn.Module | None = None,
        logger: str | None = "wandb",
        wandb_project="test_f5-tts",
        wandb_run_name="test_run",
        wandb_resume_id: str = None,
        log_samples: bool = False,
        last_per_updates=None,
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        bnb_optimizer: bool = False,
        mel_spec_type: str = "vocos",
        is_local_vocoder: bool = False,
        local_vocoder_path: str = "",
        model_cfg_dict: dict = dict(),
        duration_loss_weight: float = 0.1,
        duration_focus_updates: int = 12000,
        duration_focus_weight: float = 1.5,
        ref_texts=None,
        ref_audio_paths=None,
        ref_sample_text_prompts=None,
    ):

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        if logger == "wandb" and not wandb.api.api_key:
            logger = None
        self.log_samples = log_samples

        self.accelerator = Accelerator(
            log_with=logger if logger == "wandb" else None,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=grad_accumulation_steps,
            **accelerate_kwargs,
        )

        self.logger = logger
        if self.logger == "wandb":
            if exists(wandb_resume_id):
                init_kwargs = {
                    "wandb": {
                        "resume": "allow",
                        "name": wandb_run_name,
                        "id": wandb_resume_id,
                    }
                }
            else:
                init_kwargs = {"wandb": {"resume": "allow", "name": wandb_run_name}}

            if not model_cfg_dict:
                model_cfg_dict = {
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "num_warmup_updates": num_warmup_updates,
                    "batch_size_per_gpu": batch_size_per_gpu,
                    "batch_size_type": batch_size_type,
                    "max_samples": max_samples,
                    "grad_accumulation_steps": grad_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "noise_scheduler": noise_scheduler,
                }
            model_cfg_dict["gpus"] = self.accelerator.num_processes
            self.accelerator.init_trackers(
                project_name=wandb_project,
                init_kwargs=init_kwargs,
                config=model_cfg_dict,
            )

        elif self.logger == "tensorboard":
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=f"runs/{wandb_run_name}")

        self.model = model

        if self.is_main:
            self.ema_model = EMA(model, include_online_model=False, **ema_kwargs)
            self.ema_model.to(self.accelerator.device)

            print(f"Using logger: {logger}")
            if grad_accumulation_steps > 1:
                print(
                    "Gradient accumulation checkpointing with per_updates now, old logic per_steps used with before f992c4e"
                )

        self.phoneme_map = {}

        self.epochs = epochs
        self.num_warmup_updates = num_warmup_updates
        self.save_per_updates = save_per_updates
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        self.last_per_updates = default(last_per_updates, save_per_updates)
        self.checkpoint_path = default(checkpoint_path, "ckpts/test_f5-tts")

        self.batch_size_per_gpu = batch_size_per_gpu
        self.batch_size_type = batch_size_type
        self.max_samples = max_samples
        self.grad_accumulation_steps = grad_accumulation_steps
        self.max_grad_norm = max_grad_norm

        self.vocoder_name = mel_spec_type
        self.is_local_vocoder = is_local_vocoder
        self.local_vocoder_path = local_vocoder_path

        self.noise_scheduler = noise_scheduler

        self.duration_predictor = duration_predictor
        self.base_duration_loss_weight = duration_loss_weight

        self.duration_focus_updates = duration_focus_updates
        self.duration_focus_weight = duration_focus_weight
        self.duration_focus_phase = self.duration_focus_updates > 0

        self.alignment_manager = alignment_utils.AlignmentMethodManager()
        self.phase2_start_update = None

        self.dur_weight_scheduler = DurationWeightScheduler(
            total_epochs=self.epochs,
            initial_weight=self.duration_focus_weight,
            min_weight=self.base_duration_loss_weight,
            logger=logging.getLogger("durweight") if self.is_main else None,
        )

        self.dur_weight_scheduler.phase = 1

        self.active_duration_loss_weight = (
            self.duration_focus_weight
            if self.duration_focus_phase
            else self.base_duration_loss_weight
        )

        if self.duration_predictor is not None:

            setattr(self.model, "duration_predictor", self.duration_predictor)
            total_params = sum(
                p.numel()
                for p in self.duration_predictor.parameters()
                if p.requires_grad
            )
            print(
                f"Total number of trainable parameters in Duration Predictor: {total_params}"
            )

            self.duration_predictor_align = type(self.duration_predictor)(
                self.duration_predictor.text_embed.num_embeddings - 1,
                self.duration_predictor.in_channels,
                self.duration_predictor.filter_channels,
                self.duration_predictor.kernel_size,
                self.duration_predictor.p_dropout,
                self.duration_predictor.gin_channels,
            ).to(self.accelerator.device)

            self.duration_predictor_pred = type(self.duration_predictor)(
                self.duration_predictor.text_embed.num_embeddings - 1,
                self.duration_predictor.in_channels,
                self.duration_predictor.filter_channels,
                self.duration_predictor.kernel_size,
                self.duration_predictor.p_dropout,
                self.duration_predictor.gin_channels,
            ).to(self.accelerator.device)

            self.duration_predictor_align.load_state_dict(
                self.duration_predictor.state_dict()
            )
            self.duration_predictor_pred.load_state_dict(
                self.duration_predictor.state_dict()
            )

            self.duration_predictor_align.eval()
            for param in self.duration_predictor_align.parameters():
                param.requires_grad = False

            self.duration_predictor_pred.train()

            print(
                "Created separate duration predictor instances for alignment and prediction"
            )

            duration_params = list(self.duration_predictor.parameters())

            other_params = [
                p
                for p in self.model.parameters()
                if not any(p is dp for dp in duration_params)
            ]

            print(
                f"Duration predictor params: {len(duration_params)}, Other params: {len(other_params)}"
            )

            if self.duration_focus_phase:
                for param in other_params:
                    param.requires_grad = False
                print(
                    f"Duration focus phase active: Main model frozen for {self.duration_focus_updates} updates"
                )
                print(
                    f"Duration loss weight during focus: {self.active_duration_loss_weight}"
                )

            param_groups = [
                {
                    "params": duration_params,
                    "lr": learning_rate * 1.0,
                    "weight_decay": 0.0003,
                },
                {
                    "params": other_params,
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                },
            ]
        else:

            param_groups = self.model.parameters()

        if bnb_optimizer:
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.AdamW8bit(
                param_groups,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.98),
                eps=1e-8,
            )
        else:
            self.optimizer = AdamW(
                param_groups,
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.98),
                eps=1e-8,
            )

        self.model, self.optimizer = self.accelerator.prepare(
            self.model, self.optimizer
        )

        self.ref_texts = ref_texts or []
        self.ref_audio_paths = ref_audio_paths or []
        self.ref_sample_text_prompts = ref_sample_text_prompts or []

        max_len = max(
            len(self.ref_texts),
            len(self.ref_audio_paths),
            len(self.ref_sample_text_prompts),
        )
        if max_len > 0:
            if len(self.ref_texts) > 0 and len(self.ref_texts) != max_len:
                print(
                    f"Warning: ref_texts length ({len(self.ref_texts)}) does not match other reference lists ({max_len}). Using available items only."
                )
            if len(self.ref_audio_paths) > 0 and len(self.ref_audio_paths) != max_len:
                print(
                    f"Warning: ref_audio_paths length ({len(self.ref_audio_paths)}) does not match other reference lists ({max_len}). Using available items only."
                )
            if (
                len(self.ref_sample_text_prompts) > 0
                and len(self.ref_sample_text_prompts) != max_len
            ):
                print(
                    f"Warning: ref_sample_text_prompts length ({len(self.ref_sample_text_prompts)}) does not match other reference lists ({max_len}). Using available items only."
                )

        self.ref_mels = []
        if self.is_main and self.ref_audio_paths and log_samples:
            try:
                from f5_tts.model.modules import MelSpec

                target_sample_rate = 24000
                if hasattr(self.model, "mel_spec") and hasattr(
                    self.model.mel_spec, "target_sample_rate"
                ):
                    target_sample_rate = self.model.mel_spec.target_sample_rate

                mel_spec_kwargs = {
                    "n_fft": 1024,
                    "hop_length": 256,
                    "win_length": 1024,
                    "n_mel_channels": (
                        n_mel_channels if "n_mel_channels" in locals() else 100
                    ),
                    "target_sample_rate": target_sample_rate,
                    "mel_spec_type": mel_spec_type,
                }

                mel_spec = (
                    self.model.mel_spec
                    if hasattr(self.model, "mel_spec")
                    else MelSpec(**mel_spec_kwargs)
                )

                print(
                    f"Loading reference audios with target sample rate: {target_sample_rate}"
                )
                for audio_path in self.ref_audio_paths:
                    if os.path.exists(audio_path):

                        print(f"Loading reference audio: {audio_path}")
                        waveform, sr = torchaudio.load(audio_path)
                        print(f"  Original sample rate: {sr}")

                        if sr != target_sample_rate:
                            print(f"  Resampling from {sr} to {target_sample_rate}")
                            waveform = torchaudio.functional.resample(
                                waveform, sr, target_sample_rate
                            )

                        mel = mel_spec(waveform.to(self.accelerator.device)).cpu()
                        self.ref_mels.append(mel)
                        print(
                            f"  Successfully loaded reference audio. Mel shape: {mel.shape}"
                        )

                        idx = len(self.ref_mels) - 1
                        if idx < len(self.ref_texts):
                            print(f"  Reference text: {self.ref_texts[idx]}")
                        if idx < len(self.ref_sample_text_prompts):
                            print(
                                f"  Sample text prompt: {self.ref_sample_text_prompts[idx]}"
                            )
                    else:
                        print(f"Warning: Reference audio file not found: {audio_path}")
            except Exception as e:
                print(f"Error loading reference audio files: {e}")
                import traceback

                traceback.print_exc()

        self.stored_mel_lengths = None
        self.stored_text_lengths = None

    def generate_reference_samples(
        self, global_update, vocoder, nfe_step, cfg_strength, sway_sampling_coef
    ):
        """Generate samples based on reference audio but with new prompt text."""
        if not self.is_main or not self.log_samples or not self.ref_mels:
            return

        log_samples_path = f"{self.checkpoint_path}/ref_samples"
        os.makedirs(log_samples_path, exist_ok=True)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        model_for_sampling = (
            self.ema_model.ema_model if hasattr(self, "ema_model") else unwrapped_model
        )
        target_sample_rate = unwrapped_model.mel_spec.target_sample_rate

        model_for_sampling.eval()
        if hasattr(model_for_sampling, "clear_cache"):
            model_for_sampling.clear_cache()

        with torch.inference_mode():
            for idx, ref_mel in enumerate(self.ref_mels):
                try:

                    ref_text = self.ref_texts[idx] if idx < len(self.ref_texts) else ""
                    prompt_text = (
                        self.ref_sample_text_prompts[idx]
                        if idx < len(self.ref_sample_text_prompts)
                        else "Default test reference."
                    )

                    txt_path = (
                        f"{log_samples_path}/update_{global_update}_ref{idx}_prompt.txt"
                    )
                    with open(txt_path, "w") as f:
                        f.write(prompt_text)

                    ref_mel = ref_mel.to(self.accelerator.device)
                    ref_mel_for_sample = ref_mel.permute(0, 2, 1)

                    if self.vocoder_name == "vocos":
                        ref_audio = vocoder.decode(ref_mel).cpu()
                    elif self.vocoder_name == "bigvgan":
                        ref_audio = vocoder(ref_mel).squeeze(0).cpu()

                    if ref_audio.ndim == 1:
                        ref_audio = ref_audio.unsqueeze(0)
                    elif ref_audio.ndim == 3 and ref_audio.shape[1] == 1:
                        ref_audio = ref_audio.squeeze(1)

                    ref_path = (
                        f"{log_samples_path}/update_{global_update}_ref{idx}_source.wav"
                    )
                    torchaudio.save(ref_path, ref_audio, target_sample_rate)

                    if hasattr(model_for_sampling, "clear_cache"):
                        model_for_sampling.clear_cache()

                    print(f"Generating audio from prompt text: {prompt_text[:50]}...")

                    chars_per_frame = 0.33
                    min_frames = 100

                    char_count = len(prompt_text)
                    estimated_frames = max(
                        int(char_count / chars_per_frame), min_frames
                    )

                    print(
                        f"Text length: {char_count} chars, estimated duration: {estimated_frames} frames"
                    )
                    print(f"Using ref_mel shape: {ref_mel_for_sample.shape}")

                    ref_audio_len = ref_mel_for_sample.shape[1]

                    generated_full, _ = model_for_sampling.sample(
                        cond=ref_mel_for_sample,
                        text=[prompt_text],
                        duration=ref_audio_len + estimated_frames,
                        steps=nfe_step,
                        cfg_strength=cfg_strength,
                        sway_sampling_coef=sway_sampling_coef,
                    )

                    prompt_mel = generated_full[:, ref_audio_len:, :]

                    print(f"Generated mel shape: {prompt_mel.shape}")

                    if torch.isnan(prompt_mel).any() or torch.isinf(prompt_mel).any():
                        print("ERROR: NaN/Inf detected in generated mel")
                    else:

                        prompt_mel = prompt_mel.to(torch.float32)
                        prompt_mel_for_vocoder = prompt_mel.permute(0, 2, 1).to(
                            self.accelerator.device
                        )

                        if self.vocoder_name == "vocos":
                            prompt_audio = vocoder.decode(prompt_mel_for_vocoder).cpu()
                        elif self.vocoder_name == "bigvgan":
                            prompt_audio = (
                                vocoder(prompt_mel_for_vocoder).squeeze(0).cpu()
                            )

                        if prompt_audio.ndim == 1:
                            prompt_audio = prompt_audio.unsqueeze(0)
                        elif prompt_audio.ndim == 3 and prompt_audio.shape[1] == 1:
                            prompt_audio = prompt_audio.squeeze(1)

                        audio_len = prompt_audio.shape[1]
                        audio_min = prompt_audio.min().item()
                        audio_max = prompt_audio.max().item()
                        print(
                            f"Generated audio length: {audio_len}, range: [{audio_min:.3f}, {audio_max:.3f}]"
                        )

                        gen_path = f"{log_samples_path}/update_{global_update}_ref{idx}_gen_audio.wav"
                        torchaudio.save(gen_path, prompt_audio, target_sample_rate)
                        print(f"Saved generated audio from prompt: {gen_path}")

                    print(
                        f"Completed reference sample generation for update {global_update}"
                    )

                except Exception as e:
                    print(f"Error processing reference sample {idx}: {e}")
                    import traceback

                    traceback.print_exc()

        model_for_sampling.train()
        if hasattr(model_for_sampling, "clear_cache"):
            model_for_sampling.clear_cache()

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, update, last=False):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            checkpoint = dict(
                model_state_dict=self.accelerator.unwrap_model(self.model).state_dict(),
                optimizer_state_dict=self.accelerator.unwrap_model(
                    self.optimizer
                ).state_dict(),
                ema_model_state_dict=self.ema_model.state_dict(),
                scheduler_state_dict=self.scheduler.state_dict(),
                update=update,
            )

            if hasattr(self, "base_duration_loss_weight"):
                checkpoint["base_duration_loss_weight"] = self.base_duration_loss_weight
            if hasattr(self, "active_duration_loss_weight"):
                checkpoint["active_duration_loss_weight"] = (
                    self.active_duration_loss_weight
                )
            if hasattr(self, "duration_focus_phase"):
                checkpoint["duration_focus_phase"] = self.duration_focus_phase

            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)

            if (
                hasattr(self, "duration_predictor")
                and self.duration_predictor is not None
            ):
                if (
                    hasattr(self, "duration_focus_phase")
                    and not self.duration_focus_phase
                    and hasattr(self, "duration_focus_updates")
                    and update == self.duration_focus_updates
                ):

                    duration_checkpoint = {
                        "duration_predictor": self.accelerator.unwrap_model(
                            self.duration_predictor
                        ).state_dict(),
                        "update": update,
                        "duration_loss_weight": self.active_duration_loss_weight,
                    }
                    self.accelerator.save(
                        duration_checkpoint,
                        f"{self.checkpoint_path}/duration_predictor_{update}.pt",
                    )
                    print(
                        f"Saved separate duration predictor checkpoint at end of focus phase (update {update})"
                    )

            if last:
                self.accelerator.save(
                    checkpoint, f"{self.checkpoint_path}/model_last.pt"
                )
                print(f"Saved last checkpoint at update {update}")
            else:
                if self.keep_last_n_checkpoints == 0:
                    return
                self.accelerator.save(
                    checkpoint, f"{self.checkpoint_path}/model_{update}.pt"
                )
                if self.keep_last_n_checkpoints > 0:
                    try:

                        checkpoints = [
                            f
                            for f in os.listdir(self.checkpoint_path)
                            if f.startswith("model_")
                            and not f.startswith("pretrained_")
                            and f.endswith(".pt")
                            and f != "model_last.pt"
                        ]

                        numeric_checkpoints = []
                        for ckpt in checkpoints:
                            try:

                                num_part = ckpt.split("_")[1].split(".")[0]
                                int(num_part)
                                numeric_checkpoints.append(ckpt)
                            except (ValueError, IndexError):

                                print(
                                    f"Skipping non-numeric checkpoint during cleanup: {ckpt}"
                                )

                        numeric_checkpoints.sort(
                            key=lambda x: int(x.split("_")[1].split(".")[0])
                        )

                        while len(numeric_checkpoints) > self.keep_last_n_checkpoints:
                            oldest_checkpoint = numeric_checkpoints.pop(0)
                            os.remove(
                                os.path.join(self.checkpoint_path, oldest_checkpoint)
                            )
                            print(f"Removed old checkpoint: {oldest_checkpoint}")

                    except Exception as e:
                        print(f"Warning: Error during checkpoint cleanup: {e}")

    def load_checkpoint(self):
        """Loads the most recent checkpoint from the checkpoint_path."""

        latest_checkpoint_name = None
        checkpoint_path_to_load = None
        if exists(self.checkpoint_path) and os.path.isdir(self.checkpoint_path):

            last_ckpt_path = os.path.join(self.checkpoint_path, "model_last.pt")
            if os.path.exists(last_ckpt_path):
                latest_checkpoint_name = "model_last.pt"
                self.accelerator.print(
                    f"Found last checkpoint: {latest_checkpoint_name}"
                )
            else:

                all_checkpoints = [
                    f
                    for f in os.listdir(self.checkpoint_path)
                    if (f.startswith("model_") or f.startswith("pretrained_"))
                    and f.endswith((".pt", ".safetensors"))
                ]
                training_checkpoints = [
                    f
                    for f in all_checkpoints
                    if f.startswith("model_") and f != "model_last.pt"
                ]
                pretrained_checkpoints = [
                    f for f in all_checkpoints if f.startswith("pretrained_")
                ]

                latest_update = -1
                latest_training_ckpt = None
                if training_checkpoints:
                    try:
                        for ckpt_name in training_checkpoints:
                            match = re.search(r"model_(\d+)", ckpt_name)
                            if match:
                                update_num = int(match.group(1))
                                if update_num > latest_update:
                                    latest_update = update_num
                                    latest_training_ckpt = ckpt_name
                        if latest_training_ckpt:
                            latest_checkpoint_name = latest_training_ckpt
                            self.accelerator.print(
                                f"Found latest training checkpoint: {latest_checkpoint_name}"
                            )
                    except Exception as e:
                        self.accelerator.print(
                            f"Warning: Failed to sort training checkpoints by update number: {e}. Using last alphabetically."
                        )
                        latest_checkpoint_name = (
                            sorted(training_checkpoints)[-1]
                            if training_checkpoints
                            else None
                        )
                elif pretrained_checkpoints:
                    latest_checkpoint_name = sorted(pretrained_checkpoints)[0]
                    self.accelerator.print(
                        f"No training checkpoints found. Using pretrained checkpoint: {latest_checkpoint_name}"
                    )
                else:
                    self.accelerator.print(
                        "No suitable checkpoints (last, training, or pretrained) found."
                    )

            if latest_checkpoint_name:
                checkpoint_path_to_load = os.path.join(
                    self.checkpoint_path, latest_checkpoint_name
                )

        if checkpoint_path_to_load is None or not os.path.exists(
            checkpoint_path_to_load
        ):
            self.accelerator.print("No valid checkpoint found. Starting from scratch.")
            return 0

        self.accelerator.print(f"Loading checkpoint: {checkpoint_path_to_load}")
        checkpoint = None
        try:
            if checkpoint_path_to_load.endswith(".safetensors"):
                from safetensors.torch import load_file

                loaded_data = load_file(checkpoint_path_to_load, device="cpu")
                checkpoint = {"state_dict_loaded_from_safetensors": loaded_data}
                self.accelerator.print("Loaded state_dict from .safetensors file.")
            elif checkpoint_path_to_load.endswith(".pt"):
                self.accelerator.print(
                    "Loading .pt file with weights_only=False. Ensure checkpoint source is trusted."
                )
                checkpoint = torch.load(
                    checkpoint_path_to_load, map_location="cpu", weights_only=False
                )
                self.accelerator.print("Loaded checkpoint from .pt file.")
            else:
                raise ValueError(
                    f"Unsupported checkpoint file extension: {checkpoint_path_to_load}"
                )
            if not isinstance(checkpoint, dict):
                raise TypeError("Loaded checkpoint is not a dictionary.")
        except Exception as e:
            self.accelerator.print(
                f"Error loading checkpoint file: {e}. Starting from scratch."
            )
            return 0

        model_sd_raw = None
        loaded_from_key = None
        is_ema_source = False

        search_keys = [
            "model_state_dict",
            "ema_model_state_dict",
            "state_dict",
            "model",
            "state_dict_loaded_from_safetensors",
        ]
        for key in search_keys:
            if key in checkpoint:
                potential_sd = checkpoint[key]
                if isinstance(potential_sd, dict) and potential_sd:
                    model_sd_raw = potential_sd
                    loaded_from_key = key
                    if key == "ema_model_state_dict":
                        is_ema_source = True
                    self.accelerator.print(
                        f"Found model weights under key: '{loaded_from_key}' (Is EMA source: {is_ema_source})"
                    )
                    break
                elif not isinstance(potential_sd, dict):
                    self.accelerator.print(
                        f"Warning: Key '{key}' found but value is not a dict (type: {type(potential_sd)}). Skipping."
                    )
                elif not potential_sd:
                    self.accelerator.print(
                        f"Warning: Key '{key}' found but dictionary is empty. Skipping."
                    )

        if model_sd_raw is None:
            self.accelerator.print(
                f"ERROR: Could not find usable model state_dict in checkpoint: {checkpoint_path_to_load}"
            )
            self.accelerator.print(
                f"Available top-level keys: {list(checkpoint.keys())}"
            )
            self.accelerator.print("Starting from scratch as state_dict was not found.")
            return 0

        model_sd_cleaned = {}
        prefixes_to_strip = ["module.", "model.", "_orig_mod."]
        if is_ema_source:
            ema_prefix = "ema_model."
            if any(k.startswith(ema_prefix) for k in model_sd_raw):
                prefixes_to_strip.insert(0, ema_prefix)

        used_prefix = None
        first_key = next(iter(model_sd_raw.keys()), None)
        if first_key:
            for prefix in prefixes_to_strip:
                if first_key.startswith(prefix):
                    prefix_count = sum(1 for k in model_sd_raw if k.startswith(prefix))
                    if prefix_count >= 0.8 * len(model_sd_raw):
                        used_prefix = prefix
                        break

        ignore_keys = {"initted", "step"}
        if used_prefix:
            self.accelerator.print(
                f"Stripping prefix '{used_prefix}' from state_dict keys."
            )
            prefix_len = len(used_prefix)
            for k, v in model_sd_raw.items():
                final_key = k[prefix_len:] if k.startswith(used_prefix) else k
                if final_key not in ignore_keys:
                    model_sd_cleaned[final_key] = v
                else:
                    self.accelerator.print(f"Ignoring metadata key while cleaning: {k}")
        else:
            self.accelerator.print(
                "No common prefix found or stripping not applicable."
            )
            for k, v in model_sd_raw.items():
                if k not in ignore_keys:
                    model_sd_cleaned[k] = v
                else:
                    self.accelerator.print(f"Ignoring metadata key: {k}")

        if not model_sd_cleaned:
            self.accelerator.print(
                "ERROR: State dictionary became empty after cleaning. Check original checkpoint."
            )
            return 0

        load_successful = False
        try:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            incompatible_keys = unwrapped_model.load_state_dict(
                model_sd_cleaned, strict=False
            )
            self.accelerator.print("Successfully loaded model weights into main model.")
            if incompatible_keys.missing_keys:
                self.accelerator.print(
                    f"Note: Missing keys when loading state_dict (expected if model structure changed): {incompatible_keys.missing_keys}"
                )
            if incompatible_keys.unexpected_keys:
                self.accelerator.print(
                    f"Note: Unexpected keys when loading state_dict (expected if checkpoint has extra keys): {incompatible_keys.unexpected_keys}"
                )
            load_successful = True
        except Exception as e:
            self.accelerator.print(f"ERROR loading state_dict into model: {e}")

        if not load_successful:
            self.accelerator.print(
                "Weights could not be loaded. Starting from scratch."
            )
            del checkpoint
            del model_sd_raw
            del model_sd_cleaned
            if "loaded_data" in locals():
                del loaded_data
            self.accelerator.wait_for_everyone()
            if self.accelerator.device.type == "cuda":
                torch.cuda.empty_cache()
            elif self.accelerator.device.type == "xpu":
                torch.xpu.empty_cache()
            gc.collect()
            return 0

        is_resuming_full_state = (
            "optimizer_state_dict" in checkpoint and "update" in checkpoint
        )

        start_update = 0
        if is_resuming_full_state:
            self.accelerator.print(
                "Attempting to load full training state (optimizer, scheduler, EMA, step)..."
            )
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.accelerator.print("Optimizer state loaded.")
            except Exception as e:
                self.accelerator.print(
                    f"Warning: Failed to load optimizer state: {e}. Optimizer will start fresh."
                )

            if (
                hasattr(self, "scheduler")
                and self.scheduler
                and "scheduler_state_dict" in checkpoint
            ):
                try:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    self.accelerator.print("Scheduler state loaded.")
                except Exception as e:
                    self.accelerator.print(
                        f"Warning: Failed to load scheduler state: {e}."
                    )

            if (
                self.is_main
                and hasattr(self, "ema_model")
                and "ema_model_state_dict" in checkpoint
            ):
                try:
                    ema_sd_to_load = checkpoint["ema_model_state_dict"]
                    ema_model_unwrapped = (
                        self.ema_model.module
                        if hasattr(self.ema_model, "module")
                        else self.ema_model
                    )
                    incompatible_ema_keys = ema_model_unwrapped.load_state_dict(
                        ema_sd_to_load, strict=False
                    )
                    self.accelerator.print("Trainer EMA state loaded.")
                    if incompatible_ema_keys.missing_keys:
                        self.accelerator.print(
                            f"Note: Missing EMA keys: {incompatible_ema_keys.missing_keys}"
                        )
                    if incompatible_ema_keys.unexpected_keys:
                        self.accelerator.print(
                            f"Note: Unexpected EMA keys: {incompatible_ema_keys.unexpected_keys}"
                        )
                except Exception as e:
                    self.accelerator.print(
                        f"Warning: Failed to load EMA state into trainer: {e}. EMA will start fresh."
                    )
            elif self.is_main and hasattr(self, "ema_model"):
                self.accelerator.print(
                    "EMA state not found in checkpoint for resume. EMA will start fresh."
                )

            if (
                hasattr(self, "base_duration_loss_weight")
                and "base_duration_loss_weight" in checkpoint
            ):
                self.base_duration_loss_weight = checkpoint["base_duration_loss_weight"]
                self.accelerator.print(
                    f"Loaded base duration loss weight: {self.base_duration_loss_weight}"
                )

            if (
                hasattr(self, "active_duration_loss_weight")
                and "active_duration_loss_weight" in checkpoint
            ):
                self.active_duration_loss_weight = checkpoint[
                    "active_duration_loss_weight"
                ]
                self.accelerator.print(
                    f"Loaded active duration loss weight: {self.active_duration_loss_weight}"
                )

            if (
                hasattr(self, "duration_focus_phase")
                and "duration_focus_phase" in checkpoint
            ):
                self.duration_focus_phase = checkpoint["duration_focus_phase"]
                self.accelerator.print(
                    f"Loaded duration focus phase state: {self.duration_focus_phase}"
                )

            start_update = checkpoint.get("update", checkpoint.get("step", -1))
            if start_update == -1:
                self.accelerator.print(
                    "Warning: Resuming checkpoint missing 'update' or 'step'. Starting from update 0."
                )
                start_update = 0
            else:
                if (
                    "step" in checkpoint
                    and "update" not in checkpoint
                    and self.grad_accumulation_steps > 1
                ):
                    start_update = start_update // self.grad_accumulation_steps
                    self.accelerator.print(
                        "Converted loaded 'step' to 'update' based on grad_accumulation_steps."
                    )
                start_update += 1
            self.step = start_update
            self.accelerator.print(f"Resuming training from update {start_update}")
        else:
            self.step = 0
            start_update = 0

            self.accelerator.print(
                f"Loaded pre-trained weights (from key '{loaded_from_key}'). Starting fine-tuning from update 0."
            )

        del checkpoint
        del model_sd_raw
        del model_sd_cleaned
        if "loaded_data" in locals():
            del loaded_data
        self.accelerator.wait_for_everyone()
        if self.accelerator.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.accelerator.device.type == "xpu":
            torch.xpu.empty_cache()
        gc.collect()

        self.accelerator.print("Checkpoint loading process finished.")
        return start_update

    def calculate_duration_loss(
        self,
        text_inputs,
        mel_spec,
        mel_lengths,
        text_lengths,
        phoneme_sequences,
        batch_indices,
        global_update,
        current_epoch,
    ):
        """
        Calculate duration loss using monotonic alignment search instead of uniform distribution.
        This produces more realistic phoneme-level durations.

        Args:
            text_inputs: Input text tokens
            mel_spec: Mel spectrogram [batch, time, dim]
            mel_lengths: Length of each mel spectrogram in the batch
            text_lengths: Length of each text in the batch
            phoneme_sequences: List of phoneme sequences for each sample in batch
            batch_indices: Indices of samples in the batch
            global_update: Current global update counter

        Returns:
            tuple of (duration_loss, duration_mae, alignment_viz) for logging
        """
        if self.duration_predictor is None or text_lengths is None:
            return None, None, None

        try:
            from f5_tts.model.alignment_utils import (
                phoneme_to_indices,
                monotonic_alignment_search,
                get_durations_from_alignment,
            )

            if isinstance(phoneme_sequences, list) and len(phoneme_sequences) > 0:

                phoneme_data = phoneme_sequences
            else:

                phoneme_data = text_inputs
                if self.accelerator.is_local_main_process and global_update % 100 == 0:
                    print(f"WARNING: No phoneme data found. Using text as fallback.")

            phoneme_indices_list, updated_map = phoneme_to_indices(
                phoneme_data, phoneme_map=self.phoneme_map
            )

            if len(updated_map) > len(self.phoneme_map):
                self.phoneme_map = updated_map
                if self.accelerator.is_local_main_process and global_update % 100 == 0:
                    print(f"Updated phoneme map to {len(self.phoneme_map)} phonemes")

            b = len(phoneme_indices_list)
            max_nt = max(len(seq) for seq in phoneme_indices_list)

            phoneme_tensor = torch.zeros(
                b, max_nt, dtype=torch.long, device=mel_spec.device
            )
            phoneme_mask = torch.zeros(
                b, max_nt, dtype=torch.bool, device=mel_spec.device
            )

            for i, seq in enumerate(phoneme_indices_list):
                L = len(seq)
                phoneme_tensor[i, :L] = torch.tensor(seq, device=mel_spec.device)
                phoneme_mask[i, :L] = True

            self.phoneme_mask = phoneme_mask
            self.stored_mel_lengths = mel_lengths
            self.stored_text_lengths = phoneme_mask.sum(dim=1)

            alignment_method, logs = alignment_utils.get_alignment_method(
                self.alignment_manager,
                global_update,
                duration_focus_updates=self.duration_focus_updates,
                phase2_start_update=self.phase2_start_update,
                current_epoch=current_epoch,
            )

            if "phase_transition" in logs and logs["phase_transition"]:
                self.phase2_start_update = global_update
                self.duration_focus_phase = False

                self.active_duration_loss_weight = logs["duration_weight"]

                unwrapped_model = self.accelerator.unwrap_model(self.model)
                if self.duration_predictor is not None:
                    duration_params = list(self.duration_predictor.parameters())
                    for param in unwrapped_model.parameters():
                        if not any(param is dp for dp in duration_params):
                            param.requires_grad = True

                print(f"\n===== PHASE TRANSITION at update {global_update} =====")
                print(f"Reason: {logs.get('transition_reason', 'Unknown')}")
                print(
                    f"Switching to Phase 2 - Full Model Training with Window Alignment"
                )
                print(f"Duration loss weight: {self.active_duration_loss_weight}")
                print(f"====================================================\n")

            with torch.no_grad():

                phoneme_embed = self.duration_predictor_align.text_embed(phoneme_tensor)

                phoneme_embed_norm = phoneme_embed / (
                    phoneme_embed.norm(dim=2, keepdim=True) + 1e-8
                )

                if not hasattr(self, "mel_proj_matrix"):

                    self.mel_proj_matrix = torch.randn(
                        mel_spec.shape[-1],
                        phoneme_embed.shape[-1],
                        device=mel_spec.device,
                    ) / math.sqrt(mel_spec.shape[-1])

                mel_proj = torch.matmul(mel_spec, self.mel_proj_matrix)
                mel_proj_norm = mel_proj / (mel_proj.norm(dim=2, keepdim=True) + 1e-8)

                similarity = torch.bmm(
                    phoneme_embed_norm, mel_proj_norm.transpose(1, 2)
                )

                batch_size = phoneme_tensor.shape[0]
                for i in range(batch_size):
                    p_len = phoneme_mask[i].sum().item()
                    m_len = mel_lengths[i].item()
                    if p_len > 0 and m_len > 0:

                        for p in range(p_len):

                            center = int(p * m_len / p_len)

                            window = max(3, m_len // 10)
                            start = max(0, center - window)
                            end = min(m_len, center + window)

                            similarity[i, p, start:end] += 3.0

                for i in range(batch_size):
                    if phoneme_mask[i].sum() < max_nt:
                        similarity[i, phoneme_mask[i].sum() :, :] = -float("inf")

                    if mel_lengths[i] < mel_spec.shape[1]:
                        similarity[i, :, mel_lengths[i] :] = -float("inf")

                if global_update % 100 == 0:
                    print(f"Using alignment method: {alignment_method}")
                alignment = monotonic_alignment_search(
                    similarity, algorithm=alignment_method
                )

                self.previous_alignment = alignment

                phoneme_durations = get_durations_from_alignment(alignment)

                phoneme_durations = torch.maximum(
                    phoneme_durations * phoneme_mask.float(),
                    torch.ones_like(phoneme_durations) * 0.1 * phoneme_mask.float(),
                )

                if self.accelerator.is_local_main_process and global_update % 100 == 0:

                    total_frames = phoneme_durations.sum().item()
                    total_phonemes = phoneme_mask.sum().item()
                    avg_duration = total_frames / (total_phonemes + 1e-8)

                    max_duration = phoneme_durations.max().item()

                    min_duration = (
                        torch.where(
                            phoneme_durations > 0,
                            phoneme_durations,
                            torch.full_like(phoneme_durations, float("inf")),
                        )
                        .min()
                        .item()
                    )
                    if min_duration == float("inf"):
                        min_duration = 0.0

                    print(
                        f"[DURATION STATS] Average frames per phoneme: {avg_duration:.2f}"
                    )
                    print(
                        f"[DURATION STATS] Min duration: {min_duration:.2f}, Max duration: {max_duration:.2f}"
                    )
                    print(
                        f"[DURATION STATS] Total frames: {total_frames:.0f}, Total phonemes: {total_phonemes:.0f}"
                    )

            target_logw = torch.log(phoneme_durations + 1e-6)

            logw = self.duration_predictor_pred(phoneme_tensor, phoneme_mask)
            if logw.dim() == 3 and logw.size(1) == 1:
                logw = logw.squeeze(1)

            masked_diff = (logw - target_logw.detach()) ** 2 * phoneme_mask.float()
            dur_loss = torch.sum(masked_diff) / (torch.sum(phoneme_mask) + 1e-8)

            pred_durations = torch.exp(torch.clamp(logw, -10, 10))
            true_durations = phoneme_durations
            masked_diff_mae = (
                torch.abs(pred_durations - true_durations) * phoneme_mask.float()
            )
            dur_mae = torch.sum(masked_diff_mae) / (torch.sum(phoneme_mask) + 1e-8)

            alignment_viz = None
            if self.accelerator.is_local_main_process and global_update % 500 == 0:
                try:

                    sample_idx = 0

                    if isinstance(text_inputs, list):
                        sample_text = text_inputs[sample_idx]
                    elif torch.is_tensor(text_inputs):
                        sample_text = text_inputs[sample_idx].tolist()
                    else:
                        sample_text = str(text_inputs[sample_idx])

                    if (
                        isinstance(phoneme_data, list)
                        and len(phoneme_data) > sample_idx
                    ):
                        sample_phonemes = phoneme_data[sample_idx]
                    else:
                        sample_phonemes = ["no_phonemes_available"]

                    sample_durations = phoneme_durations[sample_idx].cpu().numpy()
                    if len(sample_phonemes) > len(sample_durations):
                        sample_phonemes = sample_phonemes[: len(sample_durations)]

                    if isinstance(sample_text, (list, tuple)):
                        sample_text = [str(t) for t in sample_text]

                    if isinstance(sample_phonemes, (list, tuple)):
                        sample_phonemes = [str(p) for p in sample_phonemes]

                    alignment_viz = {
                        "text": sample_text,
                        "phonemes": sample_phonemes,
                        "durations": sample_durations.tolist(),
                    }
                except Exception as e:
                    print(f"Error creating alignment visualization: {e}")
                    alignment_viz = None

            return dur_loss, dur_mae, alignment_viz

        except Exception as e:
            print(f"[ERROR] Phoneme duration prediction failed: {e}")
            import traceback

            traceback.print_exc()

            dur_loss = torch.tensor(0.01, device=mel_spec.device, requires_grad=True)
            dur_mae = torch.tensor(1.0, device=mel_spec.device)

            return dur_loss, dur_mae, None

    def train(
        self, train_dataset: Dataset, num_workers=16, resumable_with_seed: int = None
    ):
        """
        Train the model with phoneme-based duration prediction using monotonic alignment.

        Args:
            train_dataset: Dataset to train on (must contain 'phoneme' field if duration prediction is enabled)
            num_workers: Number of workers for data loading
            resumable_with_seed: Random seed for resumable training
        """

        if self.phoneme_map is None:
            self.phoneme_map = {}
            print("Initialized empty phoneme map - will build during training")

        self.clear_GPU_steps = 100

        if self.log_samples:
            from f5_tts.infer.utils_infer import (
                cfg_strength,
                load_vocoder,
                nfe_step,
                sway_sampling_coef,
            )

            vocoder = load_vocoder(
                vocoder_name=self.vocoder_name,
                is_local=self.is_local_vocoder,
                local_path=self.local_vocoder_path,
            )
            target_sample_rate = self.accelerator.unwrap_model(
                self.model
            ).mel_spec.target_sample_rate
            log_samples_path = f"{self.checkpoint_path}/samples"
            os.makedirs(log_samples_path, exist_ok=True)

            alignment_path = f"{self.checkpoint_path}/alignments"
            os.makedirs(alignment_path, exist_ok=True)

        if exists(resumable_with_seed):
            generator = torch.Generator()
            generator.manual_seed(resumable_with_seed)
        else:
            generator = None

        def collate_with_indices(batch):

            result = collate_fn(batch)

            result["index"] = list(range(len(batch)))

            if all("phoneme" in item for item in batch):

                result["phonemes"] = [item["phoneme"] for item in batch]
            elif "phoneme" in result:

                result["phonemes"] = result.pop("phoneme")

            return result

        if self.batch_size_type == "sample":
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_with_indices,
                num_workers=8,
                pin_memory=True,
                persistent_workers=True,
                prefetch_factor=2,
                batch_size=self.batch_size_per_gpu,
                shuffle=True,
                generator=generator,
            )
        elif self.batch_size_type == "frame":
            self.accelerator.even_batches = False

            sampler = SequentialSampler(train_dataset)
            batch_sampler = DynamicBatchSampler(
                sampler,
                self.batch_size_per_gpu,
                max_samples=self.max_samples,
                random_seed=resumable_with_seed,
                drop_residual=False,
            )
            train_dataloader = DataLoader(
                train_dataset,
                collate_fn=collate_with_indices,
                num_workers=8,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True,
                batch_sampler=batch_sampler,
            )
        else:
            raise ValueError(
                f"batch_size_type must be either 'sample' or 'frame', but received {self.batch_size_type}"
            )

        steps_per_epoch = len(train_dataloader) // self.grad_accumulation_steps
        if hasattr(self, "alignment_manager") and self.alignment_manager is not None:
            self.alignment_manager.set_steps_per_epoch(steps_per_epoch)
            print(
                f"Set duration weight decay over {self.alignment_manager.decay_epochs} epochs ({steps_per_epoch * self.alignment_manager.decay_epochs} steps)"
            )

        warmup_updates = self.num_warmup_updates * self.accelerator.num_processes
        total_updates = (
            math.ceil(len(train_dataloader) / self.grad_accumulation_steps)
            * self.epochs
        )
        decay_updates = total_updates - warmup_updates
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_updates,
        )
        decay_scheduler = LinearLR(
            self.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, decay_scheduler],
            milestones=[warmup_updates],
        )

        train_dataloader, self.scheduler = self.accelerator.prepare(
            train_dataloader, self.scheduler
        )

        start_update = self.load_checkpoint()
        global_update = start_update

        if exists(resumable_with_seed):
            orig_epoch_step = len(train_dataloader)
            start_step = start_update * self.grad_accumulation_steps
            skipped_epoch = int(start_step // orig_epoch_step)
            skipped_batch = start_step % orig_epoch_step
            skipped_dataloader = self.accelerator.skip_first_batches(
                train_dataloader, num_batches=skipped_batch
            )
        else:
            skipped_epoch = 0

        for epoch in range(skipped_epoch, self.epochs):
            self.model.train()
            if exists(resumable_with_seed) and epoch == skipped_epoch:
                progress_bar_initial = math.ceil(
                    skipped_batch / self.grad_accumulation_steps
                )
                current_dataloader = skipped_dataloader
            else:
                progress_bar_initial = 0
                current_dataloader = train_dataloader

            if hasattr(train_dataloader, "batch_sampler") and hasattr(
                train_dataloader.batch_sampler, "set_epoch"
            ):
                train_dataloader.batch_sampler.set_epoch(epoch)

            progress_bar = tqdm(
                range(math.ceil(len(train_dataloader) / self.grad_accumulation_steps)),
                desc=f"Epoch {epoch + 1}/{self.epochs}",
                unit="update",
                disable=not self.accelerator.is_local_main_process,
                initial=progress_bar_initial,
            )

            for batch in current_dataloader:
                with self.accelerator.accumulate(self.model):
                    text_inputs = batch["text"]
                    mel_spec = batch["mel"].permute(0, 2, 1)
                    mel_lengths = batch["mel_lengths"]
                    text_lengths = batch.get("text_lengths")
                    batch_indices = batch.get("index", None)
                    batch_phonemes = batch.get("phonemes", None)

                    self.stored_mel_lengths = mel_lengths
                    if text_lengths is not None:
                        self.stored_text_lengths = text_lengths
                    elif (
                        hasattr(self, "phoneme_mask") and self.phoneme_mask is not None
                    ):
                        self.stored_text_lengths = self.phoneme_mask.sum(dim=1)
                    else:
                        self.stored_text_lengths = None

                    loss, cond, pred = self.model(
                        mel_spec,
                        text=text_inputs,
                        lens=mel_lengths,
                        noise_scheduler=self.noise_scheduler,
                    )

                    dur_loss, dur_mae, alignment_viz = None, None, None
                    if self.duration_predictor is not None and text_lengths is not None:
                        dur_loss, dur_mae, alignment_viz = self.calculate_duration_loss(
                            text_inputs,
                            mel_spec,
                            mel_lengths,
                            text_lengths,
                            batch_phonemes,
                            batch_indices,
                            global_update,
                            epoch,
                        )

                        if (
                            dur_loss is not None
                            and self.active_duration_loss_weight > 0
                        ):
                            loss = loss + self.active_duration_loss_weight * dur_loss

                    if (
                        self.accelerator.is_local_main_process
                        and dur_loss is not None
                        and global_update % 100 == 0
                    ):
                        print(
                            f"[DURATION] dur_loss: {dur_loss.item():.4f}, dur_mae: {dur_mae.item():.4f}"
                        )
                        print(
                            f"[DURATION] Phase: {'Focus' if self.duration_focus_phase else 'Regular'}, Weight: {self.active_duration_loss_weight}"
                        )

                        if alignment_viz is not None and global_update % 500 == 0:
                            import json

                            vis_path = f"{self.checkpoint_path}/alignments/alignment_{global_update}.json"
                            with open(vis_path, "w") as f:
                                json.dump(
                                    alignment_viz, f, ensure_ascii=False, indent=2
                                )
                            print(f"Saved alignment visualization: {vis_path}")

                    self.accelerator.backward(loss)

                    if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), self.max_grad_norm
                        )

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:

                    if (
                        hasattr(self, "duration_focus_phase")
                        and self.duration_focus_phase
                        and global_update >= self.duration_focus_updates
                    ):
                        self.duration_focus_phase = False

                        self.active_duration_loss_weight = (
                            self.base_duration_loss_weight
                        )

                        unwrapped_model = self.accelerator.unwrap_model(self.model)
                        if self.duration_predictor is not None:
                            duration_params = list(self.duration_predictor.parameters())
                            for param in unwrapped_model.parameters():
                                if not any(param is dp for dp in duration_params):
                                    param.requires_grad = True

                        print(
                            f"Duration focus phase completed at update {global_update}."
                        )
                        print(
                            f"Main model unfrozen. Duration loss weight set to {self.active_duration_loss_weight}"
                        )

                        if (
                            hasattr(self, "duration_predictor")
                            and self.duration_predictor is not None
                            and self.is_main
                        ):
                            duration_checkpoint = {
                                "duration_predictor": self.accelerator.unwrap_model(
                                    self.duration_predictor
                                ).state_dict(),
                                "update": global_update,
                                "duration_loss_weight": self.active_duration_loss_weight,
                                "phoneme_map": self.phoneme_map,
                            }
                            checkpoint_dir = (
                                os.path.dirname(self.checkpoint_path)
                                if os.path.isfile(self.checkpoint_path)
                                else self.checkpoint_path
                            )
                            os.makedirs(checkpoint_dir, exist_ok=True)
                            self.accelerator.save(
                                duration_checkpoint,
                                f"{checkpoint_dir}/duration_predictor_{global_update}.pt",
                            )
                            print(
                                f"Saved separate duration predictor checkpoint at end of focus phase (update {global_update})"
                            )

                    if self.is_main:
                        self.ema_model.update()

                    global_update += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        update=str(global_update), loss=loss.item()
                    )

                if self.accelerator.is_local_main_process:
                    log_dict = {
                        "loss": loss.item(),
                        "lr": self.scheduler.get_last_lr()[0],
                    }
                    if dur_loss is not None:
                        log_dict["duration_loss"] = dur_loss.item()
                        log_dict["duration_loss_weight"] = (
                            self.active_duration_loss_weight
                        )

                    if dur_mae is not None:
                        log_dict["duration_mae"] = dur_mae.item()

                    if (
                        hasattr(self, "previous_alignment")
                        and self.previous_alignment is not None
                    ):

                        batch_mel_lengths = mel_lengths
                        batch_text_lengths = (
                            text_lengths
                            if torch.is_tensor(text_lengths)
                            else phoneme_mask.sum(dim=1)
                        )

                        if hasattr(self.alignment_manager, "current_method"):
                            log_dict["alignment_method"] = (
                                self.alignment_manager.current_method
                            )

                    self.accelerator.log(log_dict, step=global_update)

                    if self.logger == "tensorboard":
                        self.writer.add_scalar("loss", loss.item(), global_update)
                        if dur_loss is not None:
                            self.writer.add_scalar(
                                "duration_loss", dur_loss.item(), global_update
                            )
                            self.writer.add_scalar(
                                "duration_loss_weight",
                                self.active_duration_loss_weight,
                                global_update,
                            )

                        if dur_mae is not None:
                            self.writer.add_scalar(
                                "duration_mae", dur_mae.item(), global_update
                            )

                        if (
                            hasattr(self, "previous_alignment")
                            and self.previous_alignment is not None
                        ):
                            if hasattr(self.alignment_manager, "current_method"):
                                self.writer.add_text(
                                    "alignment_method",
                                    self.alignment_manager.current_method,
                                    global_update,
                                )

                        self.writer.add_scalar(
                            "lr", self.scheduler.get_last_lr()[0], global_update
                        )

                if (
                    global_update % self.save_per_updates == 0
                    and self.accelerator.sync_gradients
                ):
                    self.save_checkpoint(global_update)

                    if self.duration_predictor is not None and self.is_main:
                        duration_checkpoint = {
                            "duration_predictor": self.accelerator.unwrap_model(
                                self.duration_predictor
                            ).state_dict(),
                            "update": global_update,
                            "duration_loss_weight": self.active_duration_loss_weight,
                            "phoneme_map": self.phoneme_map,
                        }
                        checkpoint_dir = (
                            os.path.dirname(self.checkpoint_path)
                            if os.path.isfile(self.checkpoint_path)
                            else self.checkpoint_path
                        )
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        self.accelerator.save(
                            duration_checkpoint,
                            f"{checkpoint_dir}/duration_predictor_{global_update}.pt",
                        )
                        print(
                            f"Saved duration predictor checkpoint with phoneme map at update {global_update}"
                        )

                    import gc

                    gc.collect()

                    if global_update % self.clear_GPU_steps == 0:
                        torch.cuda.empty_cache()

                    if self.log_samples and self.accelerator.is_local_main_process:

                        ref_audio_len = mel_lengths[0]
                        infer_text = [
                            text_inputs[0]
                            + ([" "] if isinstance(text_inputs[0], list) else " ")
                            + text_inputs[0]
                        ]
                        with torch.inference_mode():
                            generated, _ = self.accelerator.unwrap_model(
                                self.model
                            ).sample(
                                cond=mel_spec[0][:ref_audio_len].unsqueeze(0),
                                text=infer_text,
                                duration=ref_audio_len * 2,
                                steps=nfe_step,
                                cfg_strength=cfg_strength,
                                sway_sampling_coef=sway_sampling_coef,
                            )
                            generated = generated.to(torch.float32)
                            gen_mel_spec = (
                                generated[:, ref_audio_len:, :]
                                .permute(0, 2, 1)
                                .to(self.accelerator.device)
                            )
                            ref_mel_spec = batch["mel"][0].unsqueeze(0)
                            if self.vocoder_name == "vocos":
                                gen_audio = vocoder.decode(gen_mel_spec).cpu()
                                ref_audio = vocoder.decode(ref_mel_spec).cpu()
                            elif self.vocoder_name == "bigvgan":
                                gen_audio = vocoder(gen_mel_spec).squeeze(0).cpu()
                                ref_audio = vocoder(ref_mel_spec).squeeze(0).cpu()
                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_gen.wav",
                            gen_audio,
                            target_sample_rate,
                        )
                        torchaudio.save(
                            f"{log_samples_path}/update_{global_update}_ref.wav",
                            ref_audio,
                            target_sample_rate,
                        )

                        self.generate_reference_samples(
                            global_update,
                            vocoder,
                            nfe_step,
                            cfg_strength,
                            sway_sampling_coef,
                        )

                        self.model.train()

                if (
                    global_update % self.last_per_updates == 0
                    and self.accelerator.sync_gradients
                ):
                    self.save_checkpoint(global_update, last=True)

        self.save_checkpoint(global_update, last=True)

        self.accelerator.end_training()
