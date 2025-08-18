import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict

from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf
from importlib.resources import files
from pydub import AudioSegment, silence

from f5_tts.model import CFM
from f5_tts.model.utils import (
    get_tokenizer,
    list_str_to_idx,
)
from f5_tts.infer.utils_infer import (
    chunk_text,
    load_vocoder,
    transcribe,
    initialize_asr_pipeline,
)


class F5TTSWrapper:
    """
    A wrapper class for F5-TTS that preprocesses reference audio once
    and allows for repeated TTS generation.
    """

    def __init__(
        self,
        model_name: str = "F5TTS_v1_Base",
        ckpt_path: Optional[str] = None,
        vocab_file: Optional[str] = None,
        vocoder_name: str = "vocos",
        use_local_vocoder: bool = False,
        vocoder_path: Optional[str] = None,
        device: Optional[str] = None,
        hf_cache_dir: Optional[str] = None,
        target_sample_rate: int = 24000,
        n_mel_channels: int = 100,
        hop_length: int = 256,
        win_length: int = 1024,
        n_fft: int = 1024,
        ode_method: str = "euler",
        use_ema: bool = True,
        use_duration_predictor: bool = False,
        duration_predictor_path: Optional[str] = None,
    ):
        """
        Initialize the F5-TTS wrapper with model configuration.

        Args:
            model_name: Name of the F5-TTS model variant (e.g., "F5TTS_v1_Base")
            ckpt_path: Path to the model checkpoint file. If None, will use default path.
            vocab_file: Path to the vocab file. If None, will use default.
            vocoder_name: Name of the vocoder to use ("vocos" or "bigvgan")
            use_local_vocoder: Whether to use a local vocoder or download from HF
            vocoder_path: Path to the local vocoder. Only used if use_local_vocoder is True.
            device: Device to run the model on. If None, will automatically determine.
            hf_cache_dir: Directory to cache HuggingFace models
            target_sample_rate: Target sample rate for audio
            n_mel_channels: Number of mel channels
            hop_length: Hop length for the mel spectrogram
            win_length: Window length for the mel spectrogram
            n_fft: FFT size for the mel spectrogram
            ode_method: ODE method for sampling ("euler" or "midpoint")
            use_ema: Whether to use EMA weights from the checkpoint
            use_duration_predictor: Whether to use the duration predictor for inference
            duration_predictor_path: Path to separate duration predictor checkpoint
        """

        if device is None:
            self.device = (
                "cuda"
                if torch.cuda.is_available()
                else (
                    "xpu"
                    if torch.xpu.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu"
                )
            )
        else:
            self.device = device

        self.target_sample_rate = target_sample_rate
        self.n_mel_channels = n_mel_channels
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.mel_spec_type = vocoder_name

        self.ode_method = ode_method

        self.use_duration_predictor = use_duration_predictor
        self.phoneme_map = {}

        initialize_asr_pipeline(device=self.device)

        if ckpt_path is None:
            repo_name = "F5-TTS"
            ckpt_step = 1250000
            ckpt_type = "safetensors"

            if model_name == "F5TTS_v1_Custom_Prune_12":
                if vocoder_name == "vocos":
                    ckpt_step = 1200000
            elif model_name == "F5TTS_Base":
                if vocoder_name == "vocos":
                    ckpt_step = 1200000
                elif vocoder_name == "bigvgan":
                    model_name = "F5TTS_Base_bigvgan"
                    ckpt_type = "pt"
            elif model_name == "E2TTS_Base":
                repo_name = "E2-TTS"
                ckpt_step = 1200000
            else:
                if vocoder_name == "vocos":
                    ckpt_step = 1200000

            ckpt_path = str(
                cached_path(
                    f"hf://SWivid/{repo_name}/{model_name}/model_{ckpt_step}.{ckpt_type}"
                )
            )

        if "custom" not in model_name.lower():
            config_path = str(files("f5_tts").joinpath(f"configs/{model_name}.yaml"))
        else:
            config_path = model_name

        model_cfg = OmegaConf.load(config_path)
        model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        model_arc = model_cfg.model.arch

        if vocab_file is None:
            vocab_file = str(files("f5_tts").joinpath("infer/examples/vocab.txt"))
        tokenizer_type = "custom"
        self.vocab_char_map, vocab_size = get_tokenizer(vocab_file, tokenizer_type)

        self.model = CFM(
            transformer=model_cls(
                **model_arc, text_num_embeds=vocab_size, mel_dim=n_mel_channels
            ),
            mel_spec_kwargs=dict(
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mel_channels=n_mel_channels,
                target_sample_rate=target_sample_rate,
                mel_spec_type=vocoder_name,
            ),
            odeint_kwargs=dict(
                method=ode_method,
            ),
            vocab_char_map=self.vocab_char_map,
        ).to(self.device)

        dtype = torch.float32 if vocoder_name == "bigvgan" else None
        self._load_checkpoint(self.model, ckpt_path, dtype=dtype, use_ema=use_ema)

        self.has_duration_predictor = (
            hasattr(self.model, "duration_predictor")
            and self.model.duration_predictor is not None
        )

        if (
            use_duration_predictor
            and not self.has_duration_predictor
            and duration_predictor_path
        ):
            try:
                from f5_tts.model import DurationPredictor

                print(
                    f"Loading separate duration predictor from {duration_predictor_path}"
                )

                if duration_predictor_path.endswith(".safetensors"):
                    from safetensors.torch import load_file

                    checkpoint = load_file(duration_predictor_path, device=self.device)
                else:
                    checkpoint = torch.load(
                        duration_predictor_path, map_location=self.device
                    )

                if "duration_predictor" in checkpoint:
                    duration_state_dict = checkpoint["duration_predictor"]
                elif (
                    "model_state_dict" in checkpoint
                    and "duration_predictor" in checkpoint["model_state_dict"]
                ):

                    duration_state_dict = {
                        k.replace("duration_predictor.", ""): v
                        for k, v in checkpoint["model_state_dict"].items()
                        if k.startswith("duration_predictor.")
                    }
                else:

                    duration_state_dict = checkpoint

                self.duration_predictor = DurationPredictor(
                    vocab_size, 512, 32, 3, 0.5
                ).to(self.device)

                incompatible = self.duration_predictor.load_state_dict(
                    duration_state_dict, strict=False
                )
                if incompatible.missing_keys:
                    print(
                        f"Warning: Missing keys in duration predictor: {incompatible.missing_keys}"
                    )
                if incompatible.unexpected_keys:
                    print(
                        f"Warning: Unexpected keys in duration predictor: {incompatible.unexpected_keys}"
                    )

                self.duration_predictor.eval()

                if "phoneme_map" in checkpoint:
                    self.phoneme_map = checkpoint["phoneme_map"]
                    print(f"Loaded phoneme map with {len(self.phoneme_map)} phonemes")

                setattr(self.model, "duration_predictor", self.duration_predictor)
                self.has_duration_predictor = True

                print("Duration predictor loaded successfully")
            except Exception as e:
                print(f"Error loading duration predictor: {e}")
                import traceback

                traceback.print_exc()
                self.has_duration_predictor = False

        if use_duration_predictor and not self.has_duration_predictor:
            print(
                "Warning: Duration predictor requested but not available. Using fallback duration calculation."
            )
            self.use_duration_predictor = False
        elif self.has_duration_predictor:
            print("Duration predictor found and will be used for inference.")
            self.use_duration_predictor = True

        if vocoder_path is None:
            if vocoder_name == "vocos":
                vocoder_path = "../checkpoints/vocos-mel-24khz"
            elif vocoder_name == "bigvgan":
                vocoder_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"

        self.vocoder = load_vocoder(
            vocoder_name=vocoder_name,
            is_local=use_local_vocoder,
            local_path=vocoder_path,
            device=self.device,
            hf_cache_dir=hf_cache_dir,
        )

        self.ref_audio_processed = None
        self.ref_text = None
        self.ref_audio_len = None

        self.target_rms = 0.1
        self.cross_fade_duration = 0.15
        self.nfe_step = 32
        self.cfg_strength = 2.0
        self.sway_sampling_coef = -1.0
        self.speed = 1.0
        self.fix_duration = None

    def _load_checkpoint(self, model, ckpt_path, dtype=None, use_ema=True):
        """
        Load model checkpoint with proper handling of different checkpoint formats.

        Args:
            model: The model to load weights into
            ckpt_path: Path to the checkpoint file
            dtype: Data type for model weights
            use_ema: Whether to use EMA weights from the checkpoint

        Returns:
            Loaded model
        """
        if dtype is None:
            dtype = (
                torch.float16
                if "cuda" in self.device
                and torch.cuda.get_device_properties(self.device).major >= 7
                and not torch.cuda.get_device_name().endswith("[ZLUDA]")
                else torch.float32
            )
        model = model.to(dtype)

        ckpt_type = ckpt_path.split(".")[-1]
        if ckpt_type == "safetensors":
            from safetensors.torch import load_file

            checkpoint = load_file(ckpt_path, device=self.device)
        else:
            checkpoint = torch.load(
                ckpt_path, map_location=self.device, weights_only=True
            )

        if use_ema:
            if ckpt_type == "safetensors":
                checkpoint = {"ema_model_state_dict": checkpoint}
            checkpoint["model_state_dict"] = {
                k.replace("ema_model.", ""): v
                for k, v in checkpoint["ema_model_state_dict"].items()
                if k not in ["initted", "step"]
            }

            for key in [
                "mel_spec.mel_stft.mel_scale.fb",
                "mel_spec.mel_stft.spectrogram.window",
            ]:
                if key in checkpoint["model_state_dict"]:
                    del checkpoint["model_state_dict"][key]

            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            if ckpt_type == "safetensors":
                checkpoint = {"model_state_dict": checkpoint}
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        del checkpoint
        torch.cuda.empty_cache()

        return model.to(self.device)

    def preprocess_reference(
        self, ref_audio_path: str, ref_text: str = "", clip_short: bool = True
    ):
        """
        Preprocess the reference audio and text, storing them for later use.

        Args:
            ref_audio_path: Path to the reference audio file
            ref_text: Text transcript of reference audio. If empty, will auto-transcribe.
            clip_short: Whether to clip long audio to shorter segments

        Returns:
            Tuple of processed audio and text
        """
        print("Converting audio...")

        aseg = AudioSegment.from_file(ref_audio_path)

        if clip_short:

            non_silent_segs = silence.split_on_silence(
                aseg,
                min_silence_len=1000,
                silence_thresh=-50,
                keep_silence=1000,
                seek_step=10,
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                if (
                    len(non_silent_wave) > 6000
                    and len(non_silent_wave + non_silent_seg) > 12000
                ):
                    print("Audio is over 12s, clipping short. (1)")
                    break
                non_silent_wave += non_silent_seg

            if len(non_silent_wave) > 12000:
                non_silent_segs = silence.split_on_silence(
                    aseg,
                    min_silence_len=100,
                    silence_thresh=-40,
                    keep_silence=1000,
                    seek_step=10,
                )
                non_silent_wave = AudioSegment.silent(duration=0)
                for non_silent_seg in non_silent_segs:
                    if (
                        len(non_silent_wave) > 6000
                        and len(non_silent_wave + non_silent_seg) > 12000
                    ):
                        print("Audio is over 12s, clipping short. (2)")
                        break
                    non_silent_wave += non_silent_seg

            aseg = non_silent_wave

            if len(aseg) > 12000:
                aseg = aseg[:12000]
                print("Audio is over 12s, clipping short. (3)")

        aseg = self._remove_silence_edges(aseg) + AudioSegment.silent(duration=50)

        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            aseg.export(tmp_file.name, format="wav")
            processed_audio_path = tmp_file.name

        if not ref_text.strip():
            print("No reference text provided, transcribing reference audio...")
            ref_text = transcribe(processed_audio_path)
        else:
            print("Using custom reference text...")

        if not ref_text.endswith(". ") and not ref_text.endswith("。"):
            if ref_text.endswith("."):
                ref_text += " "
            else:
                ref_text += ". "

        print("\nReference text:", ref_text)

        audio, sr = torchaudio.load(processed_audio_path)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < self.target_rms:
            audio = audio * self.target_rms / rms

        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            audio = resampler(audio)

        audio = audio.to(self.device)

        self.ref_audio_processed = audio
        self.ref_text = ref_text
        self.ref_audio_len = audio.shape[-1] // self.hop_length

        os.unlink(processed_audio_path)

        return audio, ref_text

    def _remove_silence_edges(self, audio, silence_threshold=-42):
        """
        Remove silence from the start and end of audio.

        Args:
            audio: AudioSegment to process
            silence_threshold: dB threshold to consider as silence

        Returns:
            Processed AudioSegment
        """

        non_silent_start_idx = silence.detect_leading_silence(
            audio, silence_threshold=silence_threshold
        )
        audio = audio[non_silent_start_idx:]

        non_silent_end_duration = audio.duration_seconds
        for ms in reversed(audio):
            if ms.dBFS > silence_threshold:
                break
            non_silent_end_duration -= 0.001
        trimmed_audio = audio[: int(non_silent_end_duration * 1000)]

        return trimmed_audio

    def calculate_duration_with_predictor(self, text_input, local_speed=1.0):
        """
        Calculate duration using the model's duration predictor with phonemizer.

        Args:
            text_input: Text input string or token sequence
            local_speed: Speed factor for duration

        Returns:
            Predicted duration in frames
        """
        if not self.has_duration_predictor:
            raise ValueError("Duration predictor not available")

        from f5_tts.model.utils import list_str_to_idx
        from f5_tts.model.alignment_utils import (
            text_to_phonemes_espeak,
            phoneme_to_indices,
        )

        try:

            phoneme_seq = text_to_phonemes_espeak(text_input)

            phoneme_indices_list, updated_map = phoneme_to_indices(
                phoneme_seq, phoneme_map=self.phoneme_map
            )

            if len(updated_map) > len(self.phoneme_map):
                self.phoneme_map = updated_map
                print(f"Updated phoneme map to {len(self.phoneme_map)} phonemes")

            max_nt = max(len(seq) for seq in phoneme_indices_list)
            b = len(phoneme_indices_list)

            phoneme_tensor = torch.zeros(
                b, max_nt, dtype=torch.long, device=self.device
            )
            phoneme_mask = torch.zeros(b, max_nt, dtype=torch.bool, device=self.device)

            for i, seq in enumerate(phoneme_indices_list):
                L = len(seq)
                phoneme_tensor[i, :L] = torch.tensor(seq, device=self.device)
                phoneme_mask[i, :L] = True

            with torch.inference_mode():

                if hasattr(self.model.duration_predictor, "phoneme_forward"):
                    log_durations = self.model.duration_predictor.phoneme_forward(
                        phoneme_tensor, phoneme_mask
                    )
                else:
                    log_durations = self.model.duration_predictor(
                        phoneme_tensor, phoneme_mask
                    )

                if log_durations.dim() == 3 and log_durations.size(1) == 1:
                    log_durations = log_durations.squeeze(1)

                durations = torch.exp(torch.clamp(log_durations, -20, 20)).sum(dim=1)

            duration = self.ref_audio_len + int(durations[0].item() / local_speed)

            print(
                f"Duration predictor calculated {int(durations[0].item())} frames for text portion"
            )
            return duration

        except Exception as e:
            print(f"Error in duration prediction: {e}")
            import traceback

            traceback.print_exc()

            if isinstance(text_input, str):
                text_len = len(text_input.encode("utf-8"))
            else:
                text_len = sum(len(str(t).encode("utf-8")) for t in text_input)

            ref_text_len = len(self.ref_text.encode("utf-8"))
            duration = self.ref_audio_len + int(
                self.ref_audio_len / ref_text_len * text_len / local_speed
            )

            print(f"Falling back to text ratio duration: {duration} frames")
            return duration

    def generate(
        self,
        text: str,
        output_path: Optional[str] = None,
        nfe_step: Optional[int] = None,
        cfg_strength: Optional[float] = None,
        sway_sampling_coef: Optional[float] = None,
        speed: Optional[float] = None,
        fix_duration: Optional[float] = None,
        cross_fade_duration: Optional[float] = None,
        use_duration_predictor: Optional[bool] = None,
        return_numpy: bool = False,
        return_spectrogram: bool = False,
    ) -> Union[str, Tuple[np.ndarray, int], Tuple[np.ndarray, int, np.ndarray]]:
        """
        Generate speech for the given text using the stored reference audio.

        Args:
            text: Text to synthesize
            output_path: Path to save the generated audio. If None, won't save.
            nfe_step: Number of function evaluation steps
            cfg_strength: Classifier-free guidance strength
            sway_sampling_coef: Sway sampling coefficient
            speed: Speed of generated audio
            fix_duration: Fixed duration in seconds
            cross_fade_duration: Duration of cross-fade between segments
            use_duration_predictor: Override default setting for using duration predictor
            return_numpy: If True, returns the audio as a numpy array
            return_spectrogram: If True, also returns the spectrogram

        Returns:
            If output_path provided: path to output file
            If return_numpy=True: tuple of (audio_array, sample_rate)
            If return_spectrogram=True: tuple of (audio_array, sample_rate, spectrogram)
        """
        if self.ref_audio_processed is None or self.ref_text is None:
            raise ValueError(
                "Reference audio not preprocessed. Call preprocess_reference() first."
            )

        nfe_step = nfe_step if nfe_step is not None else self.nfe_step
        cfg_strength = cfg_strength if cfg_strength is not None else self.cfg_strength
        sway_sampling_coef = (
            sway_sampling_coef
            if sway_sampling_coef is not None
            else self.sway_sampling_coef
        )
        speed = speed if speed is not None else self.speed
        fix_duration = fix_duration if fix_duration is not None else self.fix_duration
        cross_fade_duration = (
            cross_fade_duration
            if cross_fade_duration is not None
            else self.cross_fade_duration
        )
        use_predictor = (
            use_duration_predictor
            if use_duration_predictor is not None
            else self.use_duration_predictor
        )

        can_use_predictor = use_predictor and self.has_duration_predictor

        audio_len = self.ref_audio_processed.shape[-1] / self.target_sample_rate
        max_chars = int(
            len(self.ref_text.encode("utf-8")) / audio_len * (22 - audio_len)
        )
        text_batches = chunk_text(text, max_chars=max_chars)

        for i, text_batch in enumerate(text_batches):
            print(f"Text batch {i}: {text_batch}")
        print("\n")

        generated_waves = []
        spectrograms = []

        for text_batch in text_batches:

            local_speed = speed
            if len(text_batch.encode("utf-8")) < 10:
                local_speed = 0.3

            text_list = [self.ref_text + text_batch]
            final_text_list = text_list

            if fix_duration is not None:

                duration = int(fix_duration * self.target_sample_rate / self.hop_length)
                print(f"Using fixed duration: {fix_duration}s ({duration} frames)")
            elif can_use_predictor:

                duration = self.calculate_duration_with_predictor(
                    text_batch, local_speed
                )
                print(f"Duration predictor output: {duration} frames")
            else:

                ref_text_len = len(self.ref_text.encode("utf-8"))
                gen_text_len = len(text_batch.encode("utf-8"))
                duration = self.ref_audio_len + int(
                    self.ref_audio_len / ref_text_len * gen_text_len / local_speed
                )
                print(f"Calculated duration based on text ratio: {duration} frames")

            with torch.inference_mode():
                generated, _ = self.model.sample(
                    cond=self.ref_audio_processed,
                    text=final_text_list,
                    duration=duration,
                    steps=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                )

                generated = generated.to(torch.float32)
                generated = generated[:, self.ref_audio_len :, :]
                generated = generated.permute(0, 2, 1)

                if self.mel_spec_type == "vocos":
                    generated_wave = self.vocoder.decode(generated)
                elif self.mel_spec_type == "bigvgan":
                    generated_wave = self.vocoder(generated)

                rms = torch.sqrt(torch.mean(torch.square(self.ref_audio_processed)))
                if rms < self.target_rms:
                    generated_wave = generated_wave * rms / self.target_rms

                generated_wave = generated_wave.squeeze().cpu().numpy()
                generated_waves.append(generated_wave)

                if return_spectrogram or output_path is not None:
                    spectrograms.append(generated.squeeze().cpu().numpy())

        if generated_waves:
            if cross_fade_duration <= 0:

                final_wave = np.concatenate(generated_waves)
            else:

                final_wave = generated_waves[0]
                for i in range(1, len(generated_waves)):
                    prev_wave = final_wave
                    next_wave = generated_waves[i]

                    cross_fade_samples = int(
                        cross_fade_duration * self.target_sample_rate
                    )
                    cross_fade_samples = min(
                        cross_fade_samples, len(prev_wave), len(next_wave)
                    )

                    if cross_fade_samples <= 0:

                        final_wave = np.concatenate([prev_wave, next_wave])
                        continue

                    prev_overlap = prev_wave[-cross_fade_samples:]
                    next_overlap = next_wave[:cross_fade_samples]

                    fade_out = np.linspace(1, 0, cross_fade_samples)
                    fade_in = np.linspace(0, 1, cross_fade_samples)

                    cross_faded_overlap = (
                        prev_overlap * fade_out + next_overlap * fade_in
                    )

                    final_wave = np.concatenate(
                        [
                            prev_wave[:-cross_fade_samples],
                            cross_faded_overlap,
                            next_wave[cross_fade_samples:],
                        ]
                    )

            if return_spectrogram or output_path is not None:
                combined_spectrogram = np.concatenate(spectrograms, axis=1)

            if output_path is not None:
                output_dir = os.path.dirname(output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                torchaudio.save(
                    output_path,
                    torch.tensor(final_wave).unsqueeze(0),
                    self.target_sample_rate,
                )

                if return_spectrogram:
                    spectrogram_path = os.path.splitext(output_path)[0] + "_spec.png"
                    self._save_spectrogram(combined_spectrogram, spectrogram_path)

                if not return_numpy:
                    return output_path

            if return_spectrogram:
                return final_wave, self.target_sample_rate, combined_spectrogram
            else:
                return final_wave, self.target_sample_rate

        else:
            raise RuntimeError("No audio generated")

    def _save_spectrogram(self, spectrogram, path):
        """Save spectrogram as image"""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 4))
        plt.imshow(spectrogram, origin="lower", aspect="auto")
        plt.colorbar()
        plt.savefig(path)
        plt.close()

    def get_current_audio_length(self):
        """Get the length of the reference audio in seconds"""
        if self.ref_audio_processed is None:
            return 0
        return self.ref_audio_processed.shape[-1] / self.target_sample_rate
