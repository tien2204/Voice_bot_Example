import torch
import numpy as np
from viphoneme import vi2IPA
from tqdm import tqdm
import math


def text_to_phonemes(text):
    """
    Convert Vietnamese text to IPA phonemes using viphoneme.
    Improved handling of edge cases and different input types.

    Args:
        text: String or list of strings/lists to convert

    Returns:
        List of phoneme sequences
    """

    if isinstance(text, list):
        result = []
        for t in tqdm(text):

            if isinstance(t, list):

                t_str = " ".join(str(item) for item in t if item is not None)
                result.append(vi2IPA(t_str).strip().split(" "))
            else:

                result.append(vi2IPA(str(t)).strip().split(" "))
        return result

    return vi2IPA(str(text).strip().split(" "))


from phonemizer import phonemize


def text_to_phonemes_espeak(text_input):

    phoneme_seq = []

    if isinstance(text_input, str):

        phoneme_seq = phonemize(str(text_input), language="vi").strip().split(" ")

        phoneme_seq = [phoneme_seq]
    elif isinstance(text_input, list):

        phoneme_seq = []
        for text in text_input:
            if text:
                phoneme_seq.append(
                    phonemize(str(text), language="vi").strip().split(" ")
                )
            else:
                phoneme_seq.append([])
    else:
        raise ValueError(f"Unsupported input type: {type(text_input)}")

    return phoneme_seq


def phoneme_to_indices(phoneme_seq, phoneme_map=None):
    """
    Convert phoneme sequence to indices, handling edge cases better.
    If phoneme_map is provided, uses it. Otherwise creates a new one.

    Args:
        phoneme_seq: List of phoneme sequences or a single sequence
        phoneme_map: Optional mapping of phonemes to indices

    Returns:
        Tuple of (indices, phoneme_map)
    """

    if phoneme_map is None:

        unique_phonemes = set()

        if isinstance(phoneme_seq, list):
            for seq in phoneme_seq:
                if seq:
                    unique_phonemes.update(seq)
        elif phoneme_seq:
            unique_phonemes.update(phoneme_seq)

        phoneme_map = {p: i + 1 for i, p in enumerate(sorted(unique_phonemes))}

    if isinstance(phoneme_seq, list):

        indices = []
        for seq in phoneme_seq:

            if not seq:
                indices.append([0])
            else:
                indices.append([phoneme_map.get(p, 0) for p in seq])
    else:

        indices = [phoneme_map.get(p, 0) for p in (phoneme_seq or [])]

    return indices, phoneme_map


def create_phoneme_embedding(phoneme_map, embed_dim=192):
    """
    Create embedding layer for phonemes.

    Args:
        phoneme_map: Dictionary mapping phonemes to indices
        embed_dim: Embedding dimension

    Returns:
        torch.nn.Embedding layer
    """
    num_phonemes = len(phoneme_map) + 1
    return torch.nn.Embedding(num_phonemes, embed_dim)


def get_durations_from_alignment(alignment):
    """
    Extract duration of each token from alignment matrix.

    Args:
        alignment: Alignment matrix [b, nt, mel_len]

    Returns:
        durations: Duration of each token [b, nt]
    """
    return alignment.sum(dim=2)


def phonemes_to_mel_alignment(phoneme_tensor, phoneme_mask, mel_spec, model):
    """
    Calculate alignment between phonemes and mel-spectrogram frames.

    Args:
        phoneme_tensor: Tensor of phoneme indices [b, nt]
        phoneme_mask: Mask for phoneme tensor [b, nt]
        mel_spec: Mel spectrogram [b, n_mels, mel_len]
        model: Model with alignment network

    Returns:
        alignment: Hard alignment matrix [b, nt, mel_len]
        similarity: Similarity matrix [b, nt, mel_len]
    """

    similarity = model.alignment_network(
        phoneme_tensor, phoneme_mask, mel_spec.permute(0, 2, 1)
    )

    alignment = monotonic_alignment_search(similarity)

    return alignment, similarity


import torch


def viterbi_vectorized_alignment(similarity_matrix):
    b, nt, mel_len = similarity_matrix.shape
    device = similarity_matrix.device

    path_prob = torch.zeros((b, nt, mel_len), device=device)
    path_prob[:, 0, 0] = similarity_matrix[:, 0, 0]

    for n in range(nt):
        if n > 0:

            path_prob[:, n, 0] = path_prob[:, n - 1, 0] + similarity_matrix[:, n, 0]

        for t in range(1, mel_len):
            if n == 0:
                path_prob[:, 0, t] = path_prob[:, 0, t - 1] + similarity_matrix[:, 0, t]
            else:
                path_prob[:, n, t] = similarity_matrix[:, n, t] + torch.maximum(
                    path_prob[:, n - 1, t], path_prob[:, n, t - 1]
                )

    alignments = torch.zeros_like(similarity_matrix)

    boundaries = torch.zeros((b, nt), dtype=torch.long, device=device)

    for i in range(b):
        curr_mel_idx = mel_len - 1

        for n in range(nt - 1, -1, -1):
            cost_right = path_prob[i, n, curr_mel_idx]

            if n == 0:
                boundary_idx = 0
            else:

                costs = path_prob[i, n, : curr_mel_idx + 1]
                gradient = costs[1:] - costs[:-1]
                boundary_candidates = torch.where(gradient > 0)[0]

                if len(boundary_candidates) > 0:
                    boundary_idx = boundary_candidates[-1].item()
                else:
                    boundary_idx = 0

            alignments[i, n, boundary_idx : curr_mel_idx + 1] = 1
            boundaries[i, n] = boundary_idx
            curr_mel_idx = boundary_idx - 1

            if curr_mel_idx < 0:
                break

    return alignments


def windowed_monotonic_alignment(similarity_matrix, window_size=0.2):
    b, nt, mel_len = similarity_matrix.shape
    device = similarity_matrix.device

    alignments = torch.zeros_like(similarity_matrix)

    actual_window = max(2, int(mel_len * window_size))

    for i in range(b):

        frames_per_phone = mel_len / nt

        start_idx = 0

        for n in range(nt - 1):

            expected_end = int((n + 1) * frames_per_phone)

            window_start = max(start_idx, expected_end - actual_window)
            window_end = min(mel_len - 1, expected_end + actual_window)

            sub_scores = similarity_matrix[i, n, window_start : window_end + 1]
            best_offset = torch.argmax(sub_scores).item()
            best_end = window_start + best_offset

            alignments[i, n, start_idx : best_end + 1] = 1

            start_idx = best_end + 1

            if start_idx >= mel_len:
                break

        if start_idx < mel_len:
            alignments[i, -1, start_idx:] = 1

    return alignments


def progressive_monotonic_alignment(similarity_matrix):
    b, nt, mel_len = similarity_matrix.shape
    device = similarity_matrix.device

    alignments = torch.zeros_like(similarity_matrix)

    for i in range(b):

        boundaries = torch.linspace(0, mel_len, nt + 1).long()

        for n in range(nt):
            start = boundaries[n]
            end = boundaries[n + 1]
            if start < end:
                alignments[i, n, start:end] = 1

    refinement_steps = 2

    total_score = torch.sum(similarity_matrix * alignments)

    for _ in range(refinement_steps):
        for i in range(b):
            for n in range(nt - 1):

                boundary = None
                for t in range(mel_len):
                    if (
                        t < mel_len - 1
                        and alignments[i, n, t] == 1
                        and alignments[i, n, t + 1] == 0
                    ):
                        boundary = t
                        break

                if boundary is not None:

                    shift_range = min(5, mel_len // 20)
                    best_score = total_score
                    best_shift = 0

                    for shift in range(-shift_range, shift_range + 1):
                        new_boundary = boundary + shift
                        if 0 <= new_boundary < mel_len - 1:

                            test_alignment = alignments[i].clone()

                            if shift < 0:
                                test_alignment[n, new_boundary + 1 : boundary + 1] = 0
                                test_alignment[
                                    n + 1, new_boundary + 1 : boundary + 1
                                ] = 1
                            elif shift > 0:
                                test_alignment[n, boundary + 1 : new_boundary + 1] = 1
                                test_alignment[
                                    n + 1, boundary + 1 : new_boundary + 1
                                ] = 0

                            new_score = torch.sum(similarity_matrix[i] * test_alignment)

                            if new_score > best_score:
                                best_score = new_score
                                best_shift = shift

                    if best_shift != 0:
                        new_boundary = boundary + best_shift
                        if best_shift < 0:
                            alignments[i, n, new_boundary + 1 : boundary + 1] = 0
                            alignments[i, n + 1, new_boundary + 1 : boundary + 1] = 1
                        else:
                            alignments[i, n, boundary + 1 : new_boundary + 1] = 1
                            alignments[i, n + 1, boundary + 1 : new_boundary + 1] = 0

                        total_score = best_score

    return alignments


def monotonic_alignment_search(similarity_matrix, algorithm="viterbi"):
    """
    Hàm gọi một trong các giải thuật căn chỉnh monotonic dựa trên tham số.

    Args:
        similarity_matrix: Ma trận tương đồng [b, nt, mel_len]
        algorithm: Tên giải thuật ('viterbi', 'window', hoặc 'progressive')

    Returns:
        alignment: Ma trận căn chỉnh [b, nt, mel_len]
    """
    if algorithm == "viterbi":
        return viterbi_vectorized_alignment(similarity_matrix)
    elif algorithm == "window":
        return windowed_monotonic_alignment(similarity_matrix)
    elif algorithm == "progressive":
        return progressive_monotonic_alignment(similarity_matrix)
    else:
        raise ValueError(
            f"Không hỗ trợ giải thuật: {algorithm}. Chọn một trong các giải thuật: 'viterbi', 'window', 'progressive'"
        )


import torch
import numpy as np
import math


class AlignmentMethodManager:
    """
    Simple alignment method manager that transitions based on training phase and epoch.
    """

    def __init__(self):

        self.current_method = "window"
        self.phase = 1

        self.initial_dur_weight = 0.5
        self.target_dur_weight = 0.1
        self.decay_epochs = 10
        self.max_decay_steps = None

        self.viterbi_start_epoch = 3

    def set_steps_per_epoch(self, steps_per_epoch):
        """
        Sets the max_decay_steps based on actual steps per epoch.
        """
        self.max_decay_steps = steps_per_epoch * self.decay_epochs
        return self.max_decay_steps

    def should_transition_to_phase2(self, global_update, duration_focus_updates):
        """
        Check if we should transition from Phase 1 to Phase 2.
        """
        if global_update >= duration_focus_updates:
            return True, f"Reached duration focus updates: {duration_focus_updates}"
        return False, "Continuing Phase 1"

    def transition_to_phase2(self):
        """
        Transition from Phase 1 to Phase 2.
        """
        self.phase = 2
        self.current_method = "window"
        return f"Transitioned to Phase 2 with Window alignment method"

    def should_switch_to_viterbi(self, current_epoch):
        """
        Check if we should switch to Viterbi based on current epoch.
        """
        if self.phase != 2 or self.current_method == "viterbi":
            return False, "Not in Phase 2 or already using Viterbi"

        if current_epoch >= self.viterbi_start_epoch:
            return (
                True,
                f"Reached epoch {current_epoch} (threshold: {self.viterbi_start_epoch})",
            )

        return (
            False,
            f"Current epoch {current_epoch} hasn't reached threshold {self.viterbi_start_epoch}",
        )

    def switch_to_viterbi(self):
        """
        Switch from Window to Viterbi.
        """
        self.current_method = "viterbi"
        return f"Switched to Viterbi alignment method"

    def calculate_duration_weight(self, steps_in_phase2, current_epoch=None):
        """
        Calculate duration weight using cosine decay.
        """
        if self.phase == 1:
            return self.initial_dur_weight

        steps = min(steps_in_phase2, self.max_decay_steps)

        cosine_decay = 0.5 * (1 + math.cos(math.pi * steps / self.max_decay_steps))
        weight = (
            self.target_dur_weight
            + (self.initial_dur_weight - self.target_dur_weight) * cosine_decay
        )

        return weight


def get_alignment_method(
    manager,
    global_update,
    duration_focus_updates=12000,
    phase2_start_update=None,
    current_epoch=None,
):
    """
    Determine alignment method based on training phase and epoch.
    """
    logs = {
        "phase": manager.phase,
        "method": manager.current_method,
    }

    if manager.phase == 1:
        should_transition, reason = manager.should_transition_to_phase2(
            global_update, duration_focus_updates
        )
        if should_transition:
            manager.transition_to_phase2()
            logs["phase_transition"] = True
            logs["transition_reason"] = reason

    if manager.phase == 2 and current_epoch is not None:
        should_switch, reason = manager.should_switch_to_viterbi(current_epoch)
        if should_switch:
            manager.switch_to_viterbi()
            logs["method_switch"] = True
            logs["switch_reason"] = reason

    if manager.phase == 2 and phase2_start_update is not None:
        steps_in_phase2 = global_update - phase2_start_update
        duration_weight = manager.calculate_duration_weight(
            steps_in_phase2, current_epoch=current_epoch
        )
        logs["duration_weight"] = duration_weight
    else:
        logs["duration_weight"] = manager.initial_dur_weight

    return manager.current_method, logs
