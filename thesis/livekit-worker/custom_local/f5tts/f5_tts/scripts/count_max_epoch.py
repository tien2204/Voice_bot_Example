"""ADAPTIVE BATCH SIZE"""

print("Adaptive batch size: using grouping batch sampler, frames_per_gpu fixed fed in")
print("  -> least padding, gather wavs with accumulated frames in a batch\n")


total_hours = 95282
mel_hop_length = 256
mel_sampling_rate = 24000


wanted_max_updates = 1200000


gpus = 8
frames_per_gpu = 38400
grad_accum = 1


mini_batch_frames = frames_per_gpu * grad_accum * gpus
mini_batch_hours = mini_batch_frames * mel_hop_length / mel_sampling_rate / 3600
updates_per_epoch = total_hours / mini_batch_hours


epochs = wanted_max_updates / updates_per_epoch
print(
    f"epochs should be set to: {epochs:.0f} ({epochs / grad_accum:.1f} x gd_acum {grad_accum})"
)
print(f"progress_bar should show approx. 0/{updates_per_epoch:.0f} updates")


print(f"total {total_hours:.0f} hours")
print(
    f"mini-batch of {mini_batch_frames:.0f} frames, {mini_batch_hours:.2f} hours per mini-batch"
)
