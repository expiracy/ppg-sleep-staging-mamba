import numpy as np
import torch
import torch.nn.functional as F

# Each noise type targets a real PPG artifact source:
# gaussian: sensor electronics noise
# drift: baseline wander from respiration and posture shifts
# spikes: motion artifacts (sudden light-path changes)
# emg: muscle movement interference, fixed amplitude with no config key
DEFAULT_NOISE_CONFIG = {
    "noise_level": 0.1,
    "drift_amplitude": 0.1,
    "drift_frequency": 0.1,
    "spike_probability": 0.01,
    "spike_amplitude": 0.5,
}


def add_noise_to_ppg(clean_ppg, noise_config):
    """Corrupt a clean PPG batch to simulate an unfiltered wearable recording.
    Sourced from https://github.com/DavyWJW/sleep-staging-models/"""
    batch_size, _, length = clean_ppg.shape
    device = clean_ppg.device

    noisy_ppg = clean_ppg.clone()

    # Gaussian noise
    gaussian_noise = torch.randn_like(clean_ppg) * noise_config["noise_level"]
    noisy_ppg = noisy_ppg + gaussian_noise

    # Baseline drift: t normalised to [0,1] so drift_frequency is in
    # cycles per recording, independent of length
    t = torch.linspace(0, 1, length, device=device)
    drift_freq = noise_config["drift_frequency"]
    drift_amp = noise_config["drift_amplitude"]

    # Three harmonics (1x, 2x, 0.5x) with weights summing to 1, so the drift is
    # not a single clean sinusoid
    drift = drift_amp * (
        0.5 * torch.sin(2 * np.pi * drift_freq * t)
        + 0.3 * torch.sin(2 * np.pi * drift_freq * 2 * t)
        + 0.2 * torch.sin(2 * np.pi * drift_freq * 0.5 * t)
    )
    drift = drift.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
    noisy_ppg = noisy_ppg + drift

    # Smoothed motion artifact spikes
    spike_prob = noise_config["spike_probability"]
    spike_amp = noise_config["spike_amplitude"]

    spike_mask = torch.rand(batch_size, 1, length, device=device) < spike_prob
    spike_values = torch.randn(batch_size, 1, length, device=device) * spike_amp
    spikes = spike_mask.float() * spike_values

    # Smooth the impulses because real motion artifacts have finite rise time
    kernel_size = 5
    padding = kernel_size // 2
    smoothing_kernel = torch.ones(1, 1, kernel_size, device=device) / kernel_size
    spikes = F.conv1d(spikes, smoothing_kernel, padding=padding)

    noisy_ppg = noisy_ppg + spikes

    # EMG (muscle movement) interference
    emg_noise = torch.randn_like(clean_ppg) * 0.05
    noisy_ppg = noisy_ppg + emg_noise

    return noisy_ppg
