"""
AcouMap - Synthetic Fan Audio Generator
Generates realistic physics-based industrial fan acoustic recordings with a fixed random seed.

Includes:
- Normal conditions (stable rotational BPF, stable amplitude, aerodynamic turbulence, normal harmonics)
- Abnormal conditions:
  * increased_broadband_noise
  * frequency_shift
  * harmonic_distortion
  * sudden_amplitude_surge
  * impulsive_bearing_impacts
  * high_frequency_rubbing
  * periodic_tonal_modulation

All outputs are explicitly tagged as 'SYNTHETIC / DEMO DATA'.
"""

import numpy as np
from scipy.io import wavfile
import os
from typing import Tuple, Optional


def generate_pink_noise(num_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Generate 1/f pink noise approximation using Paul Kellet's filter."""
    white = rng.standard_normal(num_samples)
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = 0.0
    pink = np.zeros(num_samples, dtype=np.float32)

    for i in range(num_samples):
        w = white[i]
        b0 = 0.99886 * b0 + w * 0.0555179
        b1 = 0.99332 * b1 + w * 0.0750759
        b2 = 0.96900 * b2 + w * 0.1538520
        b3 = 0.86650 * b3 + w * 0.3104856
        b4 = 0.55000 * b4 + w * 0.5329522
        b5 = -0.7616 * b5 - w * 0.0168980
        pink[i] = b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362
        b6 = w * 0.115926

    # Normalize pink noise
    max_val = np.max(np.abs(pink))
    if max_val > 0:
        pink = pink / max_val
    return pink


class SyntheticFanAudioGenerator:
    """
    Generates deterministic, physically grounded synthetic fan acoustic recordings.
    """

    def __init__(self, sample_rate: int = 16000, seed: int = 42):
        self.sample_rate = sample_rate
        self.seed = seed

    def generate_recording(self,
                           duration_sec: float = 10.0,
                           condition: str = "normal",
                           base_rpm: float = 1800.0,
                           num_blades: int = 4,
                           anomaly_start_sec: float = 3.5,
                           anomaly_duration_sec: float = 3.0,
                           noise_level: float = 0.12,
                           recording_seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """
        Generate synthetic fan audio data.

        condition can be:
        - 'normal'
        - 'increased_broadband_noise'
        - 'frequency_shift'
        - 'harmonic_distortion'
        - 'sudden_amplitude_surge'
        - 'impulsive_bearing_impacts'
        - 'high_frequency_rubbing'
        - 'combined_abnormal'
        """
        seed = recording_seed if recording_seed is not None else self.seed
        rng = np.random.default_rng(seed)

        total_samples = int(duration_sec * self.sample_rate)
        t = np.linspace(0, duration_sec, total_samples, endpoint=False)

        # Fundamental Blade Pass Frequency: BPF = (RPM / 60) * num_blades
        nominal_bpf = (base_rpm / 60.0) * num_blades  # 120 Hz for 1800 RPM, 4 blades

        # Baseline fan sound: fundamental BPF tone + 1st, 2nd, 3rd harmonics + aerodynamic turbulence
        bpf_signal = 0.45 * np.sin(2.0 * np.pi * nominal_bpf * t)
        h1_signal = 0.22 * np.sin(2.0 * np.pi * (nominal_bpf * 2.0) * t)
        h2_signal = 0.12 * np.sin(2.0 * np.pi * (nominal_bpf * 3.0) * t)
        h3_signal = 0.06 * np.sin(2.0 * np.pi * (nominal_bpf * 4.0) * t)

        base_fan = bpf_signal + h1_signal + h2_signal + h3_signal

        # Aerodynamic turbulence pink noise
        turbulence = noise_level * generate_pink_noise(total_samples, rng)

        # Baseline combined signal
        audio = base_fan + turbulence

        # Build anomaly mask
        is_anomalous = False
        anomaly_mask = np.zeros(total_samples, dtype=bool)

        if condition != "normal":
            is_anomalous = True
            start_idx = int(anomaly_start_sec * self.sample_rate)
            end_idx = min(total_samples, int((anomaly_start_sec + anomaly_duration_sec) * self.sample_rate))
            anomaly_mask[start_idx:end_idx] = True
            t_anom = t[start_idx:end_idx]
            n_anom = len(t_anom)

            if condition == "increased_broadband_noise":
                # High turbulence flow separation / casing air turbulence
                extra_noise = 0.45 * rng.standard_normal(n_anom)
                audio[start_idx:end_idx] += extra_noise

            elif condition == "frequency_shift":
                # Motor slip / rotational frequency shift
                shifted_bpf = nominal_bpf * 1.35  # Shift to ~162 Hz
                shifted_tone = 0.6 * np.sin(2.0 * np.pi * shifted_bpf * t_anom)
                audio[start_idx:end_idx] = 0.4 * audio[start_idx:end_idx] + shifted_tone

            elif condition == "harmonic_distortion":
                # Non-linear oscillation creating rich abnormal odd/even harmonics
                distorted = np.tanh(2.5 * audio[start_idx:end_idx]) * 0.7
                sub_harmonic = 0.3 * np.sin(2.0 * np.pi * (nominal_bpf * 0.5) * t_anom)
                audio[start_idx:end_idx] = distorted + sub_harmonic

            elif condition == "sudden_amplitude_surge":
                # Sudden transient surge in aerodynamic energy
                surge_envelope = 1.0 + 1.8 * np.sin(np.pi * np.linspace(0, 1, n_anom))
                audio[start_idx:end_idx] *= surge_envelope

            elif condition == "impulsive_bearing_impacts":
                # Repetitive impact pulses (e.g., ball pass frequency outer race BPFO ~ 4.2x rotational speed)
                impact_rate = (base_rpm / 60.0) * 3.8  # ~114 Hz impacts
                impact_period = int(self.sample_rate / impact_rate)
                pulses = np.zeros(n_anom, dtype=np.float32)
                decay_len = int(0.005 * self.sample_rate)  # 5ms decay
                decay_kernel = np.exp(-np.linspace(0, 5, decay_len)) * np.sin(2.0 * np.pi * 3200.0 * np.linspace(0, 0.005, decay_len))

                for p in range(0, n_anom - decay_len, impact_period):
                    pulses[p:p + decay_len] += 0.7 * decay_kernel

                audio[start_idx:end_idx] += pulses

            elif condition == "high_frequency_rubbing":
                # High-frequency friction / blade-tip casing contact (3.5 kHz - 5 kHz acoustic activity)
                rubbing = 0.35 * np.sin(2.0 * np.pi * 3850.0 * t_anom) * (0.8 + 0.4 * rng.standard_normal(n_anom))
                audio[start_idx:end_idx] += rubbing

            elif condition == "combined_abnormal":
                # Realistic multi-feature industrial abnormality
                rubbing = 0.25 * np.sin(2.0 * np.pi * 3400.0 * t_anom)
                extra_turbulence = 0.3 * rng.standard_normal(n_anom)
                surge_envelope = 1.0 + 0.6 * np.sin(np.pi * np.linspace(0, 1, n_anom))
                audio[start_idx:end_idx] = (audio[start_idx:end_idx] * surge_envelope) + rubbing + extra_turbulence

        # Normalize audio to prevent clipping [-0.95, 0.95]
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = (audio / peak) * 0.92

        metadata = {
            "condition": condition,
            "data_type": "SYNTHETIC / DEMO DATA",
            "is_synthetic": True,
            "sample_rate": self.sample_rate,
            "duration_sec": duration_sec,
            "nominal_bpf": nominal_bpf,
            "base_rpm": base_rpm,
            "num_blades": num_blades,
            "is_anomalous": is_anomalous,
            "anomaly_start_sec": anomaly_start_sec if is_anomalous else None,
            "anomaly_duration_sec": anomaly_duration_sec if is_anomalous else None,
            "seed": seed
        }

        return audio, metadata

    def save_wav(self, audio: np.ndarray, file_path: str):
        """Save float32 audio as 16-bit PCM WAV."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        int16_data = (audio * 32767.0).astype(np.int16)
        wavfile.write(file_path, self.sample_rate, int16_data)
