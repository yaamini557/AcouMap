"""
AcouMap - Acoustic Feature Extraction Engine
Extracts 8+ physical acoustic features per time-window using SciPy & NumPy:
- MFCC (configurable 13 coefficients standard)
- RMS Energy
- Zero Crossing Rate (ZCR)
- Spectral Centroid
- Spectral Bandwidth
- Spectral Rolloff (85% energy threshold)
- Spectral Flux
- Dominant Peak Frequency
"""

import numpy as np
from scipy.io import wavfile
from scipy.fft import dct, rfft, rfftfreq
from typing import Dict, List, Tuple, Any, Optional
import os


def hz_to_mel(hz: np.ndarray) -> np.ndarray:
    """Convert Frequency in Hz to Mel scale."""
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: np.ndarray) -> np.ndarray:
    """Convert Mel scale to Frequency in Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def get_mel_filterbank(num_filters: int, n_fft: int, sample_rate: int,
                       f_min: float = 20.0, f_max: Optional[float] = None) -> np.ndarray:
    """Compute triangular Mel filterbank matrix."""
    if f_max is None:
        f_max = sample_rate / 2.0

    mel_min = hz_to_mel(np.array([f_min]))[0]
    mel_max = hz_to_mel(np.array([f_max]))[0]
    mel_points = np.linspace(mel_min, mel_max, num_filters + 2)
    hz_points = mel_to_hz(mel_points)

    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    num_bins = n_fft // 2 + 1
    filterbank = np.zeros((num_filters, num_bins))

    for i in range(1, num_filters + 1):
        left = bin_points[i - 1]
        center = bin_points[i]
        right = bin_points[i + 1]

        if center > left:
            filterbank[i - 1, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            filterbank[i - 1, center:right] = (right - np.arange(center, right)) / (right - center)

    return filterbank


class AcousticFeatureExtractor:
    """
    Modular acoustic feature extractor for industrial fan audio.
    Supports configurable window length, hop length, and MFCC counts.
    """

    def __init__(self,
                 n_mfcc: int = 13,
                 window_sec: float = 1.0,
                 hop_sec: float = 0.5,
                 target_sr: int = 16000,
                 n_fft: int = 1024):
        self.n_mfcc = n_mfcc
        self.window_sec = window_sec
        self.hop_sec = hop_sec
        self.target_sr = target_sr
        self.n_fft = n_fft
        self.mel_fb = get_mel_filterbank(num_filters=max(26, n_mfcc * 2),
                                         n_fft=n_fft,
                                         sample_rate=target_sr)

    def get_feature_names(self) -> List[str]:
        """Returns ordered list of feature vector dimension names."""
        names = [f"mfcc_{i+1}" for i in range(self.n_mfcc)]
        names.extend([
            "rms",
            "zcr",
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_rolloff",
            "spectral_flux",
            "dominant_frequency"
        ])
        return names

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio from WAV file, convert to mono float32 in [-1.0, 1.0],
        resample or downsample if necessary.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        sr, data = wavfile.read(file_path)

        # Handle bit depths
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128.0) / 128.0
        elif data.dtype in (np.float32, np.float64):
            data = data.astype(np.float32)

        # Convert stereo to mono
        if len(data.shape) > 1 and data.shape[1] > 1:
            data = np.mean(data, axis=1)

        # Resample if sample rate differs from target_sr
        if sr != self.target_sr:
            # Resample using linear interpolation for pure-Python reliability
            orig_len = len(data)
            target_len = int(orig_len * (self.target_sr / sr))
            resampled_indices = np.linspace(0, orig_len - 1, target_len)
            data = np.interp(resampled_indices, np.arange(orig_len), data).astype(np.float32)
            sr = self.target_sr

        return data, sr

    def extract_window_features(self, window: np.ndarray, prev_mag: Optional[np.ndarray] = None) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """
        Extract numerical features from a single audio window.
        Returns (feature_dict, feature_vector, current_magnitude_spectrum).
        """
        eps = 1e-10
        N = len(window)

        # 1. RMS Energy
        rms = float(np.sqrt(np.mean(window ** 2) + eps))

        # 2. Zero Crossing Rate (ZCR)
        signs = np.sign(window)
        # Handle zero signs
        signs[signs == 0] = 1
        zcr = float(np.mean(np.abs(signs[1:] - signs[:-1])) / 2.0)

        # 3. FFT Magnitude and Power Spectrum
        # Apply Hanning window
        hann = np.hanning(min(N, self.n_fft))
        # Zero-pad or truncate to n_fft
        if N < self.n_fft:
            padded = np.zeros(self.n_fft, dtype=np.float32)
            padded[:N] = window * np.hanning(N)
            window_framed = padded
        else:
            window_framed = window[:self.n_fft] * hann

        mag = np.abs(rfft(window_framed))
        power = (mag ** 2) / float(self.n_fft)
        freqs = rfftfreq(self.n_fft, 1.0 / self.target_sr)

        # 4. Spectral Centroid
        mag_sum = np.sum(mag) + eps
        centroid = float(np.sum(freqs * mag) / mag_sum)

        # 5. Spectral Bandwidth
        bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / mag_sum))

        # 6. Spectral Rolloff (85% cumulative energy)
        cum_power = np.cumsum(power)
        rolloff_threshold = 0.85 * cum_power[-1]
        rolloff_idx = np.searchsorted(cum_power, rolloff_threshold)
        rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

        # 7. Spectral Flux (difference from previous window frame if available)
        if prev_mag is not None and len(prev_mag) == len(mag):
            # Normalized difference
            p_norm = prev_mag / (np.sum(prev_mag) + eps)
            c_norm = mag / (np.sum(mag) + eps)
            flux = float(np.sum((c_norm - p_norm) ** 2))
        else:
            flux = 0.0

        # 8. Dominant Frequency
        dom_idx = int(np.argmax(mag))
        dominant_freq = float(freqs[dom_idx])

        # 9. Mel-Frequency Cepstral Coefficients (MFCC)
        mel_energy = np.dot(self.mel_fb, power)
        log_mel_energy = np.log(mel_energy + eps)
        # Type-II DCT
        mfcc_all = dct(log_mel_energy, type=2, norm='ortho')
        mfccs = [float(mfcc_all[i]) for i in range(self.n_mfcc)]

        feat_dict: Dict[str, float] = {}
        for i, val in enumerate(mfccs):
            feat_dict[f"mfcc_{i+1}"] = val

        feat_dict.update({
            "rms": rms,
            "zcr": zcr,
            "spectral_centroid": centroid,
            "spectral_bandwidth": bandwidth,
            "spectral_rolloff": rolloff,
            "spectral_flux": flux,
            "dominant_frequency": dominant_freq
        })

        vector = np.array([feat_dict[name] for name in self.get_feature_names()], dtype=np.float32)
        return feat_dict, vector, mag

    def process_file(self, file_path: str) -> Tuple[np.ndarray, List[Dict[str, Any]], float]:
        """
        Process an entire audio file into sliding windows.
        Returns:
            feature_matrix: (num_windows, num_features)
            windows_metadata: list of window dicts with start_time, end_time, features
            duration_sec: total duration
        """
        audio, sr = self.load_audio(file_path)
        total_samples = len(audio)
        duration_sec = float(total_samples / sr)

        window_samples = int(self.window_sec * sr)
        hop_samples = int(self.hop_sec * sr)

        if total_samples < window_samples:
            # Pad short audio
            pad_width = window_samples - total_samples
            audio = np.pad(audio, (0, pad_width), mode='reflect')
            total_samples = len(audio)

        windows_metadata: List[Dict[str, Any]] = []
        vectors: List[np.ndarray] = []

        start = 0
        w_idx = 0
        prev_mag = None

        while start + window_samples <= total_samples:
            end = start + window_samples
            window = audio[start:end]

            start_time = float(start / sr)
            end_time = float(end / sr)

            feat_dict, vec, curr_mag = self.extract_window_features(window, prev_mag=prev_mag)
            prev_mag = curr_mag

            meta = {
                "window_index": w_idx,
                "start_time": round(start_time, 3),
                "end_time": round(end_time, 3),
                "features": feat_dict
            }
            windows_metadata.append(meta)
            vectors.append(vec)

            w_idx += 1
            start += hop_samples

        if len(vectors) == 0:
            # Handle edge case: single window
            feat_dict, vec, _ = self.extract_window_features(audio[:window_samples])
            windows_metadata.append({
                "window_index": 0,
                "start_time": 0.0,
                "end_time": round(float(window_samples / sr), 3),
                "features": feat_dict
            })
            vectors.append(vec)

        feature_matrix = np.vstack(vectors)
        return feature_matrix, windows_metadata, duration_sec
