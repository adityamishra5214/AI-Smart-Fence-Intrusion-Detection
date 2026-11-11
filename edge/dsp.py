"""
Sliding window segmentation, FFT, and statistical 14-D feature extraction module.
Computes RMS, Kurtosis, Zero-Crossing Rate, Band-Power Ratios, and Spectral Centroid.
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy import stats

FEATURE_NAMES = (
    "rms_x",
    "rms_y",
    "rms_z",
    "rms_mag",
    "kurtosis_z",
    "kurtosis_mag",
    "zcr_x",
    "zcr_y",
    "zcr_z",
    "band1_ratio",      # 0 - 5 Hz (Wind dominance)
    "band2_ratio",      # 5 - 15 Hz (Animal disturbances)
    "band3_ratio",      # 15 - 50 Hz (Human structural contact)
    "spectral_centroid", # Weighted average frequency
    "delta_gz",         # Mean vertical displacement |mean(az) - 1.0|
)


def compute_rms(signal: np.ndarray) -> float:
    """Computes Root Mean Square (RMS) of a 1D array."""
    if len(signal) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(signal))))


def compute_kurtosis(signal: np.ndarray) -> float:
    """
    Computes Pearson Kurtosis (normal distribution = 3.0).
    Formula: E[(x - mu)^4] / (E[(x - mu)^2])^2
    """
    if len(signal) == 0:
        return 0.0
    variance = np.var(signal)
    if variance < 1e-9:
        return 3.0  # Flatline or zero variance defaults to nominal normal baseline
    return float(stats.kurtosis(signal, fisher=False))


def compute_zcr(signal: np.ndarray) -> int:
    """Computes Zero-Crossing Rate (number of sign changes) on mean-centered signal."""
    if len(signal) < 2:
        return 0
    centered = signal - np.mean(signal)
    signs = np.signbit(centered)
    crossings = np.sum(signs[:-1] != signs[1:])
    return int(crossings)


def compute_spectral_features(
    signal: np.ndarray,
    fs: float = 100.0,
    band1_range: Tuple[float, float] = (0.0, 5.0),
    band2_range: Tuple[float, float] = (5.0, 15.0),
    band3_range: Tuple[float, float] = (15.0, 50.0),
) -> Tuple[float, float, float, float]:
    """
    Computes FFT energy ratios across Band 1, Band 2, Band 3, and the Spectral Centroid.
    Returns:
        (band1_ratio, band2_ratio, band3_ratio, spectral_centroid)
    """
    n = len(signal)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Mean-center to eliminate zero-frequency DC spike for spectral distribution
    detrended = signal - np.mean(signal)

    # Apply Hanning window to reduce spectral leakage
    window = np.hanning(n)
    fft_vals = np.fft.rfft(detrended * window)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    magnitude_spectrum = np.abs(fft_vals)
    power_spectrum = magnitude_spectrum ** 2

    total_power = np.sum(power_spectrum)
    if total_power < 1e-12:
        return 0.0, 0.0, 0.0, 0.0

    # Band masks
    mask1 = (freqs >= band1_range[0]) & (freqs < band1_range[1])
    mask2 = (freqs >= band2_range[0]) & (freqs < band2_range[1])
    mask3 = (freqs >= band3_range[0]) & (freqs <= band3_range[1])

    band1_ratio = float(np.sum(power_spectrum[mask1]) / total_power)
    band2_ratio = float(np.sum(power_spectrum[mask2]) / total_power)
    band3_ratio = float(np.sum(power_spectrum[mask3]) / total_power)

    # Spectral Centroid: sum(f * |X(f)|) / sum(|X(f)|)
    mag_sum = np.sum(magnitude_spectrum)
    if mag_sum < 1e-12:
        spectral_centroid = 0.0
    else:
        spectral_centroid = float(np.sum(freqs * magnitude_spectrum) / mag_sum)

    return band1_ratio, band2_ratio, band3_ratio, spectral_centroid


def extract_features(data: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """
    Extracts the complete 14-dimensional feature vector from a (N, 3) acceleration window [ax, ay, az].
    """
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"Expected shape (N, 3), got {data.shape}")

    ax = data[:, 0]
    ay = data[:, 1]
    az = data[:, 2]

    # Magnitude series
    mag = np.sqrt(ax**2 + ay**2 + az**2)

    # 1. RMS features
    rms_x = compute_rms(ax)
    rms_y = compute_rms(ay)
    rms_z = compute_rms(az)
    rms_mag = compute_rms(mag)

    # 2. Kurtosis features
    kurtosis_z = compute_kurtosis(az)
    kurtosis_mag = compute_kurtosis(mag)

    # 3. Zero-Crossing Rates
    zcr_x = float(compute_zcr(ax))
    zcr_y = float(compute_zcr(ay))
    zcr_z = float(compute_zcr(az))

    # 4. Spectral power bands & centroid (using vertical/magnitude dynamic vibration)
    # Using detrended vertical dynamic shock component (az) as primary structural indicator
    b1, b2, b3, centroid = compute_spectral_features(az, fs=fs)

    # 5. Sustained mass loading displacement
    delta_gz = float(abs(np.mean(az) - 1.0))

    return np.array([
        rms_x,
        rms_y,
        rms_z,
        rms_mag,
        kurtosis_z,
        kurtosis_mag,
        zcr_x,
        zcr_y,
        zcr_z,
        b1,
        b2,
        b3,
        centroid,
        delta_gz,
    ], dtype=np.float64)


def extract_features_dict(data: np.ndarray, fs: float = 100.0) -> Dict[str, float]:
    """Extracts features and returns as a named dictionary."""
    vec = extract_features(data, fs=fs)
    return {name: float(val) for name, val in zip(FEATURE_NAMES, vec)}


def slice_windows(data: np.ndarray, window_size: int = 200, overlap: int = 100) -> List[np.ndarray]:
    """
    Slices raw (M, 3) time-series data into sliding windows of size window_size with given overlap.
    """
    step = window_size - overlap
    if len(data) < window_size:
        return []
    windows = []
    for start in range(0, len(data) - window_size + 1, step):
        windows.append(data[start : start + window_size])
    return windows
