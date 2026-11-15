"""
Digital Signal Processing (DSP) & Feature Extraction Pipeline.

Implements:
1. Sliding window segmentation (2-second windows, 50% overlap, 100-200 samples).
2. Time-domain feature extractors:
   - Mean, Standard Deviation, Peak-to-Peak (PTP), RMS, Skewness, Kurtosis, Zero-Crossing Rate (ZCR).
3. Frequency-domain feature extractors:
   - Fast Fourier Transform (FFT) power bands:
     * Band 1: 0 - 5 Hz (Wind flutter & ambient mesh sway)
     * Band 2: 5 - 15 Hz (Animal disturbances & incidental wildlife contact)
     * Band 3: 15 - 50 Hz (Human structural contact: climbing, scaling, wire-cutting)
   - Dominant frequency & Spectral centroid.
4. Sustained mass loading displacement (Delta-Gz).
5. High-speed vectorized computation (<2 ms execution latency per window).
"""

from typing import Dict, List, Tuple
import time
import numpy as np

FEATURE_NAMES = (
    # Time-Domain: Mean
    "mean_x",
    "mean_y",
    "mean_z",
    # Time-Domain: Standard Deviation
    "std_x",
    "std_y",
    "std_z",
    # Time-Domain: Peak-to-Peak (PTP)
    "ptp_x",
    "ptp_y",
    "ptp_z",
    # Time-Domain: RMS
    "rms_x",
    "rms_y",
    "rms_z",
    "rms_mag",
    # Time-Domain: Skewness
    "skewness_z",
    "skewness_mag",
    # Time-Domain: Kurtosis (Fisher / Excess Kurtosis)
    "kurtosis_z",
    "kurtosis_mag",
    # Time-Domain: Zero-Crossing Rate (ZCR)
    "zcr_x",
    "zcr_y",
    "zcr_z",
    # Frequency-Domain: FFT Power Bands
    "band1_ratio",  # 0 - 5 Hz
    "band2_ratio",  # 5 - 15 Hz
    "band3_ratio",  # 15 - 50 Hz
    # Frequency-Domain: Dominant frequency & Spectral Centroid
    "dominant_freq",
    "spectral_centroid",
    # Static preload displacement
    "delta_gz",
)


def compute_rms(signal: np.ndarray) -> float:
    """Computes Root Mean Square (RMS) of a 1D array: sqrt(mean(x^2))."""
    if len(signal) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(signal))))


def compute_kurtosis(signal: np.ndarray, fisher: bool = True) -> float:
    """
    Computes vectorized Kurtosis.
    If fisher=True (default), excess kurtosis is returned (normal distribution = 0.0).
    If fisher=False, Pearson kurtosis is returned (normal distribution = 3.0).
    """
    n = len(signal)
    if n == 0:
        return 0.0
    diff = signal - np.mean(signal)
    var = np.mean(diff ** 2)
    if var < 1e-12:
        return 0.0 if fisher else 3.0
    kurt = np.mean(diff ** 4) / (var ** 2)
    return float(kurt - 3.0 if fisher else kurt)


def compute_skewness(signal: np.ndarray) -> float:
    """Computes sample skewness (3rd standardized moment)."""
    n = len(signal)
    if n == 0:
        return 0.0
    diff = signal - np.mean(signal)
    var = np.mean(diff ** 2)
    if var < 1e-12:
        return 0.0
    skew = np.mean(diff ** 3) / (var ** 1.5)
    return float(skew)


def compute_zcr(signal: np.ndarray) -> int:
    """Computes Zero-Crossing Rate (number of sign changes) on a mean-centered signal."""
    if len(signal) < 2:
        return 0
    centered = signal - np.mean(signal)
    signs = np.signbit(centered)
    crossings = np.count_nonzero(signs[:-1] != signs[1:])
    return int(crossings)


def compute_frequency_features(
    signal: np.ndarray,
    fs: float = 100.0,
    band1_range: Tuple[float, float] = (0.0, 5.0),
    band2_range: Tuple[float, float] = (5.0, 15.0),
    band3_range: Tuple[float, float] = (15.0, 50.0),
) -> Tuple[float, float, float, float, float]:
    """
    Computes:
    - Band 1 energy ratio (0 - 5 Hz)
    - Band 2 energy ratio (5 - 15 Hz)
    - Band 3 energy ratio (15 - 50 Hz)
    - Dominant peak frequency (Hz)
    - Spectral centroid (Hz)
    """
    n = len(signal)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Mean-center to remove DC spike before computing vibration power
    detrended = signal - np.mean(signal)

    # Windowing with Hanning window to prevent spectral leakage
    window = np.hanning(n)
    fft_vals = np.fft.rfft(detrended * window)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    magnitude_spectrum = np.abs(fft_vals)
    power_spectrum = magnitude_spectrum ** 2

    total_power = np.sum(power_spectrum)
    if total_power < 1e-12:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Band ratios
    mask1 = (freqs >= band1_range[0]) & (freqs < band1_range[1])
    mask2 = (freqs >= band2_range[0]) & (freqs < band2_range[1])
    mask3 = (freqs >= band3_range[0]) & (freqs <= band3_range[1])

    b1_ratio = float(np.sum(power_spectrum[mask1]) / total_power)
    b2_ratio = float(np.sum(power_spectrum[mask2]) / total_power)
    b3_ratio = float(np.sum(power_spectrum[mask3]) / total_power)

    # Dominant frequency
    peak_idx = int(np.argmax(magnitude_spectrum))
    dominant_freq = float(freqs[peak_idx])

    # Spectral Centroid: sum(f * |X(f)|) / sum(|X(f)|)
    mag_sum = np.sum(magnitude_spectrum)
    if mag_sum < 1e-12:
        spectral_centroid = 0.0
    else:
        spectral_centroid = float(np.sum(freqs * magnitude_spectrum) / mag_sum)

    return b1_ratio, b2_ratio, b3_ratio, dominant_freq, spectral_centroid


def extract_features(data: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """
    Extracts the full feature vector from a (N, 3) acceleration window [ax, ay, az].
    Returns a 1D numpy array of shape (len(FEATURE_NAMES),).
    """
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"Expected input of shape (N, 3), got {data.shape}")

    ax = data[:, 0]
    ay = data[:, 1]
    az = data[:, 2]
    mag = np.sqrt(ax**2 + ay**2 + az**2)

    # Time-Domain: Mean
    mean_x = float(np.mean(ax))
    mean_y = float(np.mean(ay))
    mean_z = float(np.mean(az))

    # Time-Domain: Std Dev
    std_x = float(np.std(ax))
    std_y = float(np.std(ay))
    std_z = float(np.std(az))

    # Time-Domain: Peak-to-Peak
    ptp_x = float(np.ptp(ax))
    ptp_y = float(np.ptp(ay))
    ptp_z = float(np.ptp(az))

    # Time-Domain: RMS
    rms_x = compute_rms(ax)
    rms_y = compute_rms(ay)
    rms_z = compute_rms(az)
    rms_mag = compute_rms(mag)

    # Time-Domain: Skewness
    skew_z = compute_skewness(az)
    skew_mag = compute_skewness(mag)

    # Time-Domain: Kurtosis (Fisher excess kurtosis)
    kurt_z = compute_kurtosis(az, fisher=True)
    kurt_mag = compute_kurtosis(mag, fisher=True)

    # Time-Domain: Zero-Crossing Rate
    zcr_x = float(compute_zcr(ax))
    zcr_y = float(compute_zcr(ay))
    zcr_z = float(compute_zcr(az))

    # Frequency-Domain: FFT bands & centroid
    b1, b2, b3, dom_freq, centroid = compute_frequency_features(az, fs=fs)

    # Static preload displacement
    delta_gz = float(abs(mean_z - 1.0))

    return np.array([
        mean_x, mean_y, mean_z,
        std_x, std_y, std_z,
        ptp_x, ptp_y, ptp_z,
        rms_x, rms_y, rms_z, rms_mag,
        skew_z, skew_mag,
        kurt_z, kurt_mag,
        zcr_x, zcr_y, zcr_z,
        b1, b2, b3,
        dom_freq,
        centroid,
        delta_gz,
    ], dtype=np.float64)


def extract_features_dict(data: np.ndarray, fs: float = 100.0) -> Dict[str, float]:
    """Extracts features and returns a key-value dictionary."""
    vec = extract_features(data, fs=fs)
    return {name: float(val) for name, val in zip(FEATURE_NAMES, vec)}


def slice_windows(data: np.ndarray, window_size: int = 200, overlap: int = 100) -> List[np.ndarray]:
    """
    Slices raw (M, 3) time-series data into sliding windows of size window_size with overlap.
    Default: 200 samples (2.0s at 100 Hz), 50% overlap (100 samples).
    """
    step = window_size - overlap
    if len(data) < window_size:
        return []
    windows = []
    for start in range(0, len(data) - window_size + 1, step):
        windows.append(data[start : start + window_size])
    return windows


def benchmark_extraction_latency(num_runs: int = 500, window_size: int = 200) -> Dict[str, float]:
    """
    Benchmarks feature extraction latency per window in milliseconds.
    Target: < 2.0 ms per window.
    """
    rng = np.random.default_rng(42)
    sample_window = rng.normal(0, 0.1, size=(window_size, 3))
    sample_window[:, 2] += 1.0

    # Warmup
    for _ in range(20):
        _ = extract_features(sample_window)

    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = extract_features(sample_window)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    latencies = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "num_runs": num_runs,
        "target_met": bool(np.mean(latencies) < 2.0),
    }


if __name__ == "__main__":
    print("=" * 60)
    print("AI-Smart-Fence: DSP Feature Extraction Benchmark")
    print("=" * 60)
    res = benchmark_extraction_latency(num_runs=1000)
    print(f"Number of runs   : {res['num_runs']}")
    print(f"Mean Latency     : {res['mean_ms']:.4f} ms")
    print(f"Median Latency   : {res['median_ms']:.4f} ms")
    print(f"95th Percentile  : {res['p95_ms']:.4f} ms")
    print(f"99th Percentile  : {res['p99_ms']:.4f} ms")
    print(f"Target (< 2.0 ms): {'PASSED' if res['target_met'] else 'FAILED'}")
    print("=" * 60)
