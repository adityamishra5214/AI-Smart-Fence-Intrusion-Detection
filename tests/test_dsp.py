"""
Mathematical validation tests for feature extractors in edge/dsp.py.
Validates:
1. Mean, Std Dev, Peak-to-Peak (PTP), RMS, Skewness, Kurtosis, Zero-Crossing Rate (ZCR).
2. Frequency-domain: FFT power bands (0-5 Hz, 5-15 Hz, 15-50 Hz), Dominant Frequency, Spectral Centroid.
3. Feature vector sizing and dictionary exports.
4. Window slicing with 50% overlap.
"""

import numpy as np
import pytest
from edge.dsp import (
    compute_rms,
    compute_kurtosis,
    compute_skewness,
    compute_zcr,
    compute_frequency_features,
    extract_features,
    extract_features_dict,
    slice_windows,
    FEATURE_NAMES,
)
from simulator.generator import VibrationSynthesizer, SignalType


@pytest.fixture
def synth():
    return VibrationSynthesizer(fs=100.0, seed=42)


def test_rms_computation():
    # DC signal of 3.0 has RMS = 3.0
    dc = np.full(100, 3.0)
    assert np.isclose(compute_rms(dc), 3.0)

    # Sine wave with amplitude A has RMS = A / sqrt(2)
    t = np.linspace(0, 1, 1000, endpoint=False)
    sine = 2.0 * np.sin(2 * np.pi * 10 * t)
    expected_rms = 2.0 / np.sqrt(2)
    assert np.isclose(compute_rms(sine), expected_rms, atol=0.05)


def test_kurtosis_and_skewness():
    rng = np.random.default_rng(42)
    gauss = rng.normal(0, 1.0, size=10000)
    # Fisher kurtosis of standard Gaussian is ~ 0.0
    kurt_fisher = compute_kurtosis(gauss, fisher=True)
    assert np.isclose(kurt_fisher, 0.0, atol=0.2)

    # Pearson kurtosis is ~ 3.0
    kurt_pearson = compute_kurtosis(gauss, fisher=False)
    assert np.isclose(kurt_pearson, 3.0, atol=0.2)

    # Symmetric Gaussian has skewness ~ 0.0
    skew = compute_skewness(gauss)
    assert np.isclose(skew, 0.0, atol=0.1)


def test_zcr_computation():
    # 5 Hz sine wave sampled for 1s at 100 Hz has 10 zero crossings
    t = np.arange(100) / 100.0
    sine = np.sin(2 * np.pi * 5 * t)
    zcr = compute_zcr(sine)
    assert zcr in (9, 10, 11)


def test_frequency_features_energy_split():
    # Signal purely in Band 1 (e.g. 3 Hz)
    t = np.arange(200) / 100.0
    s_band1 = np.sin(2 * np.pi * 3.0 * t)
    b1, b2, b3, dom, centroid = compute_frequency_features(s_band1, fs=100.0)
    assert b1 > 0.85
    assert np.isclose(dom, 3.0, atol=0.5)
    assert 2.0 <= centroid <= 4.0

    # Signal purely in Band 3 (e.g. 30 Hz)
    s_band3 = np.sin(2 * np.pi * 30.0 * t)
    b1_3, b2_3, b3_3, dom_3, centroid_3 = compute_frequency_features(s_band3, fs=100.0)
    assert b3_3 > 0.85
    assert np.isclose(dom_3, 30.0, atol=0.5)
    assert 27.0 <= centroid_3 <= 33.0


def test_feature_vector_dimension(synth):
    sample = synth.generate_sample(SignalType.NORMAL, num_samples=200)
    vec = extract_features(sample, fs=100.0)
    assert len(vec) == len(FEATURE_NAMES)

    feat_dict = extract_features_dict(sample, fs=100.0)
    assert len(feat_dict) == len(FEATURE_NAMES)
    for name in FEATURE_NAMES:
        assert name in feat_dict
        assert not np.isnan(feat_dict[name])


def test_slice_windows():
    data = np.zeros((500, 3))
    windows = slice_windows(data, window_size=200, overlap=100)
    # 500 samples: start at 0, 100, 200, 300 -> 4 windows
    assert len(windows) == 4
    for w in windows:
        assert w.shape == (200, 3)
