"""
Mathematical validation tests for feature extractors in edge/dsp.py.
Validates RMS, Kurtosis, ZCR, FFT energy band ratios, and Spectral Centroid.
"""

import numpy as np
import pytest
from edge.dsp import (
    compute_rms,
    compute_kurtosis,
    compute_zcr,
    compute_spectral_features,
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
    # DC signal of 3.0 should have RMS = 3.0
    dc = np.full(100, 3.0)
    assert np.isclose(compute_rms(dc), 3.0)

    # Sine wave with amplitude A has RMS = A / sqrt(2)
    t = np.linspace(0, 1, 1000, endpoint=False)
    sine = 2.0 * np.sin(2 * np.pi * 10 * t)
    expected_rms = 2.0 / np.sqrt(2)
    assert np.isclose(compute_rms(sine), expected_rms, atol=0.05)


def test_kurtosis_computation():
    # Gaussian noise should have Pearson kurtosis close to 3.0
    rng = np.random.default_rng(42)
    gauss = rng.normal(0, 1.0, size=10000)
    kurt = compute_kurtosis(gauss)
    assert np.isclose(kurt, 3.0, atol=0.2)


def test_zcr_computation():
    # 5 Hz sine wave sampled for 1s at 100 Hz has 10 zero crossings
    t = np.arange(100) / 100.0
    sine = np.sin(2 * np.pi * 5 * t)
    zcr = compute_zcr(sine)
    assert zcr in (9, 10, 11)


def test_spectral_features_energy_split():
    # Signal purely in Band 1 (e.g. 3 Hz)
    t = np.arange(200) / 100.0
    s_band1 = np.sin(2 * np.pi * 3.0 * t)
    b1, b2, b3, centroid = compute_spectral_features(s_band1, fs=100.0)
    assert b1 > 0.85
    assert 2.0 <= centroid <= 4.0

    # Signal purely in Band 3 (e.g. 30 Hz)
    s_band3 = np.sin(2 * np.pi * 30.0 * t)
    b1_3, b2_3, b3_3, centroid_3 = compute_spectral_features(s_band3, fs=100.0)
    assert b3_3 > 0.85
    assert 27.0 <= centroid_3 <= 33.0


def test_feature_vector_dimension(synth):
    sample = synth.generate_sample(SignalType.NORMAL, num_samples=200)
    vec = extract_features(sample, fs=100.0)
    assert len(vec) == 14
    assert len(FEATURE_NAMES) == 14

    feat_dict = extract_features_dict(sample, fs=100.0)
    assert len(feat_dict) == 14
    for name in FEATURE_NAMES:
        assert name in feat_dict


def test_slice_windows():
    data = np.zeros((500, 3))
    windows = slice_windows(data, window_size=200, overlap=100)
    # 500 samples: start at 0, 100, 200, 300 -> 4 windows
    assert len(windows) == 4
    for w in windows:
        assert w.shape == (200, 3)
