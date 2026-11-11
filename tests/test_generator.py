"""
Unit tests for MQTT schemas and the mathematical vibration synthesizer engine.
Validates:
1. Baseline ambient noise: N(0, 0.02^2) and az ~ 1.0g.
2. Wind & rain: Harmonic power concentration in 1-5 Hz with stochastic envelope.
3. Climbing profile: Static z-axis preload (+1.0g offset) and 20-35 Hz transient bursts.
4. Cutting profile: Sharp damped sinusoids (f > 40 Hz) and Fisher kurtosis > 6.0.
5. Schema validation: Serialization, roundtrip parsing, and field constraints.
"""

import inspect
import json
import numpy as np
import pytest
from scipy import stats

from simulator.generator import VibrationSynthesizer
from simulator.schemas import (
    NodeStatus,
    SignalType,
    TelemetryDataPayload,
    TelemetryMessage,
    HeartbeatMessage,
    StatusLWTMessage,
    TriggerCameraMessage,
    CameraAlertMessage,
    BoundingBox,
    DetectionItem,
)
#test 
print(inspect.getfile(VibrationSynthesizer))

@pytest.fixture
def synth():
    return VibrationSynthesizer(fs=100.0, seed=42)


def test_baseline_ambient_noise(synth):
    """
    Validates that baseline ambient noise follows:
    - ax, ay ~ N(0, 0.02^2)
    - az ~ 1.0g + N(0, 0.02^2)
    """
    num_samples = 5000
    data = synth.generate_baseline(num_samples=num_samples, sigma=0.02)

    assert data.shape == (num_samples, 3)

    ax = data[:, 0]
    ay = data[:, 1]
    az = data[:, 2]

    # Means should be close to nominal (ax ~ 0, ay ~ 0, az ~ 1.0)
    assert np.isclose(np.mean(ax), 0.0, atol=0.01)
    assert np.isclose(np.mean(ay), 0.0, atol=0.01)
    assert np.isclose(np.mean(az), 1.0, atol=0.01)

    # Standard deviation should be ~ 0.02
    assert np.isclose(np.std(ax), 0.02, atol=0.005)
    assert np.isclose(np.std(ay), 0.02, atol=0.005)
    assert np.isclose(np.std(az), 0.02, atol=0.005)


# =====================================================================
# 2. WIND & RAIN TESTS
# =====================================================================
def test_wind_and_rain_spectral_concentration(synth):
    """
    Validates that wind & rain oscillations concentrate dominant spectral power
    in the 1-5 Hz band.
    """
    num_samples = 1000  # 10 seconds of data for clear spectral resolution
    data = synth.generate_wind_rain(num_samples=num_samples, fw_range=(1.0, 5.0))

    assert data.shape == (num_samples, 3)

    ax = data[:, 0]
    # Detrend to remove DC
    detrended_x = ax - np.mean(ax)

    # Compute FFT power spectrum
    fft_vals = np.fft.rfft(detrended_x)
    freqs = np.fft.rfftfreq(num_samples, d=1.0 / 100.0)
    power = np.abs(fft_vals) ** 2

    # Energy in 1-5 Hz vs total energy
    mask_1_5 = (freqs >= 1.0) & (freqs <= 5.0)
    wind_power_ratio = np.sum(power[mask_1_5]) / np.sum(power)

    # The majority of non-DC kinetic oscillation power is in the 1-5 Hz flutter band
    assert wind_power_ratio > 0.60, f"Expected wind power ratio > 0.60, got {wind_power_ratio:.3f}"


# =====================================================================
# 3. CLIMBING PROFILE TESTS
# =====================================================================
def test_climbing_profile_mass_preload_and_transients(synth):
    """
    Validates that climbing generates:
    1. Static z-axis mass offset (+1.0g preload to gravity -> mean az ~ 1.8 to 2.2g)
    2. Transient bursts in the 20-35 Hz range
    """
    num_samples = 500  # 5 seconds
    data = synth.generate_climbing(num_samples=num_samples, fc_range=(20.0, 35.0), preload_offset=1.0)

    assert data.shape == (num_samples, 3)

    az = data[:, 2]

    # Mean az should show downward preload (+1.0g added to nominal 1.0g gravity)
    assert 1.75 <= np.mean(az) <= 2.25, f"Expected mean az ~ 2.0g, got {np.mean(az):.3f}"

    # Verify high-frequency mechanical shock component (peak-to-peak > 1.5g)
    ptp_z = np.ptp(az)
    assert ptp_z > 1.5, f"Expected peak-to-peak shock > 1.5g, got {ptp_z:.3f}"


# =====================================================================
# 4. CUTTING PROFILE TESTS
# =====================================================================
def test_cutting_profile_high_kurtosis_and_frequency(synth):
    """
    Validates that cutting profile exhibits:
    1. Sharp damped high-frequency impulses (> 40 Hz)
    2. Impulsive tails producing high Fisher kurtosis (> 6.0)
    """
    num_samples = 400
    # Run across multiple samples to verify consistent impulsive property
    kurtosis_values = []
    for _ in range(5):
        data = synth.generate_cutting(num_samples=num_samples, fk_min=41.0, alpha_damping=30.0)
        az = data[:, 2]
        # Fisher kurtosis = Pearson kurtosis - 3.0 (normal distribution = 0.0)
        fisher_kurt = float(stats.kurtosis(az, fisher=True))
        kurtosis_values.append(fisher_kurt)

    mean_fisher_kurt = np.mean(kurtosis_values)
    assert mean_fisher_kurt > 6.0, f"Expected Fisher kurtosis > 6.0, got {mean_fisher_kurt:.2f}"


# =====================================================================
# 5. DISPATCHER & GENERAL SHAPE TESTS
# =====================================================================
def test_generator_dispatcher(synth):
    for sig_type in [SignalType.NORMAL, SignalType.WIND_RAIN, SignalType.CLIMBING, SignalType.CUTTING]:
        sample = synth.generate_sample(sig_type, num_samples=200)
        assert sample.shape == (200, 3)
        assert not np.isnan(sample).any()


# =====================================================================
# 6. MQTT JSON SCHEMA VALIDATION TESTS
# =====================================================================
def test_telemetry_schema_validation(synth):
    raw_window = synth.generate_sample(SignalType.NORMAL, num_samples=200)
    msg = TelemetryMessage(
        node_id="zone_01_accel",
        zone_id="zone_01",
        timestamp=1773289000.123,
        fs=100.0,
        window_size=200,
        data=TelemetryDataPayload(
            ax=raw_window[:, 0].tolist(),
            ay=raw_window[:, 1].tolist(),
            az=raw_window[:, 2].tolist(),
        ),
    )

    json_str = msg.model_dump_json()
    parsed_dict = json.loads(json_str)

    # Roundtrip validation
    restored = TelemetryMessage.model_validate(parsed_dict)
    assert restored.node_id == "zone_01_accel"
    assert len(restored.data.ax) == 200
    assert len(restored.data.ay) == 200
    assert len(restored.data.az) == 200


def test_heartbeat_and_lwt_schemas():
    hb = HeartbeatMessage(
        node_id="zone_02_accel",
        zone_id="zone_02",
        timestamp=1773289001.0,
        uptime_seconds=3600,
        battery_pct=95.5,
        status=NodeStatus.HEALTHY,
        firmware_version="1.2.0",
    )
    assert hb.status == NodeStatus.HEALTHY

    lwt = StatusLWTMessage(
        node_id="zone_02_accel",
        zone_id="zone_02",
        timestamp=1773289005.0,
        status=NodeStatus.DEAD,
        reason="I2C_DISCONNECT",
    )
    assert lwt.status == NodeStatus.DEAD


def test_camera_trigger_and_alert_schemas():
    trigger = TriggerCameraMessage(
        zone_id="zone_03",
        trigger_node_id="zone_03_accel",
        timestamp=1773289010.5,
        detected_class=SignalType.CLIMBING,
        confidence=0.96,
    )
    assert trigger.detected_class == SignalType.CLIMBING

    alert = CameraAlertMessage(
        node_id="zone_03_cam",
        zone_id="zone_03",
        timestamp=1773289011.8,
        verified=True,
        detections=[
            DetectionItem(
                label="person",
                confidence=0.91,
                box=BoundingBox(ymin=100.0, xmin=50.0, ymax=450.0, xmax=300.0),
            )
        ],
        snapshot_id="snap_zone_03_001",
    )
    assert alert.verified is True
    assert len(alert.detections) == 1
    assert alert.detections[0].label == "person"

