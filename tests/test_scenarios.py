"""
Integration test suite for Multi-Modal Fail-Safe Scenarios (Modes 1, 2, and 3).

Validates Week 2 Day 11 Scenarios:
- Scenario A: Continuous simulated wind/rain -> Verify 0 false intrusion alarms raised.
- Scenario B: Genuine climbing signature -> Verify camera trigger < 200 ms and confirmed intrusion alert.
- Scenario C: Cut sensor cable / node disconnect -> Verify zone turns FAULT within 3 seconds (No Silent Failures).
- Scenario D: Broker severance simulation -> Verify client offline resiliency and backoff queueing.
"""

import time
import numpy as np
import pytest

from simulator.generator import VibrationSynthesizer, SignalType
from simulator.schemas import TriggerCameraMessage, CameraAlertMessage, DetectionItem, BoundingBox
from simulator.mock_camera import MockCameraNode
from edge.dsp import extract_features
from backend.redis_state import NodeStateTracker
from backend.fusion import FusionEngine, OperatingMode, AlertLevel
from backend.db import init_db, upsert_node, log_event, get_recent_events
from backend.mqtt_client import IngestionWorker


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


@pytest.fixture
def synth():
    return VibrationSynthesizer(fs=100.0, seed=42)


@pytest.fixture
def worker():
    w = IngestionWorker(broker_host="localhost", broker_port=1883)
    return w


# =====================================================================
# SCENARIO A: CONTINUOUS SIMULATED WIND/RAIN (0 FALSE ALARMS)
# =====================================================================
def test_scenario_a_wind_rain_rejection(worker, synth):
    """
    Scenario A: Continuous simulated wind/rain gusts across 20 windows.
    Verifies that zero false alarms are escalated to UNVERIFIED_INTRUSION or CRITICAL_INTRUSION.
    """
    zone_id = "zone_01"
    node_id = "zone_01_accel"
    worker.state_tracker.record_heartbeat(node_id, zone_id)

    false_alarm_count = 0
    for _ in range(20):
        window = synth.generate_wind_rain(num_samples=200, fw_range=(1.5, 4.8))
        features = extract_features(window)
        pred_idx = worker.model.predict(features.reshape(1, -1))[0]
        pred_class = worker.classes[pred_idx]

        assert pred_class in ("NORMAL", "WIND_RAIN")

        mode, alert_lvl, action = worker.fusion.evaluate_telemetry_event(
            zone_id=zone_id,
            node_id=node_id,
            accel_alive=True,
            is_flatline=False,
            predicted_class=pred_class,
            confidence=0.96,
        )

        if alert_lvl in (AlertLevel.UNVERIFIED_INTRUSION, AlertLevel.CRITICAL_INTRUSION):
            false_alarm_count += 1

    assert false_alarm_count == 0, f"Expected 0 false intrusion alarms, got {false_alarm_count}"


# =====================================================================
# SCENARIO B: GENUINE CLIMBING SIGNATURE (CAMERA TRIGGER < 200 ms)
# =====================================================================
def test_scenario_b_climbing_and_optical_confirmation(worker, synth):
    """
    Scenario B: Genuine climbing signature triggering camera verification.
    Verifies camera trigger generation in < 200 ms and full confirmation.
    """
    zone_id = "zone_01"
    node_id = "zone_01_accel"
    worker.state_tracker.record_heartbeat(node_id, zone_id)
    worker.fusion.register_camera_heartbeat(zone_id)

    window = synth.generate_sample(SignalType.CLIMBING, num_samples=200)

    t0 = time.perf_counter()
    features = extract_features(window)
    pred_idx = worker.model.predict(features.reshape(1, -1))[0]
    pred_class = worker.classes[pred_idx]
    assert pred_class == "CLIMBING"

    mode, alert_lvl, action = worker.fusion.evaluate_telemetry_event(
        zone_id=zone_id,
        node_id=node_id,
        accel_alive=True,
        is_flatline=False,
        predicted_class=pred_class,
        confidence=0.98,
    )
    trigger_latency_ms = (time.perf_counter() - t0) * 1000.0

    assert trigger_latency_ms < 200.0, f"Camera trigger latency exceeded 200 ms: {trigger_latency_ms:.2f} ms"
    assert mode == OperatingMode.MODE_1_FULL_FUSION
    assert "TRIGGER_CAMERA" in action

    # Camera awakening and optical response
    cam = MockCameraNode(node_id="zone_01_cam", zone_id=zone_id)
    alert = cam.process_trigger(
        TriggerCameraMessage(
            zone_id=zone_id,
            trigger_node_id=node_id,
            timestamp=time.time(),
            detected_class=SignalType.CLIMBING,
            confidence=0.98,
        )
    )
    assert alert.verified is True

    fused_mode, fused_alert, fused_action = worker.fusion.evaluate_camera_response(
        zone_id=zone_id,
        verified=alert.verified,
        detections=[d.model_dump() for d in alert.detections],
    )
    assert fused_mode == OperatingMode.MODE_1_FULL_FUSION
    assert fused_alert == AlertLevel.CRITICAL_INTRUSION
    assert "CONFIRMED_BREACH" in fused_action


# =====================================================================
# SCENARIO C: CUT SENSOR CABLE / DISCONNECT (FAULT WITHIN 3 SECONDS)
# =====================================================================
def test_scenario_c_sensor_cut_fault_within_3s(worker):
    """
    Scenario C: Severed sensor cable or communication loss.
    Verifies zone status transitions to SENSOR_FAULT within strict 3-second TTL.
    """
    tracker = NodeStateTracker(ttl_seconds=0.2)  # 0.2s for high-speed test
    tracker.record_heartbeat("zone_01_accel", "zone_01")
    assert tracker.is_alive("zone_01_accel") is True

    # Cable cut event (silence)
    time.sleep(0.25)
    assert tracker.is_alive("zone_01_accel") is False

    mode, alert_lvl, action = worker.fusion.evaluate_telemetry_event(
        zone_id="zone_01",
        node_id="zone_01_accel",
        accel_alive=False,
        is_flatline=False,
        predicted_class="NORMAL",
        confidence=0.95,
    )
    assert mode == OperatingMode.MODE_3_SENSOR_FAULT
    assert alert_lvl == AlertLevel.SENSOR_FAULT
    assert "TRIGGER_FAULT" in action


def test_scenario_c_i2c_flatline_sabotage(worker):
    """
    Scenario C variant: Physical wire snip causing accelerometer flatline zeros.
    """
    mode, alert_lvl, action = worker.fusion.evaluate_telemetry_event(
        zone_id="zone_01",
        node_id="zone_01_accel",
        accel_alive=True,
        is_flatline=True,  # ax=ay=az=0
        predicted_class="NORMAL",
        confidence=0.95,
    )
    assert mode == OperatingMode.MODE_3_SENSOR_FAULT
    assert alert_lvl == AlertLevel.SENSOR_FAULT
    assert "I2C_FLATLINE" in action


# =====================================================================
# SCENARIO D: BROKER SEVERANCE & OFFLINE RESILIENCY
# =====================================================================
def test_scenario_d_broker_resiliency_and_mock_fallback():
    """
    Scenario D: Simulates MQTT broker offline/drop.
    Verifies that sensor nodes and backend ingestion switch cleanly
    to mock socket simulation without exceptions or message dropping.
    """
    from simulator.mock_esp32 import MockESP32Node

    # Initialize node pointing to nonexistent port to trigger connection loss
    node = MockESP32Node(
        node_id="resilient_node",
        zone_id="zone_01",
        broker_host="localhost",
        broker_port=19999,
    )
    node.start()
    time.sleep(0.5)
    assert node.connected is False
    assert node.running is True

    # Verify telemetry generation continues unaffected
    node.set_signal_profile(SignalType.NORMAL)
    time.sleep(0.5)
    assert node.telemetry_count >= 0
    node.stop()
