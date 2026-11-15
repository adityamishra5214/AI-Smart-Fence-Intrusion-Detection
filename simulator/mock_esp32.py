"""
Multithreaded virtual ESP32 node publisher for AI-Based PIDS.

Simulates 3 concurrent fence zones (zone_01, zone_02, zone_03).
Features:
- Windowed acceleration arrays (100 Hz MPU6050 synthetic data, 2.0s window, 50% overlap) -> fence/node/{id}/telemetry
- 1 Hz node health heartbeat -> fence/node/{id}/heartbeat
- MQTT Last Will and Testament (LWT) retained message -> fence/node/{id}/status
- Dynamic fault injection hooks:
  * 'flatline': I2C sensor flatline (ax=ay=az=0.0)
  * 'network_drop': silent network disconnect (triggers 3s TTL timeout)
  * 'jitter': excessive out-of-range sensor jitter
  * 'wind': continuous wind/rain environmental storm
  * 'climb': structural climbing intrusion
  * 'cut': wire-cutting structural shock
"""

import sys
from pathlib import Path
import os
import time
import json
import threading
from typing import Dict, Optional, List
import numpy as np

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    import paho.mqtt.client as mqtt
    PAHO_AVAILABLE = True
except ImportError:
    PAHO_AVAILABLE = False

from simulator.generator import VibrationSynthesizer, SignalType
from simulator.schemas import (
    TelemetryMessage,
    TelemetryDataPayload,
    HeartbeatMessage,
    StatusLWTMessage,
    NodeStatus,
)


class MockESP32Node:
    """
    Virtual ESP32 firmware running MPU6050 acquisition, windowing,
    heartbeat timer, and MQTT publisher.
    """

    def __init__(
        self,
        node_id: str,
        zone_id: str,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        fs: float = 100.0,
        window_size: int = 200,
        overlap: int = 100,
        firmware_version: str = "1.0.4",
    ):
        self.node_id = node_id
        self.zone_id = zone_id
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.fs = fs
        self.window_size = window_size
        self.overlap = overlap
        self.step_samples = window_size - overlap
        self.firmware_version = firmware_version

        self.synth = VibrationSynthesizer(fs=fs)
        self.client: Optional[mqtt.Client] = None
        self.connected = False
        self.running = False

        # Node state & telemetry buffer
        self.current_signal = SignalType.NORMAL
        self.fault_mode: Optional[str] = None  # None, 'flatline', 'network_drop', 'jitter'
        self.battery_pct = 99.4
        self.start_time = time.time()
        self.telemetry_count = 0
        self.heartbeat_count = 0

        # Ring buffer for 100 Hz simulation
        self._buffer: List[List[float]] = []
        self._lock = threading.Lock()

        # Threads
        self._telemetry_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None

    def _setup_mqtt(self):
        """Configures MQTT client with Last Will and Testament (LWT)."""
        if not PAHO_AVAILABLE:
            print(f"[{self.node_id}] Paho-MQTT not installed. Running in mock socket mode.")
            return

        lwt_payload = StatusLWTMessage(
            node_id=self.node_id,
            zone_id=self.zone_id,
            timestamp=time.time(),
            status=NodeStatus.DEAD,
            reason="UNEXPECTED_CONNECTION_LOSS",
        ).model_dump_json()

        lwt_topic = f"fence/node/{self.node_id}/status"

        try:
            # Paho MQTT 2.0+ supports CallbackAPIVersion
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"esp32_{self.node_id}_{int(time.time())}")
        except (AttributeError, TypeError):
            self.client = mqtt.Client(client_id=f"esp32_{self.node_id}_{int(time.time())}")

        self.client.will_set(lwt_topic, payload=lwt_payload, qos=1, retain=True)

        try:
            self.client.connect(self.broker_host, self.broker_port, keepalive=10)
            self.client.loop_start()
            self.connected = True
            # Publish initial healthy status
            online_payload = json.dumps({"status": "ONLINE", "node_id": self.node_id, "zone_id": self.zone_id})
            self.client.publish(f"fence/node/{self.node_id}/status", online_payload, qos=1, retain=True)
            print(f"[{self.node_id}] Connected to MQTT broker at {self.broker_host}:{self.broker_port}")
        except Exception as e:
            print(f"[{self.node_id}] Warning: Could not connect to MQTT broker ({e}). Simulating locally.")
            self.connected = False

    def publish_message(self, topic: str, payload_json: str):
        """Dispatches message to broker or logs locally."""
        if self.fault_mode == "network_drop":
            # Silent dropout: do not transmit packets
            return

        if self.connected and self.client:
            self.client.publish(topic, payload_json, qos=0)
        else:
            # Standalone / mock simulation sink
            pass

    def set_signal_profile(self, signal: SignalType):
        """Alters environmental or breach signal synthesizer profile."""
        with self._lock:
            self.current_signal = signal
            self.fault_mode = None
            print(f"[{self.node_id}] Switched profile to {signal.value}")

    def inject_fault(self, fault_type: str):
        """
        Injects real-world hardware or network failure:
        - 'flatline': I2C wire cut or accelerometer freeze (ax=ay=az=0.0)
        - 'network_drop': silent dropped packets (triggers backend TTL fault)
        - 'jitter': out-of-range catastrophic sensor noise
        """
        with self._lock:
            self.fault_mode = fault_type
            print(f"[{self.node_id}] FAULT INJECTED: {fault_type.upper()}")

    def clear_fault(self):
        """Restores normal operation."""
        with self._lock:
            self.fault_mode = None
            self.current_signal = SignalType.NORMAL
            print(f"[{self.node_id}] Fault cleared. Restored to NORMAL.")

    def _telemetry_loop(self):
        """Generates continuous 100 Hz acceleration batches published every step interval."""
        interval = self.step_samples / self.fs  # 1.0 second for 50% overlap of 200 samples

        while self.running:
            start_ts = time.time()

            with self._lock:
                current_sig = self.current_signal
                fault = self.fault_mode

            # Generate synthetic window
            if fault == "flatline":
                # Flatline zero condition
                window = np.zeros((self.window_size, 3))
            elif fault == "jitter":
                # Severe electrical jitter / power rail corruption
                window = np.random.uniform(-16.0, 16.0, size=(self.window_size, 3))
            else:
                window = self.synth.generate_sample(current_sig, num_samples=self.window_size)

            msg = TelemetryMessage(
                node_id=self.node_id,
                zone_id=self.zone_id,
                timestamp=time.time(),
                fs=self.fs,
                window_size=self.window_size,
                data=TelemetryDataPayload(
                    ax=[round(float(v), 5) for v in window[:, 0]],
                    ay=[round(float(v), 5) for v in window[:, 1]],
                    az=[round(float(v), 5) for v in window[:, 2]],
                ),
            )

            topic = f"fence/node/{self.node_id}/telemetry"
            self.publish_message(topic, msg.model_dump_json())
            self.telemetry_count += 1

            # Precision pacing
            elapsed = time.time() - start_ts
            sleep_time = max(0.01, interval - elapsed)
            time.sleep(sleep_time)

    def _heartbeat_loop(self):
        """Transmits 1 Hz node vitality, battery, and uptime."""
        while self.running:
            uptime = int(time.time() - self.start_time)
            self.battery_pct = max(0.0, self.battery_pct - 0.0001)

            status = NodeStatus.FAULT if self.fault_mode else NodeStatus.HEALTHY

            hb = HeartbeatMessage(
                node_id=self.node_id,
                zone_id=self.zone_id,
                timestamp=time.time(),
                uptime_seconds=uptime,
                battery_pct=round(self.battery_pct, 2),
                status=status,
                firmware_version=self.firmware_version,
            )

            topic = f"fence/node/{self.node_id}/heartbeat"
            self.publish_message(topic, hb.model_dump_json())
            self.heartbeat_count += 1

            time.sleep(1.0)

    def start(self):
        """Starts worker threads."""
        self.running = True
        self._setup_mqtt()

        self._telemetry_thread = threading.Thread(target=self._telemetry_loop, daemon=True)
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)

        self._telemetry_thread.start()
        self._heartbeat_thread.start()
        print(f"[{self.node_id}] Started 100 Hz telemetry & 1 Hz heartbeat streaming.")

    def stop(self):
        """Stops worker threads and disconnects MQTT client."""
        self.running = False
        if self.connected and self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass
        print(f"[{self.node_id}] Node stopped. (Telemetry sent: {self.telemetry_count}, Heartbeats: {self.heartbeat_count})")


class FenceSimulationCluster:
    """Manages concurrent emulation of 3 fence zones."""

    def __init__(self, broker_host: str = "localhost", broker_port: int = 1883):
        self.nodes: Dict[str, MockESP32Node] = {
            "zone_01_accel": MockESP32Node("zone_01_accel", "zone_01", broker_host, broker_port),
            "zone_02_accel": MockESP32Node("zone_02_accel", "zone_02", broker_host, broker_port),
            "zone_03_accel": MockESP32Node("zone_03_accel", "zone_03", broker_host, broker_port),
        }

    def start_all(self):
        print("=" * 60)
        print("Starting 3-Zone Concurrent Fence Node Simulator")
        print("=" * 60)
        for node in self.nodes.values():
            node.start()

    def stop_all(self):
        print("\nStopping simulated fence nodes...")
        for node in self.nodes.values():
            node.stop()

    def get_node(self, node_id: str) -> Optional[MockESP32Node]:
        return self.nodes.get(node_id)


if __name__ == "__main__":
    cluster = FenceSimulationCluster()
    try:
        cluster.start_all()
        print("\nSimulator running. Press Ctrl+C to stop.")
        # Emulate 5 seconds normal, then inject wind in zone_02, climb in zone_03
        time.sleep(2)
        print("\n--> [DEMO] Injecting WIND storm into zone_02...")
        cluster.get_node("zone_02_accel").set_signal_profile(SignalType.WIND_RAIN)

        time.sleep(2)
        print("\n--> [DEMO] Injecting CLIMBING breach into zone_03...")
        cluster.get_node("zone_03_accel").set_signal_profile(SignalType.CLIMBING)

        time.sleep(2)
        print("\n--> [DEMO] Injecting I2C FLATLINE fault into zone_01...")
        cluster.get_node("zone_01_accel").inject_fault("flatline")

        time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        cluster.stop_all()
