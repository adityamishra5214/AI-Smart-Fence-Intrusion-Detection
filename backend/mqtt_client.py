"""
Asynchronous MQTT ingestion worker subscribing to telemetry, heartbeat, and camera alerts.
Extracts features, runs ML inference, drives the multi-modal fusion state machine,
and writes to PostgreSQL/SQLite persistence.
"""

import sys
from pathlib import Path
import os
import json
import time
import joblib
import numpy as np
from typing import Optional, Dict, Any, List

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    import paho.mqtt.client as mqtt
    PAHO_AVAILABLE = True
except ImportError:
    PAHO_AVAILABLE = False

from edge.dsp import extract_features
from backend.redis_state import NodeStateTracker
from backend.fusion import FusionEngine, OperatingMode, AlertLevel
from backend.db import upsert_node, log_telemetry_batch, log_event, log_fault
from simulator.schemas import TelemetryMessage, HeartbeatMessage, StatusLWTMessage, CameraAlertMessage

MODEL_PATH = _project_root / "ml" / "model.joblib"


class IngestionWorker:
    """Telemetry ingestion engine and real-time fusion processor."""

    def __init__(
        self,
        broker_host: str = "localhost",
        broker_port: int = 1883,
        redis_host: str = "localhost",
        redis_port: int = 6379,
    ):
        self.broker_host = broker_host
        self.broker_port = broker_port

        self.state_tracker = NodeStateTracker(host=redis_host, port=redis_port, ttl_seconds=3.0)
        self.fusion = FusionEngine(camera_timeout_seconds=3.0)

        # Load ML model
        self.model = None
        self.classes = ["NORMAL", "WIND_RAIN", "CLIMBING", "CUTTING"]
        self._load_model()

        # Telemetry batching queue
        self._telemetry_queue: List[Dict[str, Any]] = []
        self._last_flush_time = time.time()
        self._flush_interval = 0.5  # Flush DB every 500 ms
        self._batch_size = 20

        # MQTT Client
        self.client: Optional[mqtt.Client] = None
        self.running = False
        self.processed_count = 0
        self.event_count = 0

    def _load_model(self):
        if MODEL_PATH.exists():
            try:
                pkg = joblib.load(MODEL_PATH)
                self.model = pkg["model"]
                self.model.n_jobs = 1
                self.classes = pkg.get("classes", self.classes)
                print(f"[IngestionWorker] Loaded edge model from {MODEL_PATH}")
            except Exception as e:
                print(f"[IngestionWorker] Warning: Failed to load model ({e}). Using heuristic fallback.")
        else:
            print("[IngestionWorker] Notice: ml/model.joblib not found. Run ml/train_baseline.py.")

    def on_connect(self, client, userdata, flags, rc, properties=None):
        print(f"[IngestionWorker] Connected to MQTT broker (rc={rc})")
        # Subscribe to all sensor node topics
        client.subscribe("fence/node/+/telemetry")
        client.subscribe("fence/node/+/heartbeat")
        client.subscribe("fence/node/+/status")
        client.subscribe("fence/node/+/camera_alert")

    def on_message(self, client, userdata, msg):
        topic = msg.topic
        payload_str = msg.payload.decode("utf-8", errors="ignore")

        try:
            if topic.endswith("/telemetry"):
                self.handle_telemetry(payload_str)
            elif topic.endswith("/heartbeat"):
                self.handle_heartbeat(payload_str)
            elif topic.endswith("/status"):
                self.handle_status_lwt(payload_str)
            elif topic.endswith("/camera_alert"):
                self.handle_camera_alert(payload_str)
        except Exception as e:
            print(f"[IngestionWorker] Error processing {topic}: {e}")

    def handle_heartbeat(self, payload_str: str):
        data = json.loads(payload_str)
        hb = HeartbeatMessage.model_validate(data)
        self.state_tracker.record_heartbeat(hb.node_id, hb.zone_id)
        upsert_node(hb.node_id, hb.zone_id, status=hb.status.value, firmware_version=hb.firmware_version)

    def handle_status_lwt(self, payload_str: str):
        data = json.loads(payload_str)
        lwt = StatusLWTMessage.model_validate(data)
        if lwt.status.value == "DEAD":
            print(f"[IngestionWorker] LWT ALERT: Node {lwt.node_id} reported DEAD ({lwt.reason})")
            self.state_tracker.mark_dead(lwt.node_id)
            log_fault(lwt.node_id, "LWT_DISCONNECT", lwt.reason)
            log_event(
                zone_id=lwt.zone_id,
                node_id=lwt.node_id,
                timestamp=lwt.timestamp,
                operating_mode=OperatingMode.MODE_3_SENSOR_FAULT.value,
                vibration_class="UNKNOWN",
                camera_verified=False,
                confidence=1.0,
                alert_level=AlertLevel.SENSOR_FAULT.value,
                metadata_json={"reason": lwt.reason},
            )

    def handle_camera_alert(self, payload_str: str):
        data = json.loads(payload_str)
        alert = CameraAlertMessage.model_validate(data)
        self.fusion.register_camera_heartbeat(alert.zone_id)

        mode, alert_level, action = self.fusion.evaluate_camera_response(
            zone_id=alert.zone_id,
            verified=alert.verified,
            detections=[d.model_dump() for d in alert.detections],
        )

        if alert_level == AlertLevel.CRITICAL_INTRUSION:
            print(f"[IngestionWorker] [CRITICAL_INTRUSION] Confirmed person intrusion in {alert.zone_id}!")
            log_event(
                zone_id=alert.zone_id,
                node_id=alert.node_id,
                timestamp=alert.timestamp,
                operating_mode=mode.value,
                vibration_class="CAMERA_VERIFIED",
                camera_verified=True,
                confidence=0.98,
                alert_level=alert_level.value,
                metadata_json={"detections": [d.model_dump() for d in alert.detections]},
            )

    def handle_telemetry(self, payload_str: str):
        data = json.loads(payload_str)
        tel = TelemetryMessage.model_validate(data)
        t_arr = np.column_stack((tel.data.ax, tel.data.ay, tel.data.az))

        # Check I2C flatline failure: all zero data
        is_flatline = bool(np.all(t_arr == 0.0))
        is_alive = self.state_tracker.is_alive(tel.node_id)

        # Feature extraction
        feat_vec = extract_features(t_arr, fs=tel.fs)

        # ML Inference
        pred_class = "NORMAL"
        confidence = 0.95
        if is_flatline:
            pred_class = "SENSOR_FAULT"
        elif self.model is not None:
            pred_idx = self.model.predict(feat_vec.reshape(1, -1))[0]
            pred_class = self.classes[pred_idx]
            proba = self.model.predict_proba(feat_vec.reshape(1, -1))[0]
            confidence = float(np.max(proba))

        # Multi-Modal Fusion Evaluation
        mode, alert_lvl, action = self.fusion.evaluate_telemetry_event(
            zone_id=tel.zone_id,
            node_id=tel.node_id,
            accel_alive=is_alive,
            is_flatline=is_flatline,
            predicted_class=pred_class,
            confidence=confidence,
        )

        self.processed_count += 1

        # Queue telemetry metrics for high-throughput batch save
        record = {
            "node_id": tel.node_id,
            "timestamp": tel.timestamp,
            "rms_x": float(feat_vec[9]),  # index from FEATURE_NAMES
            "rms_y": float(feat_vec[10]),
            "rms_z": float(feat_vec[11]),
            "kurtosis_z": float(feat_vec[15]),
            "zcr_z": int(feat_vec[19]),
            "band1_ratio": float(feat_vec[20]),
            "band2_ratio": float(feat_vec[21]),
            "band3_ratio": float(feat_vec[22]),
            "dominant_freq": float(feat_vec[23]),
            "spectral_centroid": float(feat_vec[24]),
            "predicted_class": pred_class,
        }
        self._telemetry_queue.append(record)

        if len(self._telemetry_queue) >= self._batch_size or (time.time() - self._last_flush_time) > self._flush_interval:
            self._flush_telemetry_batch()

        # Log security event if intrusion or sensor fault
        if alert_lvl in (AlertLevel.UNVERIFIED_INTRUSION, AlertLevel.CRITICAL_INTRUSION, AlertLevel.SENSOR_FAULT):
            self.event_count += 1
            log_event(
                zone_id=tel.zone_id,
                node_id=tel.node_id,
                timestamp=tel.timestamp,
                operating_mode=mode.value,
                vibration_class=pred_class,
                camera_verified=False,
                confidence=confidence,
                alert_level=alert_lvl.value,
                metadata_json={"action": action, "is_flatline": is_flatline},
            )

    def _flush_telemetry_batch(self):
        if not self._telemetry_queue:
            return
        batch = list(self._telemetry_queue)
        self._telemetry_queue.clear()
        self._last_flush_time = time.time()
        log_telemetry_batch(batch)

    def start(self):
        if not PAHO_AVAILABLE:
            print("[IngestionWorker] Paho-MQTT not installed. Running in mock offline mode.")
            return

        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"backend_ingestion_{int(time.time())}")
        except (AttributeError, TypeError):
            self.client = mqtt.Client(client_id=f"backend_ingestion_{int(time.time())}")

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        try:
            self.client.connect(self.broker_host, self.broker_port, keepalive=15)
            self.client.loop_start()
            self.running = True
            print(f"[IngestionWorker] Subscribed to broker at {self.broker_host}:{self.broker_port}")
        except Exception as e:
            print(f"[IngestionWorker] Warning: Broker connection failed ({e}). Standalone mode active.")

    def stop(self):
        self.running = False
        self._flush_telemetry_batch()
        if self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass
        print(f"[IngestionWorker] Stopped. (Total telemetry windows processed: {self.processed_count})")
