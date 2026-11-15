"""
Multi-Modal Sensor Fusion & Fail-Safe State Engine.

Implements the 3 operational modes defined in the system architecture:
- Mode 1 (Full Fusion): Accel Node OK + Camera Node OK -> Verified Person Detection -> CRITICAL_INTRUSION
- Mode 2 (Degraded: Vibration Only): Accel Node OK + Camera Node OFFLINE / Timeout -> UNVERIFIED_INTRUSION (Escalate to patrol)
- Mode 3 (Degraded: Sensor Fault): Heartbeat dropped (> 3s TTL) or flatline I2C -> SENSOR_FAULT (Never 'All-Clear')

Broadcasts fused state changes via Redis Pub/Sub (channel: fence:events:broadcast)
with in-memory fallback for local simulation.
"""

import time
import json
import os
from enum import Enum
from typing import Dict, Optional, Tuple, Any, List, Callable

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class OperatingMode(str, Enum):
    MODE_1_FULL_FUSION = "MODE_1"
    MODE_2_VIBRATION_ONLY = "MODE_2"
    MODE_3_SENSOR_FAULT = "MODE_3"


class AlertLevel(str, Enum):
    NORMAL = "NORMAL"
    WIND_WARNING = "WIND_WARNING"
    UNVERIFIED_INTRUSION = "UNVERIFIED_INTRUSION"
    CRITICAL_INTRUSION = "CRITICAL_INTRUSION"
    SENSOR_FAULT = "SENSOR_FAULT"


class FusionEngine:
    """
    Evaluates multi-modal sensor state, camera verification responses,
    and fail-safe degradation triggers.
    """

    def __init__(
        self,
        camera_timeout_seconds: float = 3.0,
        redis_host: str = "localhost",
        redis_port: int = 6379,
    ):
        self.camera_timeout_seconds = camera_timeout_seconds
        # Active pending alerts awaiting camera confirmation: zone_id -> {timestamp, vibration_class, node_id}
        self.pending_camera_verifications: Dict[str, Dict[str, Any]] = {}
        # Known camera node vitality: zone_id -> last_seen_timestamp
        self.camera_heartbeats: Dict[str, float] = {}

        # Pub/Sub broadcaster
        self.redis_client: Optional[redis.Redis] = None
        self.use_redis_pubsub = False
        self._local_subscribers: List[Callable[[Dict[str, Any]], None]] = []

        enable_redis = os.getenv("ENABLE_REDIS", "false").lower() in ("true", "1", "yes")
        if enable_redis and REDIS_AVAILABLE:
            try:
                r = redis.Redis(host=redis_host, port=redis_port, socket_timeout=0.2)
                r.ping()
                self.redis_client = r
                self.use_redis_pubsub = True
            except Exception:
                self.use_redis_pubsub = False

    def subscribe_locally(self, callback: Callable[[Dict[str, Any]], None]):
        """Registers in-process event listener for UI WebSocket or streaming pipes."""
        self._local_subscribers.append(callback)

    def broadcast_state_change(
        self,
        zone_id: str,
        operating_mode: OperatingMode,
        alert_level: AlertLevel,
        details: Dict[str, Any],
    ):
        """Broadcasts state change via Redis Pub/Sub channel and local listeners."""
        payload = {
            "timestamp": time.time(),
            "zone_id": zone_id,
            "operating_mode": operating_mode.value,
            "alert_level": alert_level.value,
            "details": details,
        }
        json_str = json.dumps(payload)

        # Redis Pub/Sub
        if self.use_redis_pubsub and self.redis_client:
            try:
                self.redis_client.publish("fence:events:broadcast", json_str)
            except Exception as e:
                print(f"[FusionEngine] Redis publish error: {e}")

        # In-memory listeners
        for sub in self._local_subscribers:
            try:
                sub(payload)
            except Exception:
                pass

    def register_camera_heartbeat(self, zone_id: str):
        """Records camera vitality to verify Mode 1 capability."""
        self.camera_heartbeats[zone_id] = time.time()

    def is_camera_available(self, zone_id: str) -> bool:
        """Returns True if the optical camera node in the zone is active."""
        last_seen = self.camera_heartbeats.get(zone_id)
        if last_seen is None:
            return False
        return (time.time() - last_seen) <= 10.0  # 10s camera keepalive

    def evaluate_telemetry_event(
        self,
        zone_id: str,
        node_id: str,
        accel_alive: bool,
        is_flatline: bool,
        predicted_class: str,
        confidence: float,
    ) -> Tuple[OperatingMode, AlertLevel, Optional[str]]:
        """
        Processes new window telemetry classification from vibration node.
        Returns: (operating_mode, alert_level, dispatch_action)
        """
        now = time.time()

        # Check Mode 3 (Sensor Fault): Heartbeat dropped or flatline I2C data
        if not accel_alive or is_flatline:
            reason = "I2C_FLATLINE" if is_flatline else "HEARTBEAT_TTL_EXPIRED"
            mode = OperatingMode.MODE_3_SENSOR_FAULT
            alert = AlertLevel.SENSOR_FAULT
            action = f"TRIGGER_FAULT:{reason}"
            self.broadcast_state_change(zone_id, mode, alert, {"node_id": node_id, "reason": reason})
            return mode, alert, action

        # Environmental noise suppression
        if predicted_class == "NORMAL":
            return OperatingMode.MODE_1_FULL_FUSION, AlertLevel.NORMAL, None

        if predicted_class == "WIND_RAIN":
            # Environmental flutter safely classified and isolated
            mode = OperatingMode.MODE_1_FULL_FUSION
            alert = AlertLevel.WIND_WARNING
            self.broadcast_state_change(zone_id, mode, alert, {"node_id": node_id, "info": "Wind buffeting suppressed"})
            return mode, alert, None

        # Structural breach: CLIMBING or CUTTING
        camera_online = self.is_camera_available(zone_id)

        if camera_online:
            # Mode 1: Trigger camera optical verification
            self.pending_camera_verifications[zone_id] = {
                "timestamp": now,
                "vibration_class": predicted_class,
                "node_id": node_id,
                "confidence": confidence,
            }
            mode = OperatingMode.MODE_1_FULL_FUSION
            alert = AlertLevel.UNVERIFIED_INTRUSION
            action = f"TRIGGER_CAMERA:{zone_id}:{predicted_class}"
            self.broadcast_state_change(zone_id, mode, alert, {"node_id": node_id, "vibration_class": predicted_class, "awaiting_camera": True})
            return mode, alert, action
        else:
            # Mode 2: Camera node offline, immediate degraded vibration escalation
            mode = OperatingMode.MODE_2_VIBRATION_ONLY
            alert = AlertLevel.UNVERIFIED_INTRUSION
            action = f"ESCALATE_PATROL:{zone_id}:{predicted_class}"
            self.broadcast_state_change(zone_id, mode, alert, {"node_id": node_id, "vibration_class": predicted_class, "camera_offline": True})
            return mode, alert, action

    def evaluate_camera_response(
        self,
        zone_id: str,
        verified: bool,
        detections: list,
    ) -> Tuple[OperatingMode, AlertLevel, str]:
        """Processes camera verification report."""
        pending = self.pending_camera_verifications.pop(zone_id, None)

        if verified and any(d.get("label") == "person" for d in detections):
            # Mode 1: Confirmed person breach
            mode = OperatingMode.MODE_1_FULL_FUSION
            alert = AlertLevel.CRITICAL_INTRUSION
            action = f"CONFIRMED_BREACH:{zone_id}"
            self.broadcast_state_change(zone_id, mode, alert, {"detections": detections, "confirmed": True})
            return mode, alert, action
        else:
            # Camera checked and found false trigger or wildlife
            mode = OperatingMode.MODE_1_FULL_FUSION
            alert = AlertLevel.NORMAL
            action = f"CAMERA_REJECTED:{zone_id}"
            self.broadcast_state_change(zone_id, mode, alert, {"rejected": True, "detections": detections})
            return mode, alert, action

    def check_pending_camera_timeouts(self) -> list:
        """
        Audits pending camera triggers.
        If camera does not reply within camera_timeout_seconds, degrade to Mode 2.
        """
        now = time.time()
        degraded_events = []
        expired_zones = []

        for zone_id, info in self.pending_camera_verifications.items():
            if (now - info["timestamp"]) > self.camera_timeout_seconds:
                expired_zones.append(zone_id)
                event_data = {
                    "zone_id": zone_id,
                    "node_id": info["node_id"],
                    "operating_mode": OperatingMode.MODE_2_VIBRATION_ONLY.value,
                    "alert_level": AlertLevel.UNVERIFIED_INTRUSION.value,
                    "vibration_class": info["vibration_class"],
                    "confidence": info["confidence"],
                    "reason": "CAMERA_TIMEOUT_EXCEEDED",
                }
                degraded_events.append(event_data)
                self.broadcast_state_change(
                    zone_id,
                    OperatingMode.MODE_2_VIBRATION_ONLY,
                    AlertLevel.UNVERIFIED_INTRUSION,
                    event_data,
                )

        for z in expired_zones:
            del self.pending_camera_verifications[z]

        return degraded_events
