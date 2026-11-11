"""
Pydantic schemas and JSON validation for all MQTT message topics across the PIDS network.

Topics:
- fence/node/{node_id}/telemetry
- fence/node/{node_id}/heartbeat
- fence/node/{node_id}/status (LWT)
- fence/system/trigger_cam
- fence/node/{node_id}/camera_alert
"""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class NodeStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    FAULT = "FAULT"
    DEAD = "DEAD"


class SignalType(str, Enum):
    NORMAL = "NORMAL"
    WIND_RAIN = "WIND_RAIN"
    CLIMBING = "CLIMBING"
    CUTTING = "CUTTING"


class TelemetryDataPayload(BaseModel):
    """Raw 3-axis accelerometer window arrays sampled at target fs."""
    ax: List[float] = Field(..., description="Acceleration along X-axis in g")
    ay: List[float] = Field(..., description="Acceleration along Y-axis in g")
    az: List[float] = Field(..., description="Acceleration along Z-axis in g")


class TelemetryMessage(BaseModel):
    """Topic: fence/node/{node_id}/telemetry"""
    node_id: str = Field(..., description="Identifier of the publishing sensor node")
    zone_id: str = Field(..., description="Perimeter zone identifier, e.g. zone_01")
    timestamp: float = Field(..., description="Unix epoch timestamp in seconds")
    fs: float = Field(default=100.0, description="Sampling rate in Hz")
    window_size: int = Field(default=200, description="Number of samples per window")
    data: TelemetryDataPayload


class HeartbeatMessage(BaseModel):
    """Topic: fence/node/{node_id}/heartbeat (sent at 1 Hz)"""
    node_id: str = Field(..., description="Identifier of the reporting node")
    zone_id: str = Field(..., description="Assigned perimeter zone")
    timestamp: float = Field(..., description="Unix epoch timestamp in seconds")
    uptime_seconds: int = Field(..., ge=0, description="Continuous node uptime")
    battery_pct: float = Field(..., ge=0.0, le=100.0, description="Battery percentage remaining")
    status: NodeStatus = Field(default=NodeStatus.HEALTHY, description="Self-reported health")
    firmware_version: str = Field(default="1.0.0", description="Firmware version string")


class StatusLWTMessage(BaseModel):
    """Topic: fence/node/{node_id}/status (Retained MQTT Last Will and Testament)"""
    node_id: str = Field(..., description="Identifier of the node")
    zone_id: str = Field(..., description="Assigned perimeter zone")
    timestamp: float = Field(..., description="Unix epoch timestamp of event")
    status: NodeStatus = Field(default=NodeStatus.DEAD, description="LWT status (DEAD upon unexpected drop)")
    reason: str = Field(default="CONNECTION_LOST", description="Disconnection reason")


class TriggerCameraMessage(BaseModel):
    """Topic: fence/system/trigger_cam (Vibration breach -> Wake optical verification)"""
    zone_id: str = Field(..., description="Perimeter zone where vibration anomaly occurred")
    trigger_node_id: str = Field(..., description="ID of vibration node triggering camera")
    timestamp: float = Field(..., description="Unix epoch timestamp of trigger")
    detected_class: SignalType = Field(..., description="Classified breach type: CLIMBING or CUTTING")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classifier confidence score")


class BoundingBox(BaseModel):
    """Normalized or pixel coordinates [ymin, xmin, ymax, xmax]"""
    ymin: float
    xmin: float
    ymax: float
    xmax: float


class DetectionItem(BaseModel):
    label: str = Field(..., description="Detected object label, e.g. person")
    confidence: float = Field(..., ge=0.0, le=1.0)
    box: BoundingBox


class CameraAlertMessage(BaseModel):
    """Topic: fence/node/{node_id}/camera_alert (Optical verification payload)"""
    node_id: str = Field(..., description="Camera node identifier")
    zone_id: str = Field(..., description="Assigned perimeter zone")
    timestamp: float = Field(..., description="Unix epoch timestamp of capture")
    verified: bool = Field(..., description="True if target person/threat detected")
    detections: List[DetectionItem] = Field(default_factory=list)
    snapshot_id: Optional[str] = Field(None, description="Snapshot reference or ID")

