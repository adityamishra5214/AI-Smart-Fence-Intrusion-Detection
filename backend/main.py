"""
FastAPI core application for AI-Based Perimeter Fence Intrusion Detection System (PIDS).

Provides REST APIs and background ingestion task management:
- /api/health: System health and connectivity
- /api/nodes: Real-time zone and node vitality tracker
- /api/events: Historical security breaches and multi-modal alerts
- /api/telemetry: Raw DSP features and predicted classes
- /api/fault-injection: Operator fault injection console
"""

import sys
from pathlib import Path
import os
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from backend.db import init_db, get_recent_events, get_all_nodes, log_event, log_fault
from backend.mqtt_client import IngestionWorker
from backend.fusion import AlertLevel, OperatingMode

worker: Optional[IngestionWorker] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global worker
    print("[FastAPI] Initializing Database & Telemetry Ingestion Worker...")
    init_db()

    broker_host = os.getenv("MQTT_BROKER_HOST", "localhost")
    broker_port = int(os.getenv("MQTT_BROKER_PORT", 1883))
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))

    worker = IngestionWorker(
        broker_host=broker_host,
        broker_port=broker_port,
        redis_host=redis_host,
        redis_port=redis_port,
    )
    worker.start()

    yield

    print("[FastAPI] Shutting down Ingestion Worker...")
    if worker:
        worker.stop()


app = FastAPI(
    title="AI-Based Perimeter Fence Intrusion Detection System (PIDS)",
    description="Multi-Modal Edge AI & Fault-Tolerant Distributed Architecture API",
    version="1.0.0",
    lifespan=lifespan,
)


class FaultInjectionRequest(BaseModel):
    node_id: str = Field(..., example="zone_01_accel")
    fault_type: str = Field(..., example="flatline", description="flatline | network_drop | jitter | storm")


class CameraVerificationRequest(BaseModel):
    zone_id: str = Field(..., example="zone_01")
    verified: bool = Field(..., description="True if person threat detected")
    confidence: float = Field(default=0.92, ge=0.0, le=1.0)


@app.get("/")
def read_root():
    return {
        "system": "AI-Based Perimeter Fence Intrusion Detection System (PIDS)",
        "version": "1.0.0",
        "status": "OPERATIONAL",
        "operating_modes": {
            "Mode 1": "Full Multi-Modal Fusion (Accel + Camera)",
            "Mode 2": "Degraded: Vibration Only (Camera blind / timeout)",
            "Mode 3": "Degraded: Sensor Fault (TTL drop or I2C flatline)",
        },
        "endpoints": ["/api/health", "/api/nodes", "/api/events", "/docs"],
    }


@app.get("/api/health")
def get_health():
    is_worker_alive = worker is not None and worker.running
    return {
        "status": "HEALTHY",
        "timestamp": time.time(),
        "ingestion_worker_active": is_worker_alive,
        "processed_windows": worker.processed_count if worker else 0,
        "logged_events": worker.event_count if worker else 0,
    }


@app.get("/api/nodes")
def get_nodes():
    """Returns real-time node health, vitality TTL, and assigned zone statuses."""
    db_nodes = get_all_nodes()
    live_statuses = worker.state_tracker.get_node_statuses() if worker else {}

    result = []
    for node in db_nodes:
        nid = node["node_id"]
        live = live_statuses.get(nid, {})
        result.append({
            "node_id": nid,
            "zone_id": node["zone_id"],
            "status": live.get("status", node["status"]),
            "alive": live.get("alive", False),
            "seconds_since_heartbeat": live.get("seconds_since_heartbeat"),
            "firmware_version": node["firmware_version"],
            "last_seen": node["last_seen"],
        })
    return result


@app.get("/api/events")
def get_events(limit: int = Query(50, ge=1, le=500)):
    """Retrieves recent security breach events and classified intrusions."""
    return get_recent_events(limit=limit)


@app.post("/api/fault-injection")
def inject_fault(req: FaultInjectionRequest):
    """Logs simulated hardware cut or network fault."""
    log_fault(req.node_id, req.fault_type.upper(), f"Manual fault injected via REST API: {req.fault_type}")
    if worker:
        if req.fault_type == "network_drop":
            worker.state_tracker.mark_dead(req.node_id)
    return {"status": "FAULT_INJECTED", "node_id": req.node_id, "fault_type": req.fault_type}


@app.post("/api/verify-camera")
def verify_camera(req: CameraVerificationRequest):
    """Processes optical verification from surveillance camera nodes."""
    if not worker:
        raise HTTPException(status_code=503, detail="Ingestion worker unavailable")

    detections = [{"label": "person", "confidence": req.confidence}] if req.verified else []
    mode, alert_lvl, action = worker.fusion.evaluate_camera_response(
        zone_id=req.zone_id,
        verified=req.verified,
        detections=detections,
    )

    if alert_lvl == AlertLevel.CRITICAL_INTRUSION:
        log_event(
            zone_id=req.zone_id,
            node_id=f"{req.zone_id}_cam",
            timestamp=time.time(),
            operating_mode=mode.value,
            vibration_class="CAMERA_VERIFIED",
            camera_verified=True,
            confidence=req.confidence,
            alert_level=alert_lvl.value,
            metadata_json={"action": action, "detections": detections},
        )

    return {
        "zone_id": req.zone_id,
        "operating_mode": mode.value,
        "alert_level": alert_lvl.value,
        "action": action,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=False)
