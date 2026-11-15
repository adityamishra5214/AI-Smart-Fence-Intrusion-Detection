"""
SQLAlchemy database engine, session factory, and repository methods.
Supports PostgreSQL with transparent SQLite fallback for hardware-free simulation.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from sqlalchemy import create_engine, select, desc
from sqlalchemy.orm import sessionmaker, Session
from backend.models import Base, NodeModel, TelemetryLogModel, EventModel, SystemFaultModel

SQLITE_PATH = Path(__file__).resolve().parent.parent / "pids.db"
DEFAULT_DB_URL = os.getenv("DATABASE_URL", f"sqlite:///{SQLITE_PATH.as_posix()}")

engine = create_engine(
    DEFAULT_DB_URL,
    echo=False,
    connect_args={"check_same_thread": False} if "sqlite" in DEFAULT_DB_URL else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initializes and creates all database tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI generator dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def upsert_node(node_id: str, zone_id: str, status: str = "ACTIVE", firmware_version: str = "1.0.0") -> NodeModel:
    """Inserts or updates node registration and last_seen timestamp."""
    with SessionLocal() as session:
        node = session.query(NodeModel).filter_by(node_id=node_id).first()
        now = datetime.now(timezone.utc)
        if node:
            node.status = status
            node.last_seen = now
        else:
            node = NodeModel(
                node_id=node_id,
                zone_id=zone_id,
                status=status,
                firmware_version=firmware_version,
                last_seen=now,
                created_at=now,
            )
            session.add(node)
        session.commit()
        return node


def log_telemetry_batch(records: List[Dict[str, Any]]):
    """High-throughput batch insertion of processed telemetry window metrics."""
    if not records:
        return
    with SessionLocal() as session:
        models = [
            TelemetryLogModel(
                node_id=r["node_id"],
                timestamp=datetime.fromtimestamp(r["timestamp"], tz=timezone.utc)
                if isinstance(r["timestamp"], (int, float))
                else r["timestamp"],
                rms_x=r.get("rms_x"),
                rms_y=r.get("rms_y"),
                rms_z=r.get("rms_z"),
                kurtosis_z=r.get("kurtosis_z"),
                zcr_z=r.get("zcr_z"),
                band1_ratio=r.get("band1_ratio"),
                band2_ratio=r.get("band2_ratio"),
                band3_ratio=r.get("band3_ratio"),
                dominant_freq=r.get("dominant_freq"),
                spectral_centroid=r.get("spectral_centroid"),
                predicted_class=r.get("predicted_class"),
            )
            for r in records
        ]
        session.bulk_save_objects(models)
        session.commit()


def log_event(
    zone_id: str,
    node_id: Optional[str],
    timestamp: float,
    operating_mode: str,
    vibration_class: str,
    camera_verified: bool,
    confidence: float,
    alert_level: str,
    metadata_json: Optional[Dict[str, Any]] = None,
) -> EventModel:
    """Inserts a security breach event."""
    with SessionLocal() as session:
        dt = (
            datetime.fromtimestamp(timestamp, tz=timezone.utc)
            if isinstance(timestamp, (int, float))
            else timestamp
        )
        event = EventModel(
            zone_id=zone_id,
            node_id=node_id,
            timestamp=dt,
            operating_mode=operating_mode,
            vibration_class=vibration_class,
            camera_verified=camera_verified,
            confidence=confidence,
            alert_level=alert_level,
            metadata_json=metadata_json,
        )
        session.add(event)
        session.commit()
        session.refresh(event)
        return event


def log_fault(node_id: str, fault_type: str, details: str) -> SystemFaultModel:
    """Logs a system or sensor fault."""
    with SessionLocal() as session:
        fault = SystemFaultModel(
            node_id=node_id,
            fault_type=fault_type,
            details=details,
            started_at=datetime.now(timezone.utc),
        )
        session.add(fault)
        session.commit()
        session.refresh(fault)
        return fault


def get_recent_events(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves recent security breach events."""
    with SessionLocal() as session:
        stmt = select(EventModel).order_by(desc(EventModel.timestamp)).limit(limit)
        events = session.execute(stmt).scalars().all()
        return [
            {
                "id": e.id,
                "zone_id": e.zone_id,
                "node_id": e.node_id,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                "operating_mode": e.operating_mode,
                "vibration_class": e.vibration_class,
                "camera_verified": e.camera_verified,
                "confidence": e.confidence,
                "alert_level": e.alert_level,
                "metadata": e.metadata_json,
            }
            for e in events
        ]


def get_all_nodes() -> List[Dict[str, Any]]:
    """Retrieves topology list of all registered nodes."""
    with SessionLocal() as session:
        stmt = select(NodeModel).order_by(NodeModel.zone_id)
        nodes = session.execute(stmt).scalars().all()
        return [
            {
                "node_id": n.node_id,
                "zone_id": n.zone_id,
                "status": n.status,
                "firmware_version": n.firmware_version,
                "last_seen": n.last_seen.isoformat() if n.last_seen else None,
            }
            for n in nodes
        ]
