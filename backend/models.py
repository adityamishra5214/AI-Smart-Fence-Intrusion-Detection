"""
SQLAlchemy ORM models: nodes, telemetry_logs, events, and system_faults.
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any
from sqlalchemy import (
    Column,
    String,
    Integer,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    Text,
    JSON,
    BigInteger,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class NodeModel(Base):
    __tablename__ = "nodes"

    node_id = Column(String(64), primary_key=True)
    zone_id = Column(String(64), nullable=False, index=True)
    status = Column(String(32), nullable=False, default="ACTIVE")
    ip_address = Column(String(45), nullable=True)
    firmware_version = Column(String(32), nullable=True, default="1.0.0")
    last_seen = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    telemetry = relationship("TelemetryLogModel", back_populates="node", cascade="all, delete-orphan")
    events = relationship("EventModel", back_populates="node", cascade="all, delete-orphan")
    faults = relationship("SystemFaultModel", back_populates="node", cascade="all, delete-orphan")


class TelemetryLogModel(Base):
    __tablename__ = "telemetry_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    node_id = Column(String(64), ForeignKey("nodes.node_id"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    rms_x = Column(Float, nullable=True)
    rms_y = Column(Float, nullable=True)
    rms_z = Column(Float, nullable=True)
    kurtosis_z = Column(Float, nullable=True)
    zcr_z = Column(Integer, nullable=True)
    band1_ratio = Column(Float, nullable=True)
    band2_ratio = Column(Float, nullable=True)
    band3_ratio = Column(Float, nullable=True)
    dominant_freq = Column(Float, nullable=True)
    spectral_centroid = Column(Float, nullable=True)
    predicted_class = Column(String(32), nullable=True)

    node = relationship("NodeModel", back_populates="telemetry")


class EventModel(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    zone_id = Column(String(64), nullable=False, index=True)
    node_id = Column(String(64), ForeignKey("nodes.node_id"), nullable=True, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    operating_mode = Column(String(32), nullable=False)  # MODE_1, MODE_2, MODE_3
    vibration_class = Column(String(32), nullable=False)
    camera_verified = Column(Boolean, default=False)
    confidence = Column(Float, default=1.0)
    alert_level = Column(String(32), nullable=False)  # CRITICAL_INTRUSION, UNVERIFIED_INTRUSION, SENSOR_FAULT
    metadata_json = Column(JSON, nullable=True)

    node = relationship("NodeModel", back_populates="events")


class SystemFaultModel(Base):
    __tablename__ = "system_faults"

    id = Column(Integer, primary_key=True, autoincrement=True)
    node_id = Column(String(64), ForeignKey("nodes.node_id"), nullable=True, index=True)
    fault_type = Column(String(64), nullable=False)
    details = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    resolved_at = Column(DateTime, nullable=True)

    node = relationship("NodeModel", back_populates="faults")
