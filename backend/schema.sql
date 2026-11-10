-- PostgreSQL schema definition for AI-Based Perimeter Fence Intrusion Detection System (PIDS)

CREATE TABLE IF NOT EXISTS nodes (
    node_id VARCHAR(64) PRIMARY KEY,
    zone_id VARCHAR(64) NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'ACTIVE',
    ip_address VARCHAR(45),
    firmware_version VARCHAR(32),
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS telemetry_logs (
    id BIGSERIAL PRIMARY KEY,
    node_id VARCHAR(64) REFERENCES nodes(node_id),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    rms_x DOUBLE PRECISION,
    rms_y DOUBLE PRECISION,
    rms_z DOUBLE PRECISION,
    kurtosis_z DOUBLE PRECISION,
    zcr_z INTEGER,
    band1_ratio DOUBLE PRECISION,
    band2_ratio DOUBLE PRECISION,
    band3_ratio DOUBLE PRECISION,
    spectral_centroid DOUBLE PRECISION,
    predicted_class VARCHAR(32)
);

CREATE TABLE IF NOT EXISTS events (
    id BIGSERIAL PRIMARY KEY,
    zone_id VARCHAR(64) NOT NULL,
    node_id VARCHAR(64) REFERENCES nodes(node_id),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    operating_mode VARCHAR(32) NOT NULL, -- MODE_1, MODE_2, MODE_3
    vibration_class VARCHAR(32) NOT NULL,
    camera_verified BOOLEAN DEFAULT FALSE,
    confidence DOUBLE PRECISION,
    alert_level VARCHAR(32) NOT NULL, -- CRITICAL_INTRUSION, UNVERIFIED_INTRUSION, SENSOR_FAULT
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS fault_logs (
    id BIGSERIAL PRIMARY KEY,
    node_id VARCHAR(64) REFERENCES nodes(node_id),
    fault_type VARCHAR(64) NOT NULL,
    details TEXT,
    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
    resolved_at TIMESTAMP WITH TIME ZONE
);

