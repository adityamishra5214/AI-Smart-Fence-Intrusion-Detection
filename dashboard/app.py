"""
Streamlit real-time operator monitoring console for AI-Based PIDS.

Renders:
1. Live Perimeter Map (Green: Normal | Yellow: Wind | Orange: Fault | Red: Alert)
2. Streaming 3-Axis Accelerometer Chart (ax, ay, az waveforms @ 100 Hz)
3. Triggered Optical Camera Snapshot with Person Bounding Boxes
4. Interactive Fault-Injection Panel (Simulate Cable Cut, Storm, Dropped Broker)
5. Historical Security Breach & Alarm Audit Log Table
"""

import sys
from pathlib import Path
import time
import pandas as pd
import streamlit as st

# Ensure project root in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from backend.db import init_db, get_recent_events, get_all_nodes, log_fault
from dashboard.components.map_view import render_zone_map
from dashboard.components.accel_chart import render_accel_chart
from dashboard.components.camera_feed import render_camera_feed
from dashboard.components.fault_panel import render_fault_panel
from simulator.generator import VibrationSynthesizer, SignalType

# Page configuration
st.set_page_config(
    page_title="AI-Based PIDS Operator Console",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize database
init_db()

# Top Title & Operational Status
st.title("🛡️ AI-Based Perimeter Fence Intrusion Detection System (PIDS)")
st.caption("Multi-Modal Edge AI & Fault-Tolerant Distributed Architecture | Operator Security Console")

# Sidebar - System Controls
st.sidebar.header("System Controls & Status")
st.sidebar.markdown("**Operational Architecture Modes:**")
st.sidebar.markdown("- **Mode 1 (Full Fusion):** Accel + Optical Active")
st.sidebar.markdown("- **Mode 2 (Degraded):** Accel Only (Camera Blind)")
st.sidebar.markdown("- **Mode 3 (Degraded):** Sensor Fault (Never 'All-Clear')")

refresh_rate = st.sidebar.slider("Auto-Refresh Interval (s)", 1, 10, 3)
st.sidebar.divider()

# Fault Callback
def handle_fault_injection(node_id: str, fault_type: str):
    log_fault(node_id, fault_type.upper(), f"Injected via Operator Dashboard: {fault_type}")

# Render Fault Injection Panel at top for operator testing
render_fault_panel(on_inject_callback=handle_fault_injection)
st.divider()

# Query Database for Nodes and Events
nodes_data = get_all_nodes()
if not nodes_data:
    # Default topology if not yet populated
    nodes_data = [
        {"node_id": "zone_01_accel", "zone_id": "zone_01", "status": "NORMAL"},
        {"node_id": "zone_02_accel", "zone_id": "zone_02", "status": "WIND_WARNING"},
        {"node_id": "zone_03_accel", "zone_id": "zone_03", "status": "NORMAL"},
    ]

recent_events = get_recent_events(limit=25)

# 1. Render Zone Map
render_zone_map(nodes_data)
st.divider()

# 2. Main Analytics Columns (Vibration Chart + Camera Feed)
col_vibe, col_cam = st.columns([1.2, 1.0])

synth = VibrationSynthesizer(fs=100.0)

with col_vibe:
    # Check if latest event is climbing or wind to show dynamic sample
    selected_signal = SignalType.NORMAL
    if recent_events and recent_events[0].get("vibration_class") == "CLIMBING":
        selected_signal = SignalType.CLIMBING
    elif recent_events and recent_events[0].get("vibration_class") == "CUTTING":
        selected_signal = SignalType.CUTTING

    sample_arr = synth.generate_sample(selected_signal, num_samples=200)
    window_data = {
        "ax": sample_arr[:, 0].tolist(),
        "ay": sample_arr[:, 1].tolist(),
        "az": sample_arr[:, 2].tolist(),
    }
    render_accel_chart(window_data)

with col_cam:
    snapshot_info = None
    if recent_events and recent_events[0].get("camera_verified"):
        snapshot_info = {
            "snapshot_id": str(_project_root / "simulator" / "frames" / "human_intruder.jpg"),
            "detections": recent_events[0].get("metadata", {}).get("detections", []),
        }
    render_camera_feed(snapshot_info)

st.divider()

# 3. Historical Alarm & Audit Log Table
st.subheader("Historical Alarm & Security Breach Audit Log")

if recent_events:
    df_events = pd.DataFrame(recent_events)
    # Format display columns
    display_cols = ["id", "timestamp", "zone_id", "operating_mode", "alert_level", "vibration_class", "camera_verified", "confidence"]
    available_cols = [c for c in display_cols if c in df_events.columns]
    st.dataframe(df_events[available_cols], use_container_width=True)
else:
    st.info("No security breach events logged in the audit database yet.")

# Rerun timer
time.sleep(refresh_rate)
st.rerun()
