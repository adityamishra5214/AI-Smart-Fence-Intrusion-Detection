"""
Perimeter Zone Grid Map Component for PIDS Dashboard.
Displays color-coded perimeter sectors:
- Green: NORMAL (All-Clear)
- Yellow: WIND_WARNING (Environmental flutter safely suppressed)
- Orange: SENSOR_FAULT (Cable cut, I2C freeze, or missing heartbeat)
- Red: CRITICAL_INTRUSION / UNVERIFIED_INTRUSION (Active breach detected)
"""

from typing import List, Dict, Any
import streamlit as st


STATUS_COLORS = {
    "NORMAL": "#28a745",          # Green
    "WIND_WARNING": "#ffc107",    # Yellow
    "SENSOR_FAULT": "#fd7e14",    # Orange
    "UNVERIFIED_INTRUSION": "#dc3545", # Red
    "CRITICAL_INTRUSION": "#721c24",   # Deep Red
    "FAULT": "#fd7e14",
    "HEALTHY": "#28a745",
}


def render_zone_map(zones: List[Dict[str, Any]]):
    """Renders the perimeter sectors grid."""
    st.subheader("Live Perimeter Map & Sector Status")

    cols = st.columns(len(zones))
    for idx, (col, zone) in enumerate(zip(cols, zones)):
        zone_id = zone.get("zone_id", f"Zone {idx+1}")
        status = zone.get("alert_level", zone.get("status", "NORMAL"))
        color = STATUS_COLORS.get(status, "#6c757d")
        mode = zone.get("operating_mode", "MODE_1")

        with col:
            st.markdown(
                f"""
                <div style="
                    border: 2px solid {color};
                    border-radius: 8px;
                    padding: 16px;
                    background-color: {color}15;
                    text-align: center;
                    margin-bottom: 12px;
                ">
                    <h4 style="margin: 0; color: #222;">{zone_id.upper()}</h4>
                    <p style="font-size: 18px; font-weight: bold; color: {color}; margin: 8px 0;">{status}</p>
                    <span style="font-size: 12px; color: #555;">Mode: {mode}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
