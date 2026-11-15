"""
Streaming 3-Axis Accelerometer Chart Component.
Renders real-time ax, ay, az series sampled at 100 Hz across the 2.0-second sliding window.
"""

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import streamlit as st


def render_accel_chart(window_data: Optional[Dict[str, Any]] = None):
    """Renders 3-axis acceleration chart."""
    st.subheader("Streaming 3-Axis Accelerometer Waveforms (MPU6050 @ 100 Hz)")

    if not window_data or "ax" not in window_data:
        # Default placeholder waveform if no streaming packet yet
        t = np.arange(200) / 100.0
        ax = np.random.normal(0, 0.02, 200)
        ay = np.random.normal(0, 0.02, 200)
        az = 1.0 + np.random.normal(0, 0.02, 200)
    else:
        ax = np.array(window_data["ax"])
        ay = np.array(window_data["ay"])
        az = np.array(window_data["az"])
        t = np.arange(len(ax)) / 100.0

    df = pd.DataFrame({
        "Time (s)": t,
        "a_x (Lateral g)": ax,
        "a_y (Longitudinal g)": ay,
        "a_z (Vertical g)": az,
    }).set_index("Time (s)")

    st.line_chart(df, height=280)

    # Metric summaries below chart
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("RMS (a_x)", f"{float(np.sqrt(np.mean(ax**2))):.3f} g")
    with col2:
        st.metric("RMS (a_y)", f"{float(np.sqrt(np.mean(ay**2))):.3f} g")
    with col3:
        st.metric("RMS (a_z)", f"{float(np.sqrt(np.mean(az**2))):.3f} g")
    with col4:
        st.metric("Peak-to-Peak", f"{float(np.ptp(az)):.3f} g")
