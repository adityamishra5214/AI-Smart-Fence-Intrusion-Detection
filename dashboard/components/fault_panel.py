"""
Interactive Fault Injection Control Panel Component.
Allows operators to inject physical and network disruptions into simulated zones.
"""

from typing import Callable, Optional
import streamlit as st


def render_fault_panel(on_inject_callback: Optional[Callable[[str, str], None]] = None):
    """Renders the fault injection control panel."""
    st.subheader("Interactive Fault-Injection Panel (Hardware-in-the-Loop)")

    col_target, col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([1.5, 1, 1, 1, 1])

    with col_target:
        target_node = st.selectbox(
            "Target Sensor Node",
            ["zone_01_accel", "zone_02_accel", "zone_03_accel"],
            key="fault_target_node",
        )

    with col_btn1:
        if st.button("Simulate Cable Cut (Flatline)", use_container_width=True, type="primary"):
            if on_inject_callback:
                on_inject_callback(target_node, "flatline")
            st.warning(f"Injected I2C Flatline failure into {target_node}!")

    with col_btn2:
        if st.button("Simulate Windstorm", use_container_width=True):
            if on_inject_callback:
                on_inject_callback(target_node, "wind")
            st.info(f"Simulating heavy windstorm on {target_node}...")

    with col_btn3:
        if st.button("Simulate Climbing", use_container_width=True):
            if on_inject_callback:
                on_inject_callback(target_node, "climb")
            st.error(f"Injected climbing intrusion on {target_node}!")

    with col_btn4:
        if st.button("Simulate Network Drop", use_container_width=True):
            if on_inject_callback:
                on_inject_callback(target_node, "network_drop")
            st.error(f"Severed network connection on {target_node} (3s TTL trigger)!")
