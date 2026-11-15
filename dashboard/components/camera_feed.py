"""
Triggered Camera Snapshot Viewer Component with Bounding Box Overlay.
Displays optical verification snapshots and detection boxes.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import streamlit as st
from PIL import Image, ImageDraw


def render_camera_feed(snapshot_info: Optional[Dict[str, Any]] = None):
    """Renders the triggered camera snapshot with bounding box annotations."""
    st.subheader("Triggered Optical Verification Node (Raspberry Pi Camera)")

    frames_dir = Path(__file__).resolve().parent.parent.parent / "simulator" / "frames"
    default_frame = frames_dir / "human_intruder.jpg"

    if snapshot_info and "snapshot_id" in snapshot_info and Path(snapshot_info["snapshot_id"]).exists():
        img_path = Path(snapshot_info["snapshot_id"])
    elif default_frame.exists():
        img_path = default_frame
    else:
        st.info("No optical snapshot available.")
        return

    try:
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Draw bounding boxes if provided in metadata
        detections = snapshot_info.get("detections", []) if snapshot_info else []
        if not detections and "human" in img_path.name:
            # Default person box
            draw.rectangle([260, 140, 375, 340], outline="red", width=4)
            draw.text((265, 120), "PERSON 94%", fill="red")
        else:
            for det in detections:
                box = det.get("box", {})
                ymin = box.get("ymin", 0)
                xmin = box.get("xmin", 0)
                ymax = box.get("ymax", 0)
                xmax = box.get("xmax", 0)
                label = det.get("label", "detection").upper()
                conf = det.get("confidence", 0.0)
                draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=4)
                draw.text((xmin + 5, max(5, ymin - 15)), f"{label} {int(conf*100)}%", fill="red")

        st.image(img, caption="Sector Optical Surveillance Frame", use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load optical frame: {e}")
