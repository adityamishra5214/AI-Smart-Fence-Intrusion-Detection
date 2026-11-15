"""
Event-triggered mock Raspberry Pi Camera / YOLO optical verification node.

Subscribes to:
- fence/system/trigger_cam (triggered by vibration breach in adjacent zone)

Publishes to:
- fence/node/{node_id}/camera_alert (optical bounding box verification payload)
- fence/node/{node_id}/heartbeat (optical node vitality keepalive)

Features:
- Renders and maintains synthetic test surveillance frames (empty fence, humans, animals, vehicles).
- Fast optical inference pipeline (generates detection bounding boxes, confidence, class label).
- Measures awakening-to-inference latency (< 300 ms target).
"""

import sys
from pathlib import Path
import os
import time
import json
import threading
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    import paho.mqtt.client as mqtt
    PAHO_AVAILABLE = True
except ImportError:
    PAHO_AVAILABLE = False

from simulator.schemas import (
    TriggerCameraMessage,
    CameraAlertMessage,
    BoundingBox,
    DetectionItem,
    HeartbeatMessage,
    NodeStatus,
)

FRAMES_DIR = _project_root / "simulator" / "frames"


def generate_surveillance_frames():
    """Generates synthetic surveillance test frames for optical evaluation."""
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    # Frame dimensions (VGA surveillance standard)
    width, height = 640, 480

    def draw_fence_background(draw):
        # Sky and ground
        draw.rectangle([0, 0, width, height // 2], fill=(135, 180, 222))  # Sky
        draw.rectangle([0, height // 2, width, height], fill=(68, 115, 60))  # Grass
        # Perimeter chain-link fence mesh
        for x in range(0, width, 20):
            draw.line([(x, height // 3), (x + 15, height)], fill=(180, 180, 180), width=1)
            draw.line([(x + 15, height // 3), (x, height)], fill=(180, 180, 180), width=1)
        # Fence steel posts
        for px in (120, 320, 520):
            draw.rectangle([px - 4, height // 3 - 20, px + 4, height], fill=(90, 90, 90))
            draw.line([(px - 8, height // 3 - 20), (px + 8, height // 3 - 20)], fill=(70, 70, 70), width=3)

    # 1. Empty Fence
    img_empty = Image.new("RGB", (width, height))
    d_empty = ImageDraw.Draw(img_empty)
    draw_fence_background(d_empty)
    d_empty.text((15, 15), "[CAM-01] Perimeter Sector Normal", fill=(255, 255, 255))
    img_empty.save(FRAMES_DIR / "empty_fence.jpg", quality=85)

    # 2. Human Climbing Fence
    img_human = Image.new("RGB", (width, height))
    d_human = ImageDraw.Draw(img_human)
    draw_fence_background(d_human)
    # Draw silhouette of person scaling the fence
    # Head
    d_human.ellipse([300, 140, 335, 175], fill=(30, 30, 30))
    # Torso
    d_human.rectangle([295, 175, 340, 270], fill=(20, 30, 60))
    # Limbs scaling fence
    d_human.line([(295, 190), (260, 160)], fill=(30, 30, 30), width=6)  # Left arm gripping wire
    d_human.line([(340, 190), (370, 150)], fill=(30, 30, 30), width=6)  # Right arm gripping top
    d_human.line([(305, 270), (280, 330)], fill=(20, 20, 20), width=7)  # Left leg
    d_human.line([(330, 270), (355, 310)], fill=(20, 20, 20), width=7)  # Right leg
    d_human.text((15, 15), "[CAM-01] Person Detected - Scaling Fence", fill=(255, 50, 50))
    img_human.save(FRAMES_DIR / "human_intruder.jpg", quality=85)

    # 3. Animal Grazing (Deer/Dog)
    img_animal = Image.new("RGB", (width, height))
    d_animal = ImageDraw.Draw(img_animal)
    draw_fence_background(d_animal)
    # Draw animal silhouette on ground
    d_animal.ellipse([380, 320, 470, 380], fill=(139, 69, 19))  # Body
    d_animal.ellipse([460, 290, 495, 330], fill=(139, 69, 19))  # Head
    d_animal.line([(395, 375), (390, 430)], fill=(110, 55, 15), width=5)
    d_animal.line([(450, 375), (455, 430)], fill=(110, 55, 15), width=5)
    d_animal.text((15, 15), "[CAM-01] Wildlife Detected - Non-Threat", fill=(255, 255, 0))
    img_animal.save(FRAMES_DIR / "animal_grazing.jpg", quality=85)

    # 4. Vehicle Patrol
    img_vehicle = Image.new("RGB", (width, height))
    d_vehicle = ImageDraw.Draw(img_vehicle)
    draw_fence_background(d_vehicle)
    d_vehicle.rectangle([180, 310, 420, 390], fill=(75, 83, 32))  # Olive patrol jeep
    d_vehicle.rectangle([220, 260, 370, 310], fill=(95, 105, 45))  # Cabin
    d_vehicle.ellipse([210, 375, 260, 425], fill=(20, 20, 20))  # Wheel
    d_vehicle.ellipse([340, 375, 390, 425], fill=(20, 20, 20))  # Wheel
    d_vehicle.text((15, 15), "[CAM-01] Authorized Vehicle Patrol", fill=(100, 255, 100))
    img_vehicle.save(FRAMES_DIR / "vehicle_patrol.jpg", quality=85)


class MockOpticalDetector:
    """
    Lightweight edge optical detector emulating YOLOv8n / MobileNet inference.
    Processes surveillance frame samples and outputs person / animal detections.
    """

    def __init__(self):
        generate_surveillance_frames()

    def detect(self, scene_type: str = "human") -> Tuple[List[DetectionItem], str, float]:
        """
        Runs optical inference on the specified scene.
        Returns: (detections, snapshot_file_path, inference_latency_ms)
        """
        t0 = time.perf_counter()

        detections = []
        if scene_type == "human":
            img_path = str(FRAMES_DIR / "human_intruder.jpg")
            # Person bounding box [ymin, xmin, ymax, xmax] in pixels on 640x480
            box = BoundingBox(ymin=140.0, xmin=260.0, ymax=340.0, xmax=375.0)
            detections.append(DetectionItem(label="person", confidence=0.94, box=box))
        elif scene_type == "animal":
            img_path = str(FRAMES_DIR / "animal_grazing.jpg")
            box = BoundingBox(ymin=290.0, xmin=380.0, ymax=430.0, xmax=495.0)
            detections.append(DetectionItem(label="animal", confidence=0.88, box=box))
        elif scene_type == "vehicle":
            img_path = str(FRAMES_DIR / "vehicle_patrol.jpg")
            box = BoundingBox(ymin=260.0, xmin=180.0, ymax=425.0, xmax=420.0)
            detections.append(DetectionItem(label="vehicle", confidence=0.92, box=box))
        else:
            img_path = str(FRAMES_DIR / "empty_fence.jpg")

        # Emulate optical image decode and CNN forward pass delay (30-80 ms on edge ARM)
        time.sleep(0.045)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return detections, img_path, latency_ms


class MockCameraNode:
    """
    Virtual surveillance camera node deployed in a perimeter zone.
    Awakes upon receiving fence/system/trigger_cam.
    """

    def __init__(
        self,
        node_id: str = "zone_01_cam",
        zone_id: str = "zone_01",
        broker_host: str = "localhost",
        broker_port: int = 1883,
    ):
        self.node_id = node_id
        self.zone_id = zone_id
        self.broker_host = broker_host
        self.broker_port = broker_port

        self.detector = MockOpticalDetector()
        self.client: Optional[mqtt.Client] = None
        self.connected = False
        self.running = False

        self.trigger_count = 0
        self.verified_count = 0
        self.latencies: List[float] = []

        self._heartbeat_thread: Optional[threading.Thread] = None

    def _setup_mqtt(self):
        if not PAHO_AVAILABLE:
            print(f"[{self.node_id}] Paho-MQTT not installed. Offline simulation mode.")
            return

        try:
            self.client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id=f"cam_{self.node_id}_{int(time.time())}",
            )
        except (AttributeError, TypeError):
            self.client = mqtt.Client(client_id=f"cam_{self.node_id}_{int(time.time())}")

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        try:
            self.client.connect(self.broker_host, self.broker_port, keepalive=15)
            self.client.loop_start()
            self.connected = True
            print(f"[{self.node_id}] Camera node connected to broker at {self.broker_host}:{self.broker_port}")
        except Exception as e:
            print(f"[{self.node_id}] Warning: MQTT broker connection failed ({e}). Simulating locally.")

    def on_connect(self, client, userdata, flags, rc, properties=None):
        print(f"[{self.node_id}] Subscribed to fence/system/trigger_cam")
        client.subscribe("fence/system/trigger_cam")

    def on_message(self, client, userdata, msg):
        topic = msg.topic
        payload_str = msg.payload.decode("utf-8", errors="ignore")

        if topic == "fence/system/trigger_cam":
            try:
                data = json.loads(payload_str)
                trigger = TriggerCameraMessage.model_validate(data)
                if trigger.zone_id == self.zone_id:
                    self.process_trigger(trigger)
            except Exception as e:
                print(f"[{self.node_id}] Error handling trigger: {e}")

    def process_trigger(self, trigger: TriggerCameraMessage) -> CameraAlertMessage:
        """Processes awakened optical verification request."""
        t_wake = time.perf_counter()
        self.trigger_count += 1

        # Select scene based on trigger class
        # CLIMBING or CUTTING indicates person intrusion attempt
        scene = "human" if trigger.detected_class.value in ("CLIMBING", "CUTTING") else "empty"

        detections, snap_path, inf_latency = self.detector.detect(scene_type=scene)
        is_person_verified = any(d.label == "person" and d.confidence >= 0.70 for d in detections)

        total_latency_ms = (time.perf_counter() - t_wake) * 1000.0
        self.latencies.append(total_latency_ms)

        if is_person_verified:
            self.verified_count += 1

        alert = CameraAlertMessage(
            node_id=self.node_id,
            zone_id=self.zone_id,
            timestamp=time.time(),
            verified=is_person_verified,
            detections=detections,
            snapshot_id=snap_path,
        )

        topic = f"fence/node/{self.node_id}/camera_alert"
        if self.connected and self.client:
            self.client.publish(topic, alert.model_dump_json(), qos=1)

        print(
            f"[{self.node_id}] Trigger processed in {total_latency_ms:.2f} ms | "
            f"Verified Person: {is_person_verified} | Target (<300ms): "
            f"{'PASSED' if total_latency_ms < 300.0 else 'FAILED'}"
        )
        return alert

    def _heartbeat_loop(self):
        """Periodically emits vitality heartbeat so Fusion Engine maintains Mode 1 capability."""
        while self.running:
            hb = HeartbeatMessage(
                node_id=self.node_id,
                zone_id=self.zone_id,
                timestamp=time.time(),
                uptime_seconds=int(time.time()),
                battery_pct=100.0,
                status=NodeStatus.HEALTHY,
                firmware_version="camera-v2.1",
            )
            topic = f"fence/node/{self.node_id}/heartbeat"
            if self.connected and self.client:
                self.client.publish(topic, hb.model_dump_json(), qos=0)
            time.sleep(2.0)

    def start(self):
        self.running = True
        self._setup_mqtt()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        print(f"[{self.node_id}] Mock Camera Node started.")

    def stop(self):
        self.running = False
        if self.connected and self.client:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass
        print(f"[{self.node_id}] Stopped. Total triggers: {self.trigger_count}, Verified: {self.verified_count}")


def benchmark_camera_latency(num_runs: int = 50) -> Dict[str, float]:
    """Measures camera awakening to inference latency across multiple trials."""
    cam = MockCameraNode()
    latencies = []

    from simulator.schemas import SignalType

    for _ in range(num_runs):
        trigger = TriggerCameraMessage(
            zone_id="zone_01",
            trigger_node_id="zone_01_accel",
            timestamp=time.time(),
            detected_class=SignalType.CLIMBING,
            confidence=0.96,
        )
        t0 = time.perf_counter()
        cam.process_trigger(trigger)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    lats = np.array(latencies)
    return {
        "mean_ms": float(np.mean(lats)),
        "median_ms": float(np.median(lats)),
        "p95_ms": float(np.percentile(lats, 95)),
        "min_ms": float(np.min(lats)),
        "max_ms": float(np.max(lats)),
        "target_met": bool(np.mean(lats) < 300.0),
    }


if __name__ == "__main__":
    import numpy as np

    print("=" * 60)
    print("AI-Smart-Fence: Mock Camera Node Latency Benchmark")
    print("=" * 60)
    results = benchmark_camera_latency(num_runs=20)
    print(f"Mean Awakening-to-Alert Latency: {results['mean_ms']:.2f} ms")
    print(f"Median Latency                  : {results['median_ms']:.2f} ms")
    print(f"95th Percentile                 : {results['p95_ms']:.2f} ms")
    print(f"Target (< 300 ms)               : {'PASSED' if results['target_met'] else 'FAILED'}")
    print("=" * 60)
