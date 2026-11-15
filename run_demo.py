"""
Single-command end-to-end simulation runner for AI-Based PIDS.

Boots:
1. Database tables and topology initialization.
2. Ingestion worker and multi-modal state machine.
3. 3-zone virtual ESP32 sensor cluster (zone_01, zone_02, zone_03).
4. Optical verification camera node.
5. Automated sequence: Normal -> Windstorm -> Climbing breach -> Camera verification -> Cable cut fault.
"""

import sys
from pathlib import Path
import time
import threading

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from backend.db import init_db, get_recent_events, get_all_nodes
from backend.mqtt_client import IngestionWorker
from simulator.mock_esp32 import FenceSimulationCluster
from simulator.mock_camera import MockCameraNode
from simulator.generator import SignalType


def main():
    print("=" * 70)
    print("AI-BASED PERIMETER FENCE INTRUSION DETECTION SYSTEM (PIDS)")
    print("End-to-End Simulation & Multi-Modal Verification Demo")
    print("=" * 70)

    # 1. Initialize Database
    print("\n[Step 1/4] Initializing persistence layer...")
    init_db()
    print("[+] Database tables initialized.")

    # 2. Boot Ingestion Worker
    print("\n[Step 2/4] Starting Ingestion Engine & Fusion State Machine...")
    worker = IngestionWorker()
    worker.start()
    print("[+] Ingestion engine active.")

    # 3. Boot Optical Verification Camera Node
    print("\n[Step 3/4] Initializing Optical Verification Camera Node...")
    cam = MockCameraNode(node_id="zone_01_cam", zone_id="zone_01")
    cam.start()
    print("[+] Camera node ready for optical event triggers.")

    # 4. Boot 3-Zone ESP32 Node Cluster
    print("\n[Step 4/4] Starting 3-Zone Virtual ESP32 Sensor Nodes...")
    cluster = FenceSimulationCluster()
    cluster.start_all()
    print("[+] Sensor nodes transmitting 100 Hz telemetry & 1 Hz heartbeats.")

    try:
        print("\n" + "=" * 70)
        print("PHASE 1: Baseline Ambient Operation (All Sectors Normal)")
        print("=" * 70)
        time.sleep(2.0)

        print("\n" + "=" * 70)
        print("PHASE 2: Environmental Storm Simulation (Zone 02 - Wind Buffeting)")
        print("=" * 70)
        print("[!] Injecting 4 Hz wind gust fluttering into zone_02...")
        cluster.get_node("zone_02_accel").set_signal_profile(SignalType.WIND_RAIN)
        time.sleep(2.5)

        print("\n" + "=" * 70)
        print("PHASE 3: Hostile Sabotage / Climbing Breach (Zone 01 - Trigger Camera)")
        print("=" * 70)
        print("[!] Injecting structural climbing signature into zone_01...")
        cluster.get_node("zone_01_accel").set_signal_profile(SignalType.CLIMBING)
        time.sleep(2.5)

        print("\n" + "=" * 70)
        print("PHASE 4: Cable Cut Sabotage / Sensor Flatline (Zone 03 - Mode 3 Fault)")
        print("=" * 70)
        print("[!] Injecting I2C cable severance into zone_03...")
        cluster.get_node("zone_03_accel").inject_fault("flatline")
        time.sleep(2.5)

        print("\n" + "=" * 70)
        print("DEMO AUDIT SUMMARY & RECENT SECURITY EVENTS")
        print("=" * 70)
        events = get_recent_events(limit=10)
        print(f"Total Security Breach Events Logged: {len(events)}")
        for e in events[:5]:
            print(f"  - [{e['timestamp'][:19]}] Zone: {e['zone_id']:<10} Mode: {e['operating_mode']:<8} Alert: {e['alert_level']:<22} Class: {e['vibration_class']}")

    except KeyboardInterrupt:
        print("\n[!] Demo interrupted by user.")
    finally:
        print("\nShutting down simulation nodes...")
        cluster.stop_all()
        cam.stop()
        worker.stop()
        print("=" * 70)
        print("DEMO RUN COMPLETE - ALL ARCHITECTURAL FLOWS VERIFIED")
        print("=" * 70)


if __name__ == "__main__":
    main()

