# AI-Based Perimeter Fence Intrusion Detection System (PIDS)
### Multi-Modal Edge AI & Fault-Tolerant Distributed Architecture

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![MQTT](https://img.shields.io/badge/MQTT-Mosquitto-orange.svg)](https://mosquitto.org/)
[![Redis](https://img.shields.io/badge/Redis-Cache%20%26%20TTL-red.svg)](https://redis.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 1. System Overview & Context

This repository houses the end-to-end software stack, Hardware-in-the-Loop (HIL) simulators, digital signal processing (DSP) pipelines, edge machine learning classifiers, telemetry backend, and visualization dashboard for an **AI-Based Perimeter Fence Intrusion Detection System (PIDS)**.

The system addresses critical challenges in physical border and high-stakes perimeter security (e.g., border outposts, military installations, and Line-of-Control style deployments):

- **Wind vs. Intrusion Ambiguity**: Eliminates false alarms caused by environmental vibrations (wind buffeting, heavy rain, foliage, animal grazing) using edge-computed DSP features and supervised classification.
- **Dual-Stage Multi-Modal Verification**: Employs continuous ultra-low-power MEMS 3-axis vibration sensing (MPU6050) as an event trigger to wake adjacent optical verification nodes (Raspberry Pi Camera / YOLO) on demand.
- **Strict "No Silent Failures" Architecture**: Enforces heartbeat TTLs, MQTT Last Will and Testament (LWT) death certificates, and automated fail-safe state degradation whenever sensors, nodes, or network layers drop.
- **Hardware-Free Simulation Environment**: Full end-to-end testing, validation, and paper-grade benchmark logging without requiring physical hardware present.

---

## 2. Architecture & Data Flow

```text
                      +─────────────────────────────────────────────+
                      |         HARDWARE-IN-THE-LOOP SIMULATORS     |
                      |                                             |
                      |  [Mock ESP32 Node]    [Mock Camera Node]    |
                      |   - 100 Hz MPU6050      - Triggered Wakeup  |
                      |   - Synthetic Signals   - YOLOv8 Inference  |
                      |   - Heartbeat + LWT     - Image Bounding Box|
                      +──────────────┬──────────────────┬───────────+
                                     │                  │
                                     ▼                  ▼
+───────────────────────────────────────────────────────────────────────────────────+
|                                 MESSAGING BROKER                                  |
|                            [Eclipse Mosquitto (MQTT)]                             |
|                                                                                   |
|  Topics:                                                                          |
|    - `fence/node/{id}/telemetry`      (100 Hz windowed acceleration arrays)       |
|    - `fence/node/{id}/heartbeat`      (1 Hz node health, uptime, battery)         |
|    - `fence/node/{id}/status`         (LWT retained dead broker notification)     |
|    - `fence/system/trigger_cam`       (Vibration alert -> Camera trigger)         |
|    - `fence/node/{id}/camera_alert`   (Optical verification bounding box payload) |
+─────────────────────────────────────────┬─────────────────────────────────────────+
                                          │
                                          ▼
+───────────────────────────────────────────────────────────────────────────────────+
|                        CONCURRENT TELEMETRY INGESTION ENGINE                      |
|                                [FastAPI + AsyncIO]                                |
|                                                                                   |
|  - Asynchronous MQTT worker subscribing via `paho-mqtt` / `aiomqtt`               |
|  - Real-time sliding window DSP & ML inference engine                             |
|  - Degraded-mode multi-modal fusion state machine                                 |
+──────────────────────────┬─────────────────────────────┬──────────────────────────+
                           │                             │
                           ▼                             ▼
+──────────────────────────────────────+     +──────────────────────────────────────+
|          REDIS CACHE & PUB/SUB       |     |         POSTGRESQL STORAGE           |
|                                      |     |                                      |
|  - Key TTL: `node:{id}:alive` (3s)   |     |  - `nodes`: Topology & calibration   |
|  - Missing key -> Immediate FAULT    |     |  - `telemetry_logs`: Raw batch data  |
|  - Zone active status broadcaster    |     |  - `events`: Fused security breaches |
|  - WebSocket push pipe to UI         |     |  - `system_faults`: System downtime  |
+──────────────────────────┬───────────+     +──────────────────────────────────────+
                           │
                           ▼
+───────────────────────────────────────────────────────────────────────────────────+
|                               OPERATOR MONITORING UI                              |
|                            [Streamlit / React Dashboard]                          |
|                                                                                   |
|  - Live Perimeter Map (Green: Normal | Yellow: Wind | Orange: Fault | Red: Alert) |
|  - Streaming 3-Axis Accelerometer Chart (a_x, a_y, a_z)                           |
|  - Triggered Camera Snapshot with Detection Bounding Boxes                        |
|  - Interactive Fault-Injection Panel (Simulate Cable Cut, Storm, Dropped Broker)  |
+───────────────────────────────────────────────────────────────────────────────────+
```

---

## 3. Sensor Synthesis & DSP Mathematics

The virtual sensor generates synthetic raw $a_x(t), a_y(t), a_z(t)$ series sampled at $f_s = 100\text{ Hz}$ across a sliding window of $N = 200$ samples ($T = 2\text{ s}$ window, 50% overlap).

### 3.1 Signal Models
- **Ambient Noise**:
  $$a_{\text{baseline}}(t) = \mathcal{N}(0, \sigma^2), \quad \sigma = 0.02g, \quad a_z(t) = 1.0g + \mathcal{N}(0, \sigma^2)$$
- **Wind Buffeting & Rain**:
  Low-frequency harmonic oscillations ($1 \le f_w \le 5\text{ Hz}$) with stochastic envelope modulation:
  $$a_{\text{wind}}(t) = (1 + \eta(t)) \sum_{i} A_i \sin(2\pi f_i t + \phi_i) + \mathcal{N}(0, \sigma_w^2)$$
- **Climbing / Sabotage**:
  High-frequency mechanical shock oscillations ($20 \le f_c \le 35\text{ Hz}$) superimposed on sustained downward mass loading:
  $$a_{\text{climb}}(t) = A_c \cdot e^{-\lambda t} \sin(2\pi f_c t) + \Delta g_z \quad (\text{mean } a_z \approx 2.0g)$$
- **Cutting**:
  High-energy impulse shocks ($f_k > 40\text{ Hz}$, damping factor $\alpha = 30$) with extreme Fisher kurtosis ($\kappa_{\text{Fisher}} > 6.0$).

---

## 4. Multi-Modal Fusion & Fail-Safe State Engine

| Mode | System Condition | Operational Logic | Dashboard Alert State |
| :--- | :--- | :--- | :--- |
| **Mode 1 (Full Fusion)** | Accel Node OK, Camera Node OK | Vibration breach triggers camera verification within $3\text{ s}$. Bounding box person detection confirmed. | `CRITICAL_INTRUSION` |
| **Mode 2 (Degraded: Vibration Only)** | Accel Node OK, Camera Node OFFLINE / Blind | Accelerometer classification exceeds threshold. Camera verification unavailable. | `UNVERIFIED_INTRUSION` (Escalate to patrol) |
| **Mode 3 (Degraded: Sensor Fault)** | Heartbeat missed ($>3\text{ s}$) or flatline I2C data ($a_x=a_y=a_z=0$) | Node marked offline via Redis TTL expiry or MQTT LWT death packet. | `SENSOR_FAULT / PERIMETER COMPROMISED` (Never "All-Clear") |

---

## 5. Physical Hardware Deployment & Replacement Guide

When transitioning from Hardware-in-the-Loop simulation to physical hardware deployment:

```text
               PHYSICAL SENSOR NODE WIRING (ESP32 -> MPU6050)
               +───────────────────+          +───────────────────+
               |     ESP32 Node    |          |   MPU6050 (I2C)   |
               |                   |          |                   |
               |              3V3  +──────────+  VCC              |
               |              GND  +──────────+  GND              |
               |      GPIO 21 (SDA)+──────────+  SDA              |
               |      GPIO 22 (SCL)+──────────+  SCL              |
               |      GPIO 19 (INT)+──────────+  INT              |
               +───────────────────+          +───────────────────+

               OPTICAL VERIFICATION NODE WIRING (Raspberry Pi Zero 2 W)
               +───────────────────+          +───────────────────+
               | Raspberry Pi Zero |          | Pi Camera Mod 3   |
               |                   |          |                   |
               |         CSI Port  +══════════+  15-pin Ribbon    |
               |      Wi-Fi / PoE  +          |  Sony IMX708      |
               +───────────────────+          +───────────────────+
```

### 5.1 ESP32 Sensor Node Setup
1. Mount the MPU6050 sensor rigidly to the fence mesh inside a weather-proof IP67 enclosure using an aluminum mounting plate.
2. Connect I2C lines: ESP32 `GPIO 21` (SDA) and `GPIO 22` (SCL), powered via 3.3V.
3. Configure WiFi/LoRa credentials in `firmware/config.h` and point MQTT host to the edge broker IP.
4. Firmware polls MPU6050 FIFO buffer at 100 Hz, windowing into 200-sample JSON payloads published over MQTT.

### 5.2 Raspberry Pi Zero Optical Node Setup
1. Mount the camera node on fence support posts angled at a $45^\circ$ downward sweep along the perimeter wire run.
2. Install dependencies: `pip install paho-mqtt pillow ultralytics`
3. Run the camera listener: `python simulator/mock_camera.py` (point broker to edge server).
4. The node remains in ultra-low-power standby until waking upon receiving a `fence/system/trigger_cam` message from the vibration node.

---

## 6. Development Roadmap & Milestones

- [x] **Day 1:** Telemetry Schema & Vibration Synthesizer Engine
- [x] **Day 2:** DSP Pipeline & Statistical Feature Extraction ($< 0.5\,\text{ms}$ latency)
- [x] **Day 3:** Baseline Edge Classifier ($100\%$ validation accuracy, $4.4\,\text{ms}$ inference)
- [x] **Day 4:** Dockerized Infrastructure Setup & Persistence Layer
- [x] **Day 5:** Virtual ESP32 Node Simulator with Dynamic Fault Injection
- [x] **Day 6:** Asynchronous Ingestion Backend & Multi-Modal State Machine
- [x] **Day 7:** Checkpoint 1 & Stress Test ($5.79\,\text{ms}$ end-to-end latency)
- [x] **Day 8:** Mock Visual Verification Node & Synthetic Surveillance Dataset
- [x] **Day 9:** Multi-Modal Fusion Engine with Redis Pub/Sub Broadcasting
- [x] **Day 10:** Operator Monitoring Dashboard (Streamlit UI with Live Waveforms & Map)
- [x] **Day 11:** End-to-End Fault Injection & Scenario Verification (Scenarios A, B, C, D)
- [x] **Day 12:** Automated 1,000-Run Paper Benchmark Harness
- [x] **Day 13:** Results Visualization, Latency Breakdown Charts & LaTeX Table Exporter
- [x] **Day 14:** Final Integration, Code Freeze & Deliverables

---

## 7. Experimental Benchmark Results (1,000 Runs)

| Evaluation Metric | Experimental Result | Target Requirement | Compliance |
| :--- | :---: | :---: | :---: |
| **False Alarm Rate (FAR)** | **`0.00%`** | $< 1.0\%$ | **PASSED** |
| **Intrusion Detection Recall** | **`100.00%`** | $> 98.0\%$ | **PASSED** |
| **Mean End-to-End Latency** | **`52.81 ms`** | $< 300.0\text{ ms}$ | **PASSED** |
| **95th Percentile Latency** | **`56.13 ms`** | $< 350.0\text{ ms}$ | **PASSED** |
| **Fault Detection Latency** | **`0.0392 ms`** | $< 5.0\text{ ms}$ | **PASSED** |

Generated publication artifacts:
- [`benchmark/fig_confusion_matrix.png`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/benchmark/fig_confusion_matrix.png): Multi-class confusion matrix.
- [`benchmark/fig_roc_pr_curves.png`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/benchmark/fig_roc_pr_curves.png): ROC & Precision-Recall comparison curves.
- [`benchmark/fig_latency_breakdown.png`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/benchmark/fig_latency_breakdown.png): Stage-by-stage latency breakdown.
- [`benchmark/table_benchmark_summary.tex`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/benchmark/table_benchmark_summary.tex): LaTeX summary table.

---

## 8. Quick Start & Execution

### 8.1 Run the Full Test Suite
```bash
python -m pytest tests/ -v
```

### 8.2 Run End-to-End Multi-Modal Simulation Demo
```bash
python run_demo.py
```

### 8.3 Launch Operator Monitoring Dashboard
```bash
streamlit run dashboard/app.py
```

### 8.4 Run Research Benchmark Harness & Generate Plots
```bash
python benchmark/run_evaluation.py
python benchmark/generate_plots.py
```
