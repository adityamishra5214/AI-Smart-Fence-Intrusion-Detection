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
|  - Asynchronous MQTT worker subscribing via `aiomqtt`                             |
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
|  - WebSocket push pipe to UI         |     |  - `fault_logs`: System downtime     |
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
  Low-frequency harmonic oscillations ($1 \le f_w \le 5\text{ Hz}$) with stochastic amplitude modulation modeling fence mesh flutter:
  $$a_{\text{wind}}(t) = A(t) \cdot \sin(2\pi f_w t + \phi) + \mathcal{N}(0, \sigma_w^2)$$

- **Climbing / Sabotage**:
  High-frequency mechanical shock oscillations ($15 \le f_c \le 35\text{ Hz}$) superimposed on sustained mass loading along the vertical axis ($a_z$):
  $$a_{\text{climb}}(t) = A_c \cdot e^{-\lambda t} \sin(2\pi f_c t) + \Delta g_z$$

- **Cutting**:
  High-energy impulse shocks ($f_k > 40\text{ Hz}$) with extreme kurtosis spikes ($\kappa > 5.0$).

### 3.2 Extracted Feature Vector ($\mathbf{x} \in \mathbb{R}^{14}$)

1. **Root Mean Square (RMS)**:
   $$\text{RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^N x_i^2}$$
2. **Kurtosis ($\kappa$)**:
   $$\kappa = \frac{\frac{1}{N}\sum_{i=1}^N (x_i - \bar{x})^4}{\left(\frac{1}{N}\sum_{i=1}^N (x_i - \bar{x})^2\right)^2}$$
3. **Zero-Crossing Rate (ZCR)**: Number of sign changes per window.
4. **Band-Power Ratios**: FFT energy split across:
   - **Band 1** ($0\text{--}5\text{ Hz}$): Wind dominance.
   - **Band 2** ($5\text{--}15\text{ Hz}$): Animal disturbances.
   - **Band 3** ($15\text{--}50\text{ Hz}$): Human structural contact.
5. **Spectral Centroid**:
   $$f_c = \frac{\sum f \cdot |X(f)|}{\sum |X(f)|}$$

---

## 4. Multi-Modal Fusion & Fail-Safe State Engine

The backend strictly prevents silent failures by maintaining three distinct operating states:

| Mode | System Condition | Operational Logic | Dashboard Alert State |
| :--- | :--- | :--- | :--- |
| **Mode 1 (Full Fusion)** | Accel Node OK, Camera Node OK | Vibration breach triggers camera verification within $3\text{ s}$. Bounding box person detection confirmed. | `CRITICAL_INTRUSION` |
| **Mode 2 (Degraded: Vibration Only)** | Accel Node OK, Camera Node OFFLINE / Blind | Accelerometer classification exceeds threshold. Camera verification unavailable. | `UNVERIFIED_INTRUSION` (Escalate to patrol) |
| **Mode 3 (Degraded: Sensor Fault)** | Heartbeat missed ($>3\text{ s}$) or flatline I2C data ($a_x=a_y=a_z=0$) | Node marked offline via Redis TTL expiry or MQTT LWT death packet. | `SENSOR_FAULT / PERIMETER COMPROMISED` (Never "All-Clear") |

---

## 5. Repository Directory Layout

```text
ai-smart-fence/
├── docker-compose.yml              # Mosquitto, PostgreSQL, Redis stack
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable configurations
│
├── simulator/
│   ├── __init__.py
│   ├── generator.py                # Mathematical MPU6050 vibration synthesizer
│   ├── mock_esp32.py               # Multithreaded virtual ESP32 node publisher
│   └── mock_camera.py              # Event-triggered mock Pi Zero YOLO verification
│
├── edge/
│   ├── __init__.py
│   ├── dsp.py                      # Windowing, FFT, and statistical feature extraction
│   └── filter.py                   # Digital bandpass & exceedance filtering
│
├── ml/
│   ├── __init__.py
│   ├── train_baseline.py           # Synthetic dataset generator & Random Forest trainer
│   ├── model.joblib                # Exported edge classifier artifact
│   └── evaluate.py                 # Confusion matrix and ROC metrics exporter
│
├── backend/
│   ├── __init__.py
│   ├── main.py                     # FastAPI core & startup hooks
│   ├── mqtt_client.py              # Asynchronous aiomqtt ingestion worker
│   ├── redis_state.py              # Node TTL monitoring & heartbeat watcher
│   ├── fusion.py                   # Multi-modal decision logic & state machine
│   ├── db.py                       # SQLAlchemy / connection pool
│   ├── models.py                   # Database table definitions
│   └── schema.sql                  # Raw SQL schema definition
│
├── dashboard/
│   ├── __init__.py
│   ├── app.py                      # Streamlit real-time monitoring console
│   └── components/                 # Plots, zone map visualizer, and fault triggers
│       └── __init__.py
│
├── tests/
│   ├── __init__.py
│   ├── test_scenarios.py           # Integration tests for Modes 1, 2, and 3
│   └── test_dsp.py                 # Mathematical validation of feature extractors
│
└── benchmark/
    ├── __init__.py
    ├── run_evaluation.py           # 1,000-run automated test harness
    └── generate_plots.py           # High-resolution LaTeX/PDF paper figures
```

---

## 6. 14-Day Development Roadmap

### Week 1: Core Signal Pipeline, DSP, Mock Nodes & Ingestion
- [ ] **Day 1: Telemetry Schema & Vibration Synthesizer Engine**  
  Define MQTT JSON payload formats (telemetry, heartbeat, alert). Implement `simulator/generator.py` for synthetic 3-axis waveforms (ambient noise, wind, climbing, cutting).
- [ ] **Day 2: DSP Pipeline & Statistical Feature Extraction**  
  Implement sliding window segmentation (2-second windows, 50% overlap). Build time-domain (RMS, Kurtosis, ZCR) and frequency-domain (FFT power bands, Spectral Centroid) feature extractors.
- [ ] **Day 3: Baseline Classifier (Wind vs. Intrusion)**  
  Generate 10,000 synthetic windows across 4 classes (`NORMAL`, `WIND_RAIN`, `CLIMBING`, `CUTTING`). Train and export a `RandomForestClassifier` with sub-5ms CPU inference.
- [ ] **Day 4: Dockerized Infrastructure Setup**  
  Configure `docker-compose.yml` for Mosquitto (1883), PostgreSQL (5432), and Redis (6379). Write `backend/schema.sql` defining node, event, and fault tables.
- [ ] **Day 5: Virtual ESP32 Node Simulator**  
  Implement `simulator/mock_esp32.py` simulating 3 concurrent fence zones (`zone_01`, `zone_02`, `zone_03`). Configure MQTT Last Will and Testament (LWT) and dynamic fault injection hooks (cut cable, zero flatline).
- [ ] **Day 6: Asynchronous Ingestion Backend (FastAPI + Redis)**  
  Implement `backend/main.py` with `aiomqtt` background tasks. Configure Redis heartbeat caching with 3-second TTL for automatic `SENSOR_FAULT` detection.
- [ ] **Day 7: Checkpoint 1 & Stress Testing**  
  Run 2-hour continuous ingestion test with 3 simulated nodes transmitting at 100 Hz. Validate sub-20ms DB ingestion latency and zero message loss.

### Week 2: Vision Node, Fail-Safe Engine, UI & Paper Evaluation
- [ ] **Day 8: Mock Visual Verification Node**  
  Implement `simulator/mock_camera.py` listening to `fence/system/trigger_cam`. Run triggered YOLOv8n object detection on test frame samples and publish bounding boxes.
- [ ] **Day 9: Multi-Modal Sensor Fusion & Fail-Safe State Engine**  
  Implement `backend/fusion.py` evaluating Mode 1 (Full Fusion), Mode 2 (Vibration Only), and Mode 3 (Fault). Broadcast fused state changes via Redis Pub/Sub channels.
- [ ] **Day 10: Operator Dashboard (Streamlit / Web UI)**  
  Build real-time perimeter zone map, live 3-axis vibration graphs, and triggered camera feeds. Add interactive fault injection console (Simulate Cable Cut, Storm, Broker Outage).
- [ ] **Day 11: End-to-End Fault Injection & Scenario Verification**  
  Validate rejection of continuous 4 Hz wind noise without false triggers. Validate sub-3-second transition to `FAULT` on simulated node disconnect.
- [ ] **Day 12: Automated Research Paper Benchmark Harness**  
  Execute 1,000 automated evaluation runs measuring False Alarm Rate (FAR), Recall, Latency, and Fault Uptime.
- [ ] **Day 13: Results Visualization & Metric Export**  
  Generate confusion matrices, ROC/PR curves, and latency breakdown charts ready for LaTeX paper insertion.
- [ ] **Day 14: Final Integration, Code Freeze & Deliverables**  
  Verify single-command startup (`docker compose up` + run scripts) and document hardware replacement steps.

---

## 7. Quick Start & Execution

### 7.1 Prerequisites
- Python 3.11+
- Docker & Docker Compose
- Git

### 7.2 Launch Infrastructure
```bash
# Clone and enter repo
git clone <repo-url>
cd ai-smart-fence

# Launch Mosquitto, PostgreSQL, and Redis
docker compose up -d

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 7.3 Run Simulation Demo
```bash
# Terminal 1: Run Ingestion Backend
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Run Mock ESP32 Vibration Node
python -m simulator.mock_esp32

# Terminal 3: Run Mock Camera Node
python -m simulator.mock_camera

# Terminal 4: Run Operator Dashboard
streamlit run dashboard/app.py
```
