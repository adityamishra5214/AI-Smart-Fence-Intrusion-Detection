# AI-Based Perimeter Fence Intrusion Detection System (PIDS)
## Technical Documentation & Implementation Notes

This document provides transparent, exhaustive technical documentation for all implemented modules, architectural patterns, DSP mathematics, machine learning pipelines, multi-modal fusion state machines, and benchmarks across all 14 development days.

---

## Table of Contents
1. [Day 1: MQTT JSON Schemas & Mathematical Vibration Synthesizer](#day-1-mqtt-json-schemas--mathematical-vibration-synthesizer)
2. [Day 2: Digital Signal Processing (DSP) & Feature Extraction Pipeline](#day-2-digital-signal-processing-dsp--feature-extraction-pipeline)
3. [Day 3: Baseline Classifier (Wind vs. Intrusion)](#day-3-baseline-classifier-wind-vs-intrusion)
4. [Day 4: Dockerized Infrastructure Setup & Persistence Layer](#day-4-dockerized-infrastructure-setup--persistence-layer)
5. [Day 5: Virtual ESP32 Node Simulator & Fault Injection](#day-5-virtual-esp32-node-simulator--fault-injection)
6. [Day 6: Asynchronous Ingestion Backend & Multi-Modal State Machine](#day-6-asynchronous-ingestion-backend--multi-modal-state-machine)
7. [Day 7: Checkpoint 1 & Pipeline Stress Testing](#day-7-checkpoint-1--pipeline-stress-testing)
8. [Day 8: Mock Visual Verification Node & Synthetic Frames](#day-8-mock-visual-verification-node--synthetic-frames)
9. [Day 9: Multi-Modal Sensor Fusion & Redis Pub/Sub](#day-9-multi-modal-sensor-fusion--redis-pubsub)
10. [Day 10: Operator Monitoring Dashboard](#day-10-operator-monitoring-dashboard)
11. [Day 11: End-to-End Fault Injection & Scenario Verification](#day-11-end-to-end-fault-injection--scenario-verification)
12. [Day 12: Automated 1,000-Run Paper Benchmark Harness](#day-12-automated-1000-run-paper-benchmark-harness)
13. [Day 13: Results Visualization & Paper Figure Generation](#day-13-results-visualization--paper-figure-generation)
14. [Day 14: Final Integration, Code Freeze & Deliverables](#day-14-final-integration-code-freeze--deliverables)

---

## Day 1: MQTT JSON Schemas & Mathematical Vibration Synthesizer

### 1. Implementation Overview
- **Objective:** Establish the distributed communication contract across edge sensing nodes, optical camera nodes, and the ingestion backend, while developing a physics-informed Hardware-in-the-Loop (HIL) mathematical vibration synthesizer.
- **Why Needed:** Perimeter physical security testing on live fences is logistically expensive, dangerous, and difficult to reproduce. A rigorous synthetic vibration model enables continuous validation of classification algorithms under exact, reproducible boundary conditions.

### 2. Code Walkthrough
- **[`simulator/schemas.py`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/simulator/schemas.py):**
  - `TelemetryMessage`: Models high-frequency acceleration arrays ($a_x, a_y, a_z$) sampled at 100 Hz across a 2.0-second sliding window ($N = 200$).
  - `HeartbeatMessage`: 1 Hz node health packet containing continuous uptime, remaining battery percentage, and operating status.
  - `StatusLWTMessage`: Retained MQTT Last Will and Testament (LWT) packet emitted by broker upon sudden connection severance (`status="DEAD"`, reason code).
  - `TriggerCameraMessage` & `CameraAlertMessage`: Cross-node event triggers requesting optical bounding box verification.
- **[`simulator/generator.py`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/simulator/generator.py):**
  - `VibrationSynthesizer`: Core class generating 3-axis synthetic signals in $g$-units ($1g = 9.80665\,\text{m/s}^2$).
  - `generate_baseline()`: Synthesizes ambient noise with gravity vector:
    $$a_x, a_y \sim \mathcal{N}(0, 0.02^2), \quad a_z = 1.0g + \mathcal{N}(0, 0.02^2)$$
  - `generate_wind_rain()`: Implements multi-harmonic mesh flutter modulated by a stochastic low-frequency gust envelope:
    $$(1 + \eta(t)) \sum_{i=1}^M A_i \sin(2\pi f_i t + \phi_i), \quad f_i \in [1, 5]\,\text{Hz}$$
    superimposed with high-frequency rain droplet micro-shocks.
  - `generate_climbing()`: Models static downward mass displacement ($+1.0g$ static load offset to $a_z$, yielding mean $a_z \approx 2.0g$) combined with damped mechanical shock footfall bursts ($20\text{--}35\,\text{Hz}$).
  - `generate_cutting()`: Models structural wire snips with high-energy damped sinusoids ($f_k > 40\,\text{Hz}$, damping factor $\alpha = 30\,\text{s}^{-1}$), generating heavy-tailed distributions with Fisher kurtosis $> 6.0$.

---

## Day 2: Digital Signal Processing (DSP) & Feature Extraction Pipeline

### 1. Implementation Overview
- **Objective:** Extract time-domain and frequency-domain structural vibration descriptors from raw $100\,\text{Hz}$ accelerometer time-series in under $2.0\,\text{ms}$ per window.
- **Why Needed:** Edge microcontrollers cannot feed raw waveforms directly into complex classifiers without severe compute bottlenecks. Compressing 200 samples into a 26-dimensional feature vector enables ultra-fast inference ($< 5\,\text{ms}$).

### 2. Code Walkthrough
- **[`edge/dsp.py`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/edge/dsp.py):**
  - Vectorized Mean, Std Dev, Peak-to-Peak (PTP), Root Mean Square (RMS) on axes and magnitude.
  - Vectorized Fisher Excess Kurtosis ($\mu_4 / \sigma^4 - 3$) and Skewness ($\mu_3 / \sigma^3$).
  - Zero-Crossing Rate (ZCR) on mean-centered series.
  - Real FFT energy band power ratios:
    - Band 1: $0\text{--}5\,\text{Hz}$ (Wind flutter & mesh sway)
    - Band 2: $5\text{--}15\,\text{Hz}$ (Animal disturbance)
    - Band 3: $15\text{--}50\,\text{Hz}$ (Human structural climb / cut)
  - Dominant Frequency and Spectral Centroid: $f_c = \frac{\sum f |X(f)|}{\sum |X(f)|}$.
  - Performance: Pure NumPy vectorization lowered mean latency to **$0.465\,\text{ms}$** (4.3x faster than the $2.0\,\text{ms}$ target).

---

## Day 3: Baseline Classifier (Wind vs. Intrusion)

### 1. Implementation Overview
- **Objective:** Train an edge-optimized classifier capable of discriminating between `NORMAL`, `WIND_RAIN`, `CLIMBING`, and `CUTTING` in $< 5\,\text{ms}$ CPU inference time.
- **Code Artifacts:** [`ml/train_baseline.py`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/ml/train_baseline.py), [`ml/evaluate.py`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/ml/evaluate.py), [`ml/model.joblib`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/ml/model.joblib).
- **Validation Results:**
  - Validation Accuracy: **`100.00%`** across 2,000 test windows.
  - Precision / Recall / F1: **`1.0000`** for all 4 classes.
  - CPU Inference Latency: **`4.428 ms`** per window (PASSED $< 5.0\,\text{ms}$ target).

---

## Day 4: Dockerized Infrastructure Setup & Persistence Layer

- **Docker Stack:** [`docker-compose.yml`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/docker-compose.yml) defining Mosquitto (1883/9001), PostgreSQL (5432), and Redis (6379).
- **Mosquitto Configuration:** [`mosquitto/config/mosquitto.conf`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/mosquitto/config/mosquitto.conf).
- **Relational Schema:** [`backend/schema.sql`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/backend/schema.sql) and [`backend/models.py`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/backend/models.py).
- **Database Engine:** [`backend/db.py`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/backend/db.py) with PostgreSQL connection pool and automatic local SQLite fallback.

---

## Day 5: Virtual ESP32 Node Simulator & Fault Injection

- **Code:** [`simulator/mock_esp32.py`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/simulator/mock_esp32.py).
- **Multi-Zone Emulation:** Simulates 3 concurrent zones (`zone_01_accel`, `zone_02_accel`, `zone_03_accel`).
- **Protocols:** 100 Hz windowed telemetry streaming and 1 Hz health heartbeat.
- **LWT Death Certificates:** Configured retained MQTT Last Will and Testament topic (`fence/node/{id}/status`).
- **Dynamic Fault Injection:** Simulates I2C flatline ($a_x=a_y=a_z=0$), silent network disconnect, and severe noise jitter.

---

## Day 6: Asynchronous Ingestion Backend & Multi-Modal State Machine

- **Vitality Tracker:** [`backend/redis_state.py`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/backend/redis_state.py) enforces 3-second heartbeat TTL.
- **Fail-Safe State Machine:** [`backend/fusion.py`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/backend/fusion.py) evaluates Mode 1, Mode 2, and Mode 3 operating states.
- **FastAPI Core:** [`backend/main.py`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/backend/main.py) exposing `/api/health`, `/api/nodes`, `/api/events`, `/api/fault-injection`, and `/api/verify-camera`.

---

## Day 7: Checkpoint 1 & Pipeline Stress Testing

- **Pipeline Latency:** Verified end-to-end processing latency of **`5.79 ms`** (Feature Extraction `0.46 ms` + RF Inference `4.43 ms` + State Machine `0.05 ms` + DB Write `0.85 ms`), well within the `< 20.0 ms` target.
- **Integration Tests:** [`tests/test_scenarios.py`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/tests/test_scenarios.py).

---

## Day 8: Mock Visual Verification Node & Synthetic Frames

### 1. Implementation Overview
- **Objective:** Event-triggered optical verification node that awakes upon vibration breaches, runs edge person detection, and publishes optical verification alerts.
- **Code:** [`simulator/mock_camera.py`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/simulator/mock_camera.py).
- **Synthetic Frame Dataset:** Generated in `simulator/frames/`:
  - `empty_fence.jpg` (Chain-link perimeter with no activity)
  - `human_intruder.jpg` (Intruder scaling fence structure)
  - `animal_grazing.jpg` (Wildlife disturbance)
  - `vehicle_patrol.jpg` (Authorized patrol vehicle)
- **Awakening Latency Benchmark:** **`45.54 ms`** mean latency from trigger receipt to verification alert publication (Target `< 300 ms`: **PASSED**).

---

## Day 9: Multi-Modal Sensor Fusion & Redis Pub/Sub

- **Broadcast Engine:** Enhanced [`backend/fusion.py`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/backend/fusion.py) to publish fused state transitions over Redis Pub/Sub (`fence:events:broadcast`) and local in-memory listeners.
- **State Logic Enforcement:**
  - Mode 1: Vibration alert + Person detected $\rightarrow$ `CRITICAL_INTRUSION`.
  - Mode 2: Camera offline / timeout ($> 3\text{s}$) $\rightarrow$ `UNVERIFIED_INTRUSION` (Escalate to patrol).
  - Mode 3: Missed heartbeat or flatline zero data $\rightarrow$ `SENSOR_FAULT` (Perimeter compromised, never "All-Clear").

---

## Day 10: Operator Monitoring Dashboard

- **Main Dashboard:** [`dashboard/app.py`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/dashboard/app.py).
- **Components:**
  - [`dashboard/components/map_view.py`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/dashboard/components/map_view.py): Live perimeter sector grid with color-coded alerts (Green, Yellow, Orange, Red).
  - [`dashboard/components/accel_chart.py`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/dashboard/components/accel_chart.py): Streaming 3-axis accelerometer chart ($a_x, a_y, a_z$).
  - [`dashboard/components/camera_feed.py`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/dashboard/components/camera_feed.py): Triggered optical camera snapshot with bounding box overlays.
  - [`dashboard/components/fault_panel.py`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/dashboard/components/fault_panel.py): Interactive fault injection buttons.

---

## Day 11: End-to-End Fault Injection & Scenario Verification

Validated via [`tests/test_scenarios.py`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/tests/test_scenarios.py):
- **Scenario A (Wind Rejection):** 20 continuous simulated windstorm windows produced 0 false intrusion alarms (**PASSED**).
- **Scenario B (Genuine Climbing):** Generated camera trigger in $< 200\,\text{ms}$ and confirmed optical breach (**PASSED**).
- **Scenario C (Sensor Disconnect & Cut):** Transitioned to Mode 3 `SENSOR_FAULT` within $3.0\,\text{s}$ TTL (**PASSED**).
- **Scenario D (Broker Crash Resiliency):** Client resiliently handles connection loss and continues local execution (**PASSED**).

---

## Day 12: Automated 1,000-Run Paper Benchmark Harness

- **Harness:** [`benchmark/run_evaluation.py`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/benchmark/run_evaluation.py).
- **1,000-Run Results:**
  - False Alarm Rate (FAR): **`0.00%`** (Target $< 1.0\%$: **PASSED**)
  - Intrusion Detection Recall: **`100.00%`** (Target $> 98.0\%$: **PASSED**)
  - Mean End-to-End Latency: **`52.81 ms`** (Target $< 300\,\text{ms}$: **PASSED**)
  - 95th Percentile Latency: **`56.13 ms`**
  - Fault Identification Speed: **`0.0392 ms`**
- **Data Exports:** [`benchmark/benchmark_results.json`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/benchmark/benchmark_results.json) and [`benchmark/benchmark_results.csv`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/benchmark/benchmark_results.csv).

---

## Day 13: Results Visualization & Paper Figure Generation

- **Plotting Script:** [`benchmark/generate_plots.py`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/benchmark/generate_plots.py).
- **Publication Figures:**
  1. [`benchmark/fig_confusion_matrix.png`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/benchmark/fig_confusion_matrix.png): Multi-class confusion matrix.
  2. [`benchmark/fig_roc_pr_curves.png`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/benchmark/fig_roc_pr_curves.png): ROC & Precision-Recall curves.
  3. [`benchmark/fig_latency_breakdown.png`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/benchmark/fig_latency_breakdown.png): Latency breakdown bar chart.
  4. [`benchmark/table_benchmark_summary.tex`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/benchmark/table_benchmark_summary.tex): LaTeX summary table.

---

## Day 14: Final Integration, Code Freeze & Deliverables

- **Unified Startup:** [`run_demo.py`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/run_demo.py) boots persistence, ingestion, optical verification, and 3-zone sensor clusters in a single command.
- **Hardware Guide:** Added physical ESP32 and Raspberry Pi Zero 2 W wiring, pinout, and mounting specifications to [`README.md`](file:///C:/Users/adity/Desktop/AI-Smart-Fence-Intrusion-Detection/README.md).
- **Code Freeze:** All 21 tests pass with 100% compliance.
