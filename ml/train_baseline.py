"""
Synthetic dataset generator and Random Forest edge classifier training pipeline.

Generates 10,000 labeled signal windows across 4 classes:
- NORMAL
- WIND_RAIN
- CLIMBING
- CUTTING

Extracts DSP feature vectors, trains a high-accuracy scikit-learn RandomForestClassifier,
evaluates precision, recall, and F1-score, benchmarks CPU inference latency (< 5 ms target),
and exports the trained artifact to ml/model.joblib.
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys
from pathlib import Path
import time
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from simulator.generator import VibrationSynthesizer, SignalType
from edge.dsp import extract_features, FEATURE_NAMES

MODEL_PATH = _project_root / "ml" / "model.joblib"
METRICS_PATH = _project_root / "ml" / "metrics.json"

CLASSES = [SignalType.NORMAL.value, SignalType.WIND_RAIN.value, SignalType.CLIMBING.value, SignalType.CUTTING.value]


def generate_dataset(samples_per_class: int = 2500, fs: float = 100.0, seed: int = 42):
    """
    Synthesizes labeled acceleration windows and extracts DSP features.
    Total samples = 4 * samples_per_class (default 10,000).
    """
    synth = VibrationSynthesizer(fs=fs, seed=seed)
    X = []
    y = []

    print(f"[*] Generating {samples_per_class * len(CLASSES)} synthetic windows across {len(CLASSES)} classes...")
    start_time = time.time()

    for label_idx, cls_name in enumerate(CLASSES):
        sig_type = SignalType(cls_name)
        print(f"    - Synthesizing {samples_per_class} windows for class '{cls_name}'...")
        for _ in range(samples_per_class):
            window = synth.generate_sample(sig_type, num_samples=200)
            feat_vec = extract_features(window, fs=fs)
            X.append(feat_vec)
            y.append(label_idx)

    elapsed = time.time() - start_time
    print(f"[+] Feature extraction completed in {elapsed:.2f}s.")
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int64)


def benchmark_inference_latency(model, sample_feature: np.ndarray, num_runs: int = 1000):
    """Measures single-window model inference latency on CPU."""
    sample_batch = sample_feature.reshape(1, -1)
    # Warmup
    for _ in range(50):
        _ = model.predict(sample_batch)

    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = model.predict(sample_batch)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    latencies = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "target_met": bool(np.mean(latencies) < 5.0),
    }


def train_and_export(samples_per_class: int = 2500, test_size: float = 0.20, random_state: int = 42):
    """Executes end-to-end dataset generation, training, validation, and serialization."""
    X, y = generate_dataset(samples_per_class=samples_per_class, seed=random_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[*] Training RandomForestClassifier (30 edge-optimized estimators) on {len(X_train)} instances...")
    clf = RandomForestClassifier(
        n_estimators=30,
        max_depth=10,
        min_samples_split=4,
        random_state=random_state,
        n_jobs=1,
    )
    train_start = time.time()
    clf.fit(X_train, y_train)
    train_dur = time.time() - train_start
    print(f"[+] Model training completed in {train_dur:.2f}s.")

    # Validation
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    report_dict = classification_report(y_test, y_pred, target_names=CLASSES, output_dict=True)
    report_text = classification_report(y_test, y_pred, target_names=CLASSES, digits=4)
    cm = confusion_matrix(y_test, y_pred).tolist()

    print("\n" + "=" * 65)
    print("CLASSIFICATION REPORT (Validation Set):")
    print("=" * 65)
    print(report_text)

    # Benchmark CPU inference latency
    latency_bench = benchmark_inference_latency(clf, X_test[0], num_runs=1000)
    print("=" * 65)
    print("CPU INFERENCE LATENCY BENCHMARK:")
    print("=" * 65)
    print(f"Mean Latency per Window   : {latency_bench['mean_ms']:.4f} ms")
    print(f"Median Latency            : {latency_bench['median_ms']:.4f} ms")
    print(f"95th Percentile           : {latency_bench['p95_ms']:.4f} ms")
    print(f"Target (< 5.0 ms)         : {'PASSED' if latency_bench['target_met'] else 'FAILED'}")
    print("=" * 65)

    # Export model artifact & metadata
    package = {
        "model": clf,
        "classes": CLASSES,
        "feature_names": FEATURE_NAMES,
        "trained_timestamp": time.time(),
        "n_samples": len(X),
        "val_accuracy": float(report_dict["accuracy"]),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(package, MODEL_PATH)
    print(f"[+] Exported model artifact to: {MODEL_PATH}")

    # Export metrics json
    metrics = {
        "classes": CLASSES,
        "accuracy": float(report_dict["accuracy"]),
        "classification_report": report_dict,
        "confusion_matrix": cm,
        "latency_benchmark": latency_bench,
        "num_features": len(FEATURE_NAMES),
        "feature_importances": {name: float(imp) for name, imp in zip(FEATURE_NAMES, clf.feature_importances_)},
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[+] Exported metrics report to: {METRICS_PATH}")

    return package, metrics


if __name__ == "__main__":
    train_and_export(samples_per_class=2500)
