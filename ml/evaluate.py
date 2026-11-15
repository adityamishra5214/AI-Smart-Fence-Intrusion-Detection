"""
Model evaluation, confusion matrix visualization, and ROC/PR metric exporter.
Generates publication-quality confusion matrix images and metrics summary.
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

METRICS_PATH = _project_root / "ml" / "metrics.json"
CM_PLOT_PATH = _project_root / "ml" / "confusion_matrix.png"


def plot_confusion_matrix(cm: list, classes: list, output_path: Path):
    """Renders and saves a high-resolution confusion matrix heatmap using matplotlib."""
    cm_arr = np.array(cm)
    cm_norm = cm_arr.astype("float") / cm_arr.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    cax = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1)
    fig.colorbar(cax, label="Normalized Accuracy")

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=10, fontweight="bold")
    ax.set_yticklabels(classes, fontsize=10, fontweight="bold")

    # Annotate cells
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            text = f"{cm_arr[i, j]}\n({cm_norm[i, j] * 100:.1f}%)"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10, fontweight="bold")

    ax.set_title("PIDS Vibration Intrusion Classifier - Confusion Matrix", fontsize=12, fontweight="bold", pad=14)
    ax.set_ylabel("True Class", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[+] Confusion matrix plot saved to: {output_path}")


def evaluate():
    """Reads saved metrics and exports visual artifacts."""
    if not METRICS_PATH.exists():
        print(f"[!] Metrics file {METRICS_PATH} not found. Please run ml/train_baseline.py first.")
        return

    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    cm = metrics["confusion_matrix"]
    classes = metrics["classes"]
    plot_confusion_matrix(cm, classes, CM_PLOT_PATH)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY:")
    print("=" * 60)
    print(f"Overall Accuracy: {metrics['accuracy'] * 100:.2f}%")
    for cls in classes:
        p = metrics["classification_report"][cls]["precision"]
        r = metrics["classification_report"][cls]["recall"]
        f1 = metrics["classification_report"][cls]["f1-score"]
        print(f"  - {cls:<12}: Precision={p:.4f}, Recall={r:.4f}, F1={f1:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()
