"""Known-miscalibration smoke check, not an aircraft or optimization benchmark."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ra_lsbo.calibration import SigmoidCalibrator
from ra_lsbo.metrics import binary_metrics


def dataset(seed, count):
    random = np.random.RandomState(seed)
    score = random.uniform(-2, 2, count)
    true_probability = 1/(1+np.exp(-(.6*score-.4)))
    raw_probability = 1/(1+np.exp(-(2.5*score+1)))
    labels = (random.uniform(size=count) < true_probability).astype(int)
    return raw_probability.tolist(), labels.tolist(), true_probability.tolist()


def run(output):
    calibration_p, calibration_y, _ = dataset(17, 1000)
    test_p, test_y, true_p = dataset(29, 2000)
    calibrator = SigmoidCalibrator(calibration_p, calibration_y)
    calibrated = calibrator.predict(test_p)
    if calibrated is None:
        raise RuntimeError("Synthetic fixture unexpectedly failed calibration gates")
    raw_metrics, calibrated_metrics = binary_metrics(test_p, test_y), binary_metrics(calibrated, test_y)
    passed = (calibrated_metrics["brier"] < raw_metrics["brier"]
              and calibrated_metrics["log_loss"] < raw_metrics["log_loss"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = Path(output).resolve() / ("calibration_synthetic_" + stamp + "_" + uuid.uuid4().hex[:8])
    destination.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "passed" if passed else "failed",
        "mode": "known_miscalibration_synthetic_smoke_check_not_optimization",
        "generator": {
            "score": "Uniform(-2,2)",
            "true_probability": "sigmoid(0.6*score-0.4)",
            "raw_probability": "sigmoid(2.5*score+1)",
            "calibration_seed": 17, "calibration_samples": 1000,
            "test_seed": 29, "test_samples": 2000,
        },
        "calibration": calibrator.diagnostics,
        "test_raw": raw_metrics, "test_calibrated": calibrated_metrics,
        "python_executable": sys.executable,
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("numpy", "scikit-learn", "matplotlib")},
        "source_sha256": {
            str(path.relative_to(Path(__file__).parent)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).parent / "ra_lsbo" / "calibration.py",
                         Path(__file__).parent / "ra_lsbo" / "metrics.py")},
        "limitations": [
            "Deliberately miscalibrated synthetic scalar scores, no VAE, GP, optimizer or MDA.",
            "Fixed method and thresholds; test labels used only for final scoring.",
            "A successful smoke check is not aircraft calibration evidence or a confidence bound.",
        ],
    }
    (destination / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    with (destination / "test_predictions.jsonl").open("x", encoding="utf-8") as stream:
        for raw, predicted, truth, label in zip(test_p, calibrated, true_p, test_y):
            stream.write(json.dumps({
                "raw": raw, "calibrated": predicted, "true_probability": truth, "label": label}) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].plot([0, 1], [0, 1], color="#777777", linestyle="--", label="Ideal")
    for metrics, color, label in ((raw_metrics, "#b54d3f", "Raw"),
                                  (calibrated_metrics, "#217b76", "Calibrated")):
        bins = [group for group in metrics["reliability"] if group["samples"]]
        axes[0].plot([group["mean_probability"] for group in bins],
                     [group["observed_fraction"] for group in bins], "o-", color=color, label=label)
    axes[0].set(xlabel="Mean predicted probability", ylabel="Observed positive fraction",
                title="Independent synthetic test: reliability", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend()
    edges = np.linspace(0, 1, 6)
    axes[1].hist([test_p, calibrated], bins=edges, color=["#b54d3f", "#217b76"], label=["Raw", "Calibrated"])
    axes[1].set(xlabel="Predicted probability", ylabel="Test samples",
                title="Prediction counts (2,000 samples)", xlim=(0, 1))
    axes[1].legend()
    fig.savefig(destination / "reliability.png", dpi=160)
    plt.close(fig)
    print("Synthetic calibration report:", destination / "report.json")
    print(f"Brier: {raw_metrics['brier']:.6f} -> {calibrated_metrics['brier']:.6f}")
    print(f"Log loss: {raw_metrics['log_loss']:.6f} -> {calibrated_metrics['log_loss']:.6f}")
    if not passed:
        raise RuntimeError("Synthetic smoke check did not improve both predeclared metrics")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "runs")
    run(parser.parse_args().output)
