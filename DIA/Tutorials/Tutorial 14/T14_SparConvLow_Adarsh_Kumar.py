#!/usr/bin/env python3
"""
AI531C — T14: Defence Review + Reproducibility Check
Project: SparConvLow — Integrating Convolution and Sparse Coding
Student: Adarsh Kumar
Date: 17 November 2026

Purpose
-------
This T14 script adapts the reproducibility and experiment-recording activity
to the SparConvLow project. It records the model configuration described in
the supplied IEEE TNNLS paper and checks that the same random seed produces
the same reproducibility signature.

IMPORTANT
---------
This file is a T14 reproducibility/experiment-tracking companion. It is NOT
a complete implementation of the SparConvLow research model. It does not
claim to train a CNN, learn the dictionary, solve the elastic-net sparse
coding problem, or optimize U/D/P. Connect your actual project training
function in run_project_experiment() before recording real accuracy/loss.

Paper-grounded configuration used here:
- CNN backbone: ResNet-18 (selected as the project configuration)
- CNN learning rate: 0.01
- CNN momentum: 0.9
- Batch size: 64 (one of the paper's reported settings)
- Reduced feature dimension l: 256 for tiny-image datasets
- Elastic-net lambda2: 1e-7
- Trace-quotient stabilizer sigma: 1e-3
- For LDA/SDA, final dimension d = number_of_classes - 1

The paper studies lambda1 and dictionary size experimentally; they are
left as editable experiment variables. Epoch count is not hard-coded because
the paper does not give one universal epoch count for all experiments.

Usage
-----
    python T14_studio.py
    python T14_studio.py support
    python T14_studio.py extension
    python T14_studio.py --show-row 23
    python T14_studio.py --record-metric 82.35 --metric-name accuracy

Install:
    pip install numpy

The script also works without NumPy, but NumPy is recommended.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
except ImportError:
    torch = None

STUDENT_NAME = "Adarsh Kumar"
SUBMISSION_DATE = "17 November 2026"
PROJECT_TITLE = (
    "Integrating Convolution and Sparse Coding for Learning "
    "Low-Dimensional Discriminative Image Representations (SparConvLow)"
)
PAPER_REFERENCE = "IEEE TNNLS, Vol. 36, No. 7, July 2025"

HERE = Path(__file__).resolve().parent
RUNS_FILE = HERE / "runs.csv"

# Values below are taken from the paper where specified.
# Dataset/backbone and experiment variables should match your implementation.
BASE_CONFIG = {
    "project": "SparConvLow",
    "dataset": "CIFAR-10",
    "backbone": "ResNet-18",
    "dimension_reduction": "orthogonal projection U",
    "reduced_feature_dimension_l": 256,
    "dictionary": "learnable task-driven dictionary D",
    "dictionary_size_k": 10000,       # editable experiment setting
    "sparse_coding": "elastic-net",
    "lambda1": 0.2,                    # paper's best reported CIFAR-10 example
    "lambda2": 1e-7,                   # paper setting
    "final_projection": "V / P on sparse coefficients",
    "final_dimension_rule": "classes - 1 for LDA/SDA",
    "batch_size": 64,                  # paper reports 64 or 256
    "cnn_learning_rate": 0.01,
    "cnn_momentum": 0.9,
    "trace_quotient_sigma": 1e-3,
    "epochs": "set according to actual implementation",
    "seed": 42,
    "metric_name": "top-1 accuracy",
}


def seed_everything(seed):
    """Set random seeds for Python, NumPy and PyTorch when installed."""
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Ask PyTorch for deterministic operations where supported.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, TypeError):
            pass


def stable_signature(config, seed):
    """Create a repeatable fingerprint of the experiment setup."""
    payload = {
        "student": STUDENT_NAME,
        "date": SUBMISSION_DATE,
        "project": PROJECT_TITLE,
        "seed": seed,
        "configuration": config,
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def run_project_experiment(seed, config):
    """
    T14 reproducibility hook.

    Replace the body of this function with the actual SparConvLow training
    and evaluation call from your project. Return a real metric, e.g.:

        from your_model_file import train_and_evaluate
        return train_and_evaluate(config=config, seed=seed)

    Until that integration is done, this function returns only a stable
    configuration signature; it does NOT invent an accuracy or loss value.
    """
    seed_everything(seed)
    return stable_signature(config, seed)


def experiment(seed=42, config=None):
    """Run the reproducibility hook and return its result."""
    cfg = dict(BASE_CONFIG)
    if config:
        cfg.update(config)
    cfg["seed"] = seed
    result = run_project_experiment(seed, cfg)
    return result, cfg


def log_run(mode, seed, config, result, metric_name="", metric_value=""):
    fields = [
        "row_id", "timestamp_local", "student", "submission_date",
        "project", "paper_reference", "mode", "seed", "dataset",
        "backbone", "batch_size", "cnn_learning_rate", "cnn_momentum",
        "reduced_feature_dimension_l", "dictionary_size_k", "lambda1",
        "lambda2", "trace_quotient_sigma", "epochs", "metric_name",
        "metric_value", "result_signature", "full_configuration",
    ]

    existing = []
    if RUNS_FILE.exists():
        with RUNS_FILE.open("r", newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))

    record = {
        "row_id": len(existing) + 1,
        "timestamp_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "student": STUDENT_NAME,
        "submission_date": SUBMISSION_DATE,
        "project": "SparConvLow",
        "paper_reference": PAPER_REFERENCE,
        "mode": mode,
        "seed": seed,
        "dataset": config.get("dataset", ""),
        "backbone": config.get("backbone", ""),
        "batch_size": config.get("batch_size", ""),
        "cnn_learning_rate": config.get("cnn_learning_rate", ""),
        "cnn_momentum": config.get("cnn_momentum", ""),
        "reduced_feature_dimension_l": config.get("reduced_feature_dimension_l", ""),
        "dictionary_size_k": config.get("dictionary_size_k", ""),
        "lambda1": config.get("lambda1", ""),
        "lambda2": config.get("lambda2", ""),
        "trace_quotient_sigma": config.get("trace_quotient_sigma", ""),
        "epochs": config.get("epochs", ""),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "result_signature": result,
        "full_configuration": json.dumps(config, sort_keys=True, default=str),
    }

    with RUNS_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(existing)
        writer.writerow(record)

    return record["row_id"]


def print_project_header():
    print("=" * 72)
    print("AI531C — T14: DEFENCE REVIEW + REPRODUCIBILITY CHECK")
    print(f"Student       : {STUDENT_NAME}")
    print(f"Submission    : {SUBMISSION_DATE}")
    print(f"Project       : SparConvLow")
    print(f"Paper         : {PAPER_REFERENCE}")
    print("=" * 72)


def reproducibility_demo():
    print("\nPART 1 — PROJECT REPRODUCIBILITY CHECK")
    first, config = experiment(42)
    second, _ = experiment(42)
    print(f"Seeded run 1 signature: {first}")
    print(f"Seeded run 2 signature: {second}")
    assert first == second, "Same seed/configuration produced different signatures."
    print("PASS: same seed + same configuration gives the same signature.")

    other, _ = experiment(43)
    print(f"Different-seed signature: {other}")
    print("A different seed changes the recorded experiment identity.")
    print("\nNote: this checks seed/configuration reproducibility only.")
    print("It does not claim that the SparConvLow network has been trained.")


def save_project_runs():
    print("\nPART 2 — SAVING PROJECT RUN RECORDS")
    for seed in (42, 42, 43):
        result, config = experiment(seed)
        row = log_run("reproducibility_check", seed, config, result)
        print(f"Saved runs.csv row {row}: seed={seed}, signature={result}")
    print(f"Run log: {RUNS_FILE}")


def team_check():
    print("\nPART 3 — APPLY THIS CHECK TO YOUR ACTUAL SparConvLow PIPELINE")
    print("Run the complete model twice with the same configuration and seed.")
    print("Compare the actual evaluation metric (accuracy/loss), not just this")
    print("script's configuration signature.")
    print("\nCheck these sources of randomness:")
    for item in [
        "training/validation split and data-loader shuffle",
        "random image augmentation",
        "CNN weight initialization",
        "dictionary D initialization",
        "sparse-code solver initialization or stopping tolerance",
        "dropout and batch-normalization behavior",
        "Python, NumPy and PyTorch random seeds",
        "GPU kernels, CUDA/cuDNN and library versions",
        "parallel data-loader workers",
    ]:
        print(f"- {item}")
    print("\nThe paper's framework jointly learns CNN parameters and U, D, P.")
    print("Keep the same dataset split, model settings, optimizer settings and")
    print("software/hardware environment when comparing repeated runs.")


def extension():
    print("\nPART 4 — EXTENSION: GRID vs OFAT FOR SparConvLow")
    print("Factors: lambda1 (sparsity strength) and dictionary size (K).")
    print("The values below are an editable small experiment plan; they are")
    print("not presented as a complete reproduction of the paper's full sweep.")

    grid = [
        {"lambda1": 0.1, "dictionary_size_k": 5000},
        {"lambda1": 0.1, "dictionary_size_k": 10000},
        {"lambda1": 0.2, "dictionary_size_k": 5000},
        {"lambda1": 0.2, "dictionary_size_k": 10000},
    ]
    # Baseline, vary lambda1 only, then dictionary size only.
    ofat = [
        {"lambda1": 0.1, "dictionary_size_k": 5000},
        {"lambda1": 0.2, "dictionary_size_k": 5000},
        {"lambda1": 0.1, "dictionary_size_k": 10000},
    ]

    for label, configs in (("grid", grid), ("OFAT", ofat)):
        print(f"\n{label.upper()}: {len(configs)} configurations x 2 seeds")
        for factors in configs:
            config = dict(BASE_CONFIG)
            config.update(factors)
            for seed in (42, 43):
                result, run_config = experiment(seed, config)
                row = log_run(f"extension_{label}", seed, run_config, result)
                print(
                    f"row={row:>3} lambda1={factors['lambda1']:.2f} "
                    f"K={factors['dictionary_size_k']} seed={seed}"
                )

    print("\nExpected planned runs: grid = 8; OFAT = 6.")
    print("These are experiment records/signatures until the real training")
    print("function is connected in run_project_experiment().")


def record_metric(metric_value, metric_name):
    """Attach a user-supplied REAL result to a project run record."""
    result, config = experiment(42)
    row = log_run(
        "actual_project_metric_entered", 42, config, result,
        metric_name=metric_name, metric_value=metric_value
    )
    print(f"Recorded supplied {metric_name}={metric_value} in runs.csv row {row}.")
    print("Only enter a metric obtained from your actual model evaluation.")


def show_row(number):
    if not RUNS_FILE.exists():
        print("runs.csv not found. Run the script first.")
        return
    with RUNS_FILE.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    record = next((r for r in rows if int(r["row_id"]) == number), None)
    if record is None:
        print(f"Row {number} not found. Current record count: {len(rows)}")
        return
    print(f"\nRUNS.CSV — ROW {number}")
    for key, value in record.items():
        print(f"{key}: {value}")


def review_and_discussion():
    print("\nPART 5 — DEFENCE REVIEW")
    print("Strength 1: CNN extracts local/high-level image features.")
    print("Strength 2: U reduces feature dimension before sparse coding,")
    print("            reducing the cost of sparse representation learning.")
    print("Question: how will the implementation verify that learned sparse")
    print("          representations improve class discrimination?")

    print("\nDISCUSSION")
    print("Q1. If seeded runs differ on another machine, two legitimate causes are:")
    print("    (a) different PyTorch/CUDA/cuDNN or numerical-library versions;")
    print("    (b) hardware-specific floating-point or nondeterministic GPU operations.")
    print("Q2. runs.csv is the experiment/run log. It protects traceability by")
    print("    storing run ID, seed, dataset, backbone, hyperparameters and result.")
    print("    Match it to the experiment-log/record box in your nine-box diagram.")
    print("Q3. runs.csv stores the exact configuration for each recorded row.")
    print("    Retrieve a row using: python T14_studio.py --show-row 23")
    print("    The record is available immediately after the run is logged.")

    print("\nPROJECT PIPELINE TO DEFEND")
    print("Input image -> CNN feature extractor -> x")
    print("x -> orthogonal projection U^T x -> reduced feature x_hat")
    print("x_hat + dictionary D -> elastic-net sparse coefficients phi")
    print("phi -> discriminative projection V^T phi (or P = V V^T)")
    print("-> low-dimensional representation y -> classifier/evaluation")
    print("The paper trains the CNN, U, D and P jointly using geometric SGD")
    print("and gradient backpropagation.")

    print("\nACTION LIST — COMPLETE WITH YOUR TEAM")
    print("1. Connect actual training/evaluation code | Owner: Adarsh Kumar | Date: 17 Nov 2026")
    print("2. Repeat same run with fixed seed         | Owner: Adarsh Kumar | Date: 17 Nov 2026")
    print("3. Save real metrics and full configs      | Owner: Adarsh Kumar | Date: 17 Nov 2026")


def main():
    parser = argparse.ArgumentParser(
        description="AI531C T14 reproducibility studio for SparConvLow"
    )
    parser.add_argument("mode", nargs="?", choices=["support", "extension"])
    parser.add_argument("--show-row", type=int, default=None)
    parser.add_argument("--record-metric", type=float, default=None)
    parser.add_argument("--metric-name", default="top-1 accuracy")
    args = parser.parse_args()

    print_project_header()

    if args.show_row is not None:
        show_row(args.show_row)
        return

    if args.record_metric is not None:
        record_metric(args.record_metric, args.metric_name)
        return

    if args.mode == "support":
        first, config = experiment(42)
        second, _ = experiment(42)
        print(f"experiment(42), run 1: {first}")
        print(f"experiment(42), run 2: {second}")
        assert first == second
        print("PASS: same seed/configuration gives the same signature.")
        row = log_run("support", 42, config, first)
        print(f"Saved to runs.csv row {row}.")
        print("Category: experiment/run log; it supports reproducibility and traceability.")
        return

    reproducibility_demo()
    save_project_runs()
    team_check()
    if args.mode == "extension":
        extension()
    review_and_discussion()
    print("\nTo retrieve an exact configuration:")
    print("python T14_studio.py --show-row 23")
    print("\nTo record a REAL metric after evaluating your model:")
    print("python T14_studio.py --record-metric 82.35 --metric-name accuracy")


if __name__ == "__main__":
    main()
