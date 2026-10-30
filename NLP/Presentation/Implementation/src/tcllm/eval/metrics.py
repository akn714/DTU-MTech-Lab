"""Evaluation: accuracy, macro-F1, per-class scores, confusion matrix and timing.

The paper reports accuracy only, having found macro-F1 always agreed with it in
ordering. We record both (project rule 10) plus training and inference time, since
the paper's own discussion of SFT leans heavily on cost.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Sequence

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from ..data import Example
from ..methods.base import Method

# Paper-reported English (SENTIMENT) accuracies, Table 6 of arXiv:2502.11830v1.
# Used as reference lines in the figures. These are the AUTHORS' numbers and are
# never merged into our result tables.
PAPER_ENGLISH_SENTIMENT = {
    "GPT-4 (zero-shot)": 0.71,
    "Qwen2.5-7B (zero-shot)": 0.68,
    "Aya-Expanse-8B (zero-shot)": 0.71,
    "FastFit (few-shot, 10/label)": 0.59,
    "Logistic Reg. on GTE (real)": 0.65,
    "mBERT fine-tuned (real)": 0.65,
    "Qwen2.5-7B SFT (real)": 0.58,
    "Logistic Reg. on GTE (synthetic)": 0.59,
    "mBERT fine-tuned (synthetic)": 0.52,
    "Qwen2.5-7B SFT (synthetic)": 0.51,
}


@dataclass
class Result:
    """One evaluated method on one test split."""

    method: str
    family: str
    train_size: int
    test_size: int
    accuracy: float
    macro_f1: float
    fit_seconds: float
    predict_seconds: float
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: list[list[int]] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    notes: str = ""
    # Row-by-row predictions, kept in memory so the caller can persist them. They
    # are stripped from ``to_dict`` to keep results.json small - a 870-row array
    # per method would dominate the file, and predictions.jsonl already holds them.
    predictions: list[int] = field(default_factory=list, repr=False)

    @property
    def examples_per_second(self) -> float:
        return self.test_size / self.predict_seconds if self.predict_seconds > 0 else float("inf")

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("predictions", None)
        data["examples_per_second"] = round(self.examples_per_second, 1)
        return data


def evaluate(
    method: Method,
    train: Sequence[Example],
    test: Sequence[Example],
    labels: Sequence[str],
    *,
    notes: str = "",
) -> Result:
    """Fit ``method`` on ``train``, score it on ``test``, and time both steps.

    ``test`` must be the official test split. Nothing here inspects the test rows
    before prediction, so the measurement cannot leak into fitting.
    """
    labels = list(labels)
    started = time.perf_counter()
    method.fit(train)
    fit_seconds = time.perf_counter() - started

    texts = [ex.text for ex in test]
    truth = [ex.label for ex in test]

    started = time.perf_counter()
    predictions = method.predict(texts)
    predict_seconds = time.perf_counter() - started

    if len(predictions) != len(truth):
        raise ValueError(f"{method.name}: produced {len(predictions)} predictions for {len(truth)} rows")

    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=range(len(labels)), zero_division=0
    )
    per_class = {
        label: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }

    return Result(
        method=method.name,
        family=method.family,
        train_size=len(train),
        test_size=len(test),
        accuracy=round(float(accuracy_score(truth, predictions)), 4),
        macro_f1=round(float(f1_score(truth, predictions, average="macro", zero_division=0)), 4),
        fit_seconds=round(fit_seconds, 4),
        predict_seconds=round(predict_seconds, 4),
        per_class=per_class,
        confusion=confusion_matrix(truth, predictions, labels=range(len(labels))).tolist(),
        labels=labels,
        notes=notes,
        predictions=list(predictions),
    )


def format_table(results: Sequence[Result]) -> str:
    """A fixed-width comparison table, printable in a terminal and pasteable into a report."""
    header = (
        f"{'method':<44}{'family':<20}{'train':>7}{'acc':>8}{'macroF1':>9}"
        f"{'fit(s)':>9}{'inf(s)':>9}"
    )
    rule = "-" * len(header)
    lines = [rule, header, rule]
    for r in sorted(results, key=lambda x: -x.accuracy):
        lines.append(
            f"{r.method[:43]:<44}{r.family[:19]:<20}{r.train_size:>7}"
            f"{r.accuracy:>8.3f}{r.macro_f1:>9.3f}{r.fit_seconds:>9.3f}{r.predict_seconds:>9.3f}"
        )
    lines.append(rule)
    return "\n".join(lines)
