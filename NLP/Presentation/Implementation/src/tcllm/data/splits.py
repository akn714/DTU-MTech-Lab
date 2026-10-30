"""Official-split handling and the leakage guards that protect them.

Project rule 4: preserve the official train/validation/test splits, and never use
test data for training, for synthetic-data generation targets, or for
hyperparameter selection.

The guard here is *structural*, not documentary. :class:`SplitBundle` holds the
three splits, and every method that manufactures training material
(:meth:`SplitBundle.few_shot_sample`) reads ``self.train`` directly. No code path
in this module can hand a test row to a learner, so the rule cannot be broken by
a careless call site. Test rows are reachable only through
:meth:`SplitBundle.final_eval`, whose name states the intent wherever it appears.

:func:`assert_no_test_leakage` closes the remaining hole: rows produced
*elsewhere* — a synthetic corpus, an augmented set — being traced back to the
test split.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from ..utils.io import fingerprint_rows

SPLIT_NAMES = ("train", "validation", "test")


class LeakageError(RuntimeError):
    """Raised when rows destined for training overlap the held-out test split."""


@dataclass(frozen=True)
class Example:
    """One labelled example.

    ``label`` is the integer index into the task's label list; ``label_name`` is
    kept alongside it because every prompt in the paper is written in terms of
    label *words* ("positive or negative or neutral"), not indices.
    """

    text: str
    label: int
    label_name: str

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "label": self.label, "label_name": self.label_name}

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Example":
        return cls(text=row["text"], label=int(row["label"]), label_name=row["label_name"])


def normalize_for_dedup(text: str) -> str:
    """Whitespace- and case-insensitive form used only for duplicate detection."""
    return " ".join(text.lower().split())


@dataclass
class SplitBundle:
    """The official splits for one (task, language) pair."""

    task: str
    language: str          # registry key, e.g. "english"
    language_iso: str      # e.g. "en"
    labels: Sequence[str]
    train: list[Example]
    validation: list[Example]
    test: list[Example]
    source: dict[str, Any] = field(default_factory=dict)

    # -- access ------------------------------------------------------------ #
    def split(self, name: str) -> list[Example]:
        if name not in SPLIT_NAMES:
            raise KeyError(f"unknown split {name!r}; expected one of {SPLIT_NAMES}")
        return getattr(self, name)

    def sizes(self) -> dict[str, int]:
        return {name: len(self.split(name)) for name in SPLIT_NAMES}

    def label_distribution(self, name: str) -> dict[str, int]:
        counts = Counter(ex.label_name for ex in self.split(name))
        return {label: counts.get(label, 0) for label in self.labels}

    def fingerprints(self) -> dict[str, str]:
        return {name: fingerprint_rows(ex.to_dict() for ex in self.split(name)) for name in SPLIT_NAMES}

    def final_eval(self) -> list[Example]:
        """The held-out test rows.

        Deliberately verbose: any call site using this is doing final evaluation
        only. Validation rows exist for model selection.
        """
        return self.test

    # -- training material (train split only, by construction) -------------- #
    def few_shot_sample(self, n_per_label: int = 10, seed: int = 42) -> list[Example]:
        """Draw ``n_per_label`` TRAIN examples per class.

        This is the paper's few-shot setting (FastFit, 10 examples per label).
        Reads ``self.train`` only — validation and test are not reachable here.
        """
        rng = random.Random(seed)
        by_label: dict[int, list[Example]] = defaultdict(list)
        for ex in self.train:
            by_label[ex.label].append(ex)

        sampled: list[Example] = []
        for label_index in range(len(self.labels)):
            pool = by_label.get(label_index, [])
            if len(pool) < n_per_label:
                raise ValueError(
                    f"{self.task}/{self.language}: label {self.labels[label_index]!r} has only "
                    f"{len(pool)} train examples, cannot draw {n_per_label}"
                )
            sampled.extend(rng.sample(pool, n_per_label))
        rng.shuffle(sampled)
        return sampled

    def train_examples_for_label(self, label_index: int, n: int, seed: int = 42) -> list[Example]:
        """TRAIN examples of a single class, for prompt inspection or seed-based generation."""
        pool = [ex for ex in self.train if ex.label == label_index]
        if n >= len(pool):
            return list(pool)
        return random.Random(seed).sample(pool, n)

    # -- metadata ---------------------------------------------------------- #
    def meta(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "language": self.language,
            "language_iso": self.language_iso,
            "labels": list(self.labels),
            "sizes": self.sizes(),
            "fingerprints": self.fingerprints(),
            "source": dict(self.source),
        }


# --------------------------------------------------------------------------- #
# guards
# --------------------------------------------------------------------------- #
def assert_official_sizes(bundle: SplitBundle, expected: dict[str, int]) -> None:
    """Check the loaded splits against the sizes declared in configs/datasets.yaml.

    A mismatch means either upstream data changed or we loaded the wrong files.
    Both are silent killers of a reproduction, so this fails loudly.
    """
    actual = bundle.sizes()
    problems = [
        f"{name}: expected {expected[name]}, loaded {actual[name]}"
        for name in expected
        if name in actual and actual[name] != expected[name]
    ]
    if problems:
        raise ValueError(
            f"{bundle.task}/{bundle.language}: official split sizes do not match "
            f"configs/datasets.yaml (paper Table 1): " + "; ".join(problems)
        )


def assert_no_test_leakage(
    rows: Iterable[Example],
    bundle: SplitBundle,
    *,
    context: str = "",
) -> None:
    """Fail if any of ``rows`` duplicates a held-out test example.

    Call this before any synthetic corpus, augmentation or few-shot set is used
    for training. Duplication between train and validation is tolerated (and
    reported separately) because it does not invalidate a test score; overlap
    with test does.
    """
    test_texts = {normalize_for_dedup(ex.text) for ex in bundle.test}
    hits = [ex.text for ex in rows if normalize_for_dedup(ex.text) in test_texts]
    if hits:
        where = f" ({context})" if context else ""
        sample = "; ".join(repr(t[:70]) for t in hits[:3])
        raise LeakageError(
            f"{bundle.task}/{bundle.language}{where}: {len(hits)} row(s) duplicate the held-out "
            f"test split. Test data must never be used for training or generation. Examples: {sample}"
        )


def overlap_report(bundle: SplitBundle) -> dict[str, int]:
    """Count exact-duplicate texts within and between splits.

    Reported rather than enforced: tweets legitimately repeat, and train/validation
    overlap does not invalidate a test score. Any train/test or validation/test
    overlap is surfaced so it can be judged rather than assumed away.
    """
    normalised = {name: [normalize_for_dedup(ex.text) for ex in bundle.split(name)] for name in SPLIT_NAMES}
    sets = {name: set(values) for name, values in normalised.items()}

    report = {
        f"{name}_duplicates": len(values) - len(sets[name])
        for name, values in normalised.items()
    }
    report["train_validation_overlap"] = len(sets["train"] & sets["validation"])
    report["train_test_overlap"] = len(sets["train"] & sets["test"])
    report["validation_test_overlap"] = len(sets["validation"] & sets["test"])
    return report
