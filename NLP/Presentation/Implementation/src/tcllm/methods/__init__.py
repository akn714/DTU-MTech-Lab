"""Classifier methods: TF-IDF baselines, zero-shot lexicon, synthetic generator."""

from .base import Method
from .classical import (
    TfidfClassifier,
    make_few_shot_logreg,
    make_logreg,
    make_naive_bayes,
    make_svm,
)
from .synthetic import DEFAULT_PER_LABEL, generate_synthetic
from .zeroshot import SeedWordZeroShot, polarity_score

__all__ = [
    "DEFAULT_PER_LABEL",
    "Method",
    "SeedWordZeroShot",
    "TfidfClassifier",
    "generate_synthetic",
    "make_few_shot_logreg",
    "make_logreg",
    "make_naive_bayes",
    "make_svm",
    "polarity_score",
]
