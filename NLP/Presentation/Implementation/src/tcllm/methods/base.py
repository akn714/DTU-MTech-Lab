"""The tiny interface every classifier in this project implements.

Keeping it to two methods is deliberate: every method — TF-IDF baselines, few-shot,
zero-shot, synthetic-trained — is fitted and queried the same way, so the comparison
loop does not need to know anything about how a particular method works.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ..data import Example


@runtime_checkable
class Method(Protocol):
    """A text classifier with a name, a family, and fit/predict."""

    name: str
    family: str
    train_size: int

    def fit(self, train: Sequence[Example]) -> None:
        """Train. ``train`` rows come from the official train split only."""

    def predict(self, texts: Sequence[str]) -> list[int]:
        """Return one integer label index per input text."""
