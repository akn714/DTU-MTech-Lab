"""Zero-shot classification with no training data and no pretrained model.

The paper's zero-shot method is LLM prompting. That needs a multi-GB model
download, which this machine will not do, so this module implements the *other*
zero-shot family the paper surveys in its related work: dataless classification
from label names / seed words (Liu et al., 2004, "Text classification by labeling
words" — cited by the paper as a pre-LLM zero-shot approach).

So this is a real zero-shot classifier, not a stub: it never sees a single labelled
example, and its accuracy is directly comparable in kind to a zero-shot LLM's —
just far weaker, because a hand-written lexicon is a much poorer opinion model than
a 7B-parameter language model. Expect roughly 0.40-0.55 where the paper's zero-shot
LLMs reach 0.58-0.71 for English.

This is a documented deviation. See docs/deviations.md.
"""

from __future__ import annotations

import re
from typing import Sequence

from ..data import Example

# Words that flip the polarity of a nearby opinion word.
NEGATORS = {
    "not", "no", "never", "none", "nothing", "nobody", "neither", "nor",
    "without", "hardly", "barely", "scarcely",
    "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "can't", "cannot", "won't", "wouldn't", "shouldn't", "couldn't",
    "hasn't", "haven't", "hadn't", "ain't",
}

POSITIVE_WORDS = {
    "love", "loved", "loves", "loving", "lovely", "adore", "adorable",
    "great", "good", "awesome", "amazing", "excellent", "fantastic", "wonderful",
    "brilliant", "best", "better", "nice", "happy", "glad", "joy", "joyful",
    "excited", "exciting", "perfect", "beautiful", "terrific", "superb",
    "outstanding", "incredible", "impressive", "recommend", "recommended",
    "thanks", "thank", "cool", "sweet", "fun", "funny", "enjoy", "enjoyed",
    "enjoying", "win", "won", "winner", "success", "successful", "positive",
    "pleased", "delighted", "thrilled", "satisfied", "helpful", "useful",
    "improved", "improvement", "congrats", "congratulations", "yay", "wow",
    "cute", "proud", "blessed", "grateful", "favorite", "favourite", "smile",
    "smiling", "laugh", "laughing", "brighter", "refreshing", "flawless",
}

NEGATIVE_WORDS = {
    "hate", "hated", "hates", "awful", "terrible", "horrible", "horrendous",
    "bad", "worse", "worst", "sad", "angry", "mad", "upset", "disappointed",
    "disappointing", "annoying", "annoyed", "stupid", "dumb", "boring", "bored",
    "ugly", "sucks", "suck", "sucked", "fail", "failed", "failing", "failure",
    "wrong", "broken", "broke", "breaks", "problem", "problems", "issue", "issues",
    "error", "errors", "crash", "crashed", "useless", "waste", "wasted",
    "pathetic", "ridiculous", "garbage", "trash", "crap", "poor", "unfortunately",
    "sadly", "hurt", "hurts", "pain", "painful", "sick", "disgusting",
    "disgusted", "offensive", "rude", "mean", "mess", "messy", "delay",
    "delayed", "cancel", "cancelled", "canceled", "refund", "complaint",
    "complain", "frustrating", "frustrated", "disaster", "unacceptable",
    "overpriced", "scam", "lied", "liar", "nasty", "grim",
}

_TOKEN = re.compile(r"[a-z']+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def polarity_score(text: str, negation_window: int = 3) -> int:
    """Sum of opinion words, with polarity flipped by a preceding negator.

    "the food was not good" scores negative rather than positive, which a naive
    word-count classifier gets wrong and which matters on a three-class task.
    """
    tokens = _tokenize(text)
    score = 0
    for index, token in enumerate(tokens):
        if token in POSITIVE_WORDS:
            weight = 1
        elif token in NEGATIVE_WORDS:
            weight = -1
        else:
            continue
        window = tokens[max(0, index - negation_window):index]
        if any(negator in NEGATORS for negator in window):
            weight = -weight
        score += weight
    return score


class SeedWordZeroShot:
    """Assigns the label whose name best matches the text's opinion polarity.

    Fitting is a no-op by design — that is what makes it zero-shot. ``fit`` exists
    only so every method in the project satisfies the same interface.
    """

    family = "zero_shot"

    def __init__(self, labels: Sequence[str], name: str = "Zero-shot seed lexicon (Liu et al., 2004)"):
        self.labels = list(labels)
        self.name = name
        self.train_size = 0
        self._index = {label: i for i, label in enumerate(self.labels)}
        for required in ("positive", "negative", "neutral"):
            if required not in self._index:
                raise ValueError(f"seed lexicon needs a {required!r} label; got {self.labels}")

    def fit(self, train: Sequence[Example]) -> None:  # noqa: ARG002 - intentionally unused
        if train:
            raise ValueError(
                "SeedWordZeroShot is a zero-shot classifier and must not be fitted on labelled "
                "data; pass an empty training set so the comparison stays honest."
            )
        self.train_size = 0

    def predict(self, texts: Sequence[str]) -> list[int]:
        positive = self._index["positive"]
        negative = self._index["negative"]
        neutral = self._index["neutral"]

        predictions = []
        for text in texts:
            score = polarity_score(text)
            if score > 0:
                predictions.append(positive)
            elif score < 0:
                predictions.append(negative)
            else:
                predictions.append(neutral)
        return predictions
