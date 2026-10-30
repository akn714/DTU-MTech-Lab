"""TF-IDF classifier family.

These stand in for the paper's embedding-based classifiers. The paper feeds
`gte-multilingual-base` embeddings to a logistic regression; this uses TF-IDF
features instead, because the embedding model is a ~1.2 GB download and this
machine has no GPU. The *pipeline* is identical — vectorise, fit a linear
classifier, evaluate on the held-out test split — and the numbers are therefore
comparable in kind, not in value.

This is a documented deviation. See docs/deviations.md.
"""

from __future__ import annotations

from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from ..data import Example

# Word 1-2 grams, sublinear term frequency: the standard starting point for short
# informal text, and what makes a TF-IDF baseline hard to beat on tweets.
DEFAULT_VECTORIZER = dict(
    lowercase=True,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=1,
    strip_accents="unicode",
)


class TfidfClassifier:
    """TF-IDF features + a scikit-learn classifier, behind the Method interface."""

    def __init__(self, classifier, name: str, family: str, *, vectorizer_kwargs: dict | None = None):
        self.classifier = classifier
        self.name = name
        self.family = family
        self.train_size = 0
        self.vectorizer = TfidfVectorizer(**(vectorizer_kwargs or DEFAULT_VECTORIZER))

    def fit(self, train: Sequence[Example]) -> None:
        texts = [ex.text for ex in train]
        labels = [ex.label for ex in train]
        if len(set(labels)) < 2:
            raise ValueError(f"{self.name}: training data has only one class")
        matrix = self.vectorizer.fit_transform(texts)
        self.classifier.fit(matrix, labels)
        self.train_size = len(train)

    def predict(self, texts: Sequence[str]) -> list[int]:
        return [int(value) for value in self.classifier.predict(self.vectorizer.transform(texts))]

    def top_features(self, label_names: Sequence[str], n: int = 12) -> dict[str, list[str]]:
        """Most weighted features per class.

        Useful for the report: a classifier that cannot be inspected is hard to trust,
        and on 3-class sentiment the top words should look obviously right.
        """
        if not hasattr(self.classifier, "coef_"):
            return {}
        names = self.vectorizer.get_feature_names_out()
        out: dict[str, list[str]] = {}
        coef = self.classifier.coef_
        for index, label in enumerate(label_names):
            row = coef[index] if coef.shape[0] > index else coef[0]
            order = row.argsort()[::-1][:n]
            out[label] = [str(names[i]) for i in order]
        return out


def make_logreg(seed: int = 42, **kwargs) -> TfidfClassifier:
    """The project's main baseline, and the paper's own strongest cheap classifier."""
    return TfidfClassifier(
        LogisticRegression(max_iter=1000, random_state=seed, **kwargs),
        name="TF-IDF + Logistic Regression",
        family="supervised_full",
    )


def make_naive_bayes() -> TfidfClassifier:
    return TfidfClassifier(MultinomialNB(), name="TF-IDF + Naive Bayes", family="supervised_full")


def make_svm(seed: int = 42) -> TfidfClassifier:
    return TfidfClassifier(
        LinearSVC(random_state=seed), name="TF-IDF + Linear SVM", family="supervised_full"
    )


def make_few_shot_logreg(seed: int = 42) -> TfidfClassifier:
    """Same classifier as :func:`make_logreg`, but intended for a handful of examples.

    Used by the few-shot experiments, which draw 10/50/100 examples per label from
    the train split. Kept as a separate constructor so the report can label the rows
    by method family without inspecting the training size.
    """
    return TfidfClassifier(
        LogisticRegression(max_iter=1000, random_state=seed),
        name="Few-shot Logistic Regression",
        family="few_shot",
    )
