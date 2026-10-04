"""Optional learned category model (gradient boosting on track features).

Train it from labelled tracks (CVAT import) once real labels exist; until then
the rule classifier is the reference. Low-confidence predictions return
``unknown`` so the learned model can never be more assertive than its
probabilities support.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ..types import Category, CategoryDecision
from .features import FEATURE_NAMES, feature_vector


class TrackCategoryModel:
    def __init__(self, model=None, classes: list[str] | None = None, min_proba: float = 0.6):
        self.model = model
        self.classes = classes or []
        self.min_proba = min_proba

    @classmethod
    def train(cls, feats: list[dict], labels: list[str], seed: int = 0, min_proba: float = 0.6):
        from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore

        X = np.stack([feature_vector(f) for f in feats])
        X = np.nan_to_num(X, nan=-1.0)
        y = np.array(labels)
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed, class_weight="balanced")
        clf.fit(X, y)
        return cls(clf, list(clf.classes_), min_proba)

    def predict(self, f: dict) -> CategoryDecision:
        if self.model is None:
            return CategoryDecision(Category.UNKNOWN, {}, ["no model"])
        X = np.nan_to_num(feature_vector(f)[None, :], nan=-1.0)
        p = self.model.predict_proba(X)[0]
        scores = {c: float(v) for c, v in zip(self.classes, p)}
        k = int(np.argmax(p))
        if p[k] < self.min_proba:
            return CategoryDecision(Category.UNKNOWN, scores, [f"model unsure (max p={p[k]:.2f})"])
        return CategoryDecision(Category.parse(self.classes[k]), scores, [f"model p={p[k]:.2f}"])

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump({"model": self.model, "classes": self.classes, "min_proba": self.min_proba,
                         "features": FEATURE_NAMES}, fh)

    @classmethod
    def load(cls, path: str | Path) -> "TrackCategoryModel":
        with open(path, "rb") as fh:
            d = pickle.load(fh)
        if d.get("features") != FEATURE_NAMES:
            raise ValueError("feature set changed since this model was trained; retrain it")
        return cls(d["model"], d["classes"], d["min_proba"])
