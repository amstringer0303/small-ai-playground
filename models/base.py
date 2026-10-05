from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ModelAdapter(ABC):
    @abstractmethod
    def fit(self, x, y, sample_weight, config, validation=None):
        """Train on prepared data; evaluation holdout is never an argument."""

    @abstractmethod
    def predict_proba(self, x) -> np.ndarray:
        """Return class probabilities in the recorded class order."""

    @abstractmethod
    def parameters(self) -> dict:
        """Expose inspectable learned state."""


def preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = [c for c in frame if pd.api.types.is_numeric_dtype(frame[c])]
    categorical = [c for c in frame if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("missing", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("category", Pipeline([
            ("missing", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)


def decisions(probabilities, classes, positive_label, threshold):
    probabilities = np.asarray(probabilities)
    positive = classes.index(positive_label)
    alternative = probabilities.copy()
    alternative[:, positive] = -1
    indices = np.where(probabilities[:, positive] >= threshold, positive, alternative.argmax(axis=1))
    return np.array(classes)[indices]


@dataclass
class FittedModel:
    model_id: str
    features: list[str]
    classes: list[str]
    preprocessing: ColumnTransformer
    adapter: ModelAdapter
    defaults: dict
    encoded_features: list[str]

    def probabilities(self, frame):
        missing = set(self.features) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing model inputs: {', '.join(sorted(missing))}")
        prepared = self.preprocessing.transform(frame[self.features])
        return self.adapter.predict_proba(prepared)

    def predict(self, frame, positive_label="alert", threshold=0.5):
        return decisions(self.probabilities(frame), self.classes, positive_label, threshold)

    def example(self, inputs: dict) -> pd.DataFrame:
        row = {feature: inputs.get(feature, self.defaults[feature]) for feature in self.features}
        return pd.DataFrame([row])
