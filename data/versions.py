import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data.datasets import DEFAULT_FEATURES, load_air_quality


def digest(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def normalize(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy().dropna(how="all")
    if not {"example_id", "included", "condition"}.issubset(frame.columns):
        raise ValueError("Keep the example_id, included, and condition columns.")
    if len(frame) > 5000:
        raise ValueError("The prototype supports at most 5,000 examples.")
    if frame.empty:
        raise ValueError("A dataset cannot be empty.")
    frame["example_id"] = frame["example_id"].fillna("").astype(str).str.strip()
    for index in frame.index[frame["example_id"].isin(["", "nan", "None", "new"])]:
        frame.at[index, "example_id"] = f"added-{uuid4().hex[:12]}"
    if frame["example_id"].duplicated().any():
        raise ValueError("Example IDs must be unique; leave new IDs blank.")
    raw_flags = frame["included"].astype(str).str.lower().str.strip()
    if not raw_flags.isin(["true", "false", "1", "0", "1.0", "0.0"]).all():
        raise ValueError("Included must be true/false or 1/0 for every example.")
    frame["included"] = raw_flags.isin(["true", "1", "1.0"])
    if frame["condition"].isna().any():
        raise ValueError("Every example needs a condition label.")
    frame["condition"] = frame["condition"].astype(str).str.strip()
    if frame["condition"].eq("").any():
        raise ValueError("Condition labels cannot be blank.")
    for column in frame.columns:
        if column in {"example_id", "included", "condition", "site", "sensor"}:
            continue
        if column in DEFAULT_FEATURES or pd.api.types.is_numeric_dtype(frame[column]):
            converted = pd.to_numeric(frame[column], errors="coerce")
            if (converted.isna() & frame[column].notna()).any():
                raise ValueError(f"{column} must contain numeric values.")
            frame[column] = converted
        else:
            converted = pd.to_numeric(frame[column], errors="coerce")
            if converted.notna().sum() == frame[column].notna().sum():
                frame[column] = converted
            else:
                frame[column] = frame[column].replace("", np.nan)
        if pd.api.types.is_numeric_dtype(frame[column]) and np.isinf(frame[column]).any():
            raise ValueError(f"{column} contains an infinite value.")
    return frame.sort_values("example_id").reset_index(drop=True)


@dataclass(frozen=True)
class DatasetVersion:
    id: str
    parent_id: str | None
    name: str
    note: str
    created_at: str
    features: list[str]
    sensitive_features: list[str]
    test_fraction: float
    split_seed: int
    label_author: str
    content_hash: str
    changes: dict


class DatasetRegistry:
    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        if not self.all():
            self.save(load_air_quality(), DEFAULT_FEATURES, ["latitude", "longitude", "site", "sensor"],
                      "Original synthetic data", "Generated with seed 23", "Synthetic rule")

    def all(self) -> list[DatasetVersion]:
        versions = [DatasetVersion(**json.loads(path.read_text())) for path in self.directory.glob("*/version.json")]
        return sorted(versions, key=lambda version: version.created_at)

    def get(self, version_id: str) -> DatasetVersion:
        for version in self.all():
            if version.id == version_id:
                return version
        raise ValueError("Unknown dataset version.")

    def frame(self, version_id: str) -> pd.DataFrame:
        version = self.get(version_id)
        frame = normalize(pd.read_csv(self.directory / version.id / "data.csv", float_precision="round_trip"))
        if digest(frame.to_json(orient="split", double_precision=10)) != version.content_hash:
            raise ValueError("Dataset snapshot was modified on disk; restore the saved version.")
        return frame

    def reference(self, fraction: float, seed: int) -> pd.DataFrame:
        original = self.frame(self.all()[0].id)
        _, test = train_test_split(original, test_size=fraction, random_state=seed, stratify=original.condition)
        return test.sort_values("example_id").reset_index(drop=True)

    def training_frame(self, version_id: str) -> pd.DataFrame:
        version = self.get(version_id)
        frame = self.frame(version_id)
        reference = self.reference(version.test_fraction, version.split_seed)
        return frame.loc[~frame.example_id.isin(reference.example_id)].reset_index(drop=True)

    def evaluation_id(self, version: DatasetVersion) -> str:
        reference = self.reference(version.test_fraction, version.split_seed)
        return digest(reference[["example_id", "condition"]].to_dict("records"))[:12]

    def save(self, frame, features, sensitive, name, note, author,
             parent_id=None, test_fraction=0.2, split_seed=42) -> DatasetVersion:
        if not 0.1 <= test_fraction <= 0.4:
            raise ValueError("Holdout fraction must be between 10% and 40%.")
        frame = normalize(frame)
        features = list(dict.fromkeys(features))
        if not features or any(f not in frame.columns or f in {"condition", "example_id", "included"} for f in features):
            raise ValueError("Choose at least one valid input feature; labels and IDs cannot be inputs.")
        if any(frame[f].isna().all() for f in features):
            raise ValueError("Each selected feature needs at least one value.")
        sensitive = list(dict.fromkeys(sensitive))
        if any(f not in frame.columns for f in sensitive):
            raise ValueError("Sensitive features must exist in the dataset.")
        changes = {}
        if parent_id:
            parent = self.get(parent_id)
            before = self.frame(parent_id).set_index("example_id")
            after = frame.set_index("example_id")
            common = before.index.intersection(after.index)
            common_columns = before.columns.intersection(after.columns)
            cell_changes = before.loc[common, common_columns].fillna("<missing>").ne(
                after.loc[common, common_columns].fillna("<missing>"))
            changes = {
                "added_examples": len(after.index.difference(before.index)),
                "removed_examples": len(before.index.difference(after.index)),
                "changed_labels": int(cell_changes["condition"].sum()),
                "edited_cells": int(cell_changes.to_numpy().sum()),
                "excluded_examples": int((~frame.included).sum()),
                "removed_features": sorted(set(parent.features) - set(features)),
                "added_features": sorted(set(features) - set(parent.features)),
                "split_changed": (parent.test_fraction, parent.split_seed) != (test_fraction, split_seed),
            }
            # Original evaluation rows cannot be rewritten through a dataset branch.
            original = self.frame(self.all()[0].id).set_index("example_id")
            protected = self.reference(test_fraction, split_seed).example_id
            if not set(protected).issubset(after.index):
                raise ValueError("Keep held-out examples; use the training editor to exclude observations.")
            for column in original.columns:
                if column not in after.columns:
                    raise ValueError("Remove features from the input selection, not the raw snapshot columns.")
                left = after.loc[protected, column].fillna("<missing>")
                right = original.loc[protected, column].fillna("<missing>")
                if pd.api.types.is_numeric_dtype(after[column]) and pd.api.types.is_numeric_dtype(original[column]):
                    changed = not np.allclose(pd.to_numeric(left), pd.to_numeric(right), equal_nan=True, rtol=0, atol=1e-9)
                else:
                    changed = left.astype(str).ne(right.astype(str)).any()
                if changed:
                    raise ValueError("This split would put edited observations in the holdout. Choose another split or revert those edits.")
        content_hash = digest(frame.to_json(orient="split", double_precision=10))
        version = DatasetVersion(
            id=f"data-{uuid4().hex[:12]}", parent_id=parent_id, name=name.strip() or "Dataset variant",
            note=note, created_at=datetime.now(timezone.utc).isoformat(), features=features,
            sensitive_features=sensitive, test_fraction=float(test_fraction), split_seed=int(split_seed),
            label_author=author.strip() or "Unnamed profile", content_hash=content_hash, changes=changes,
        )
        target = self.directory / version.id
        target.mkdir()
        frame.to_csv(target / "data.csv", index=False, float_format="%.17g")
        (target / "version.json").write_text(json.dumps(asdict(version), indent=2), encoding="utf-8")
        return version
