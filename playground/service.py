import json
import threading
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from data.datasets import GEO_FEATURES
from experiments.comparison import compare_goals, compare_records, prediction_changes
from experiments.manager import ExperimentManager
from experiments.runner import run_experiment
from experiments.schema import ExperimentConfig
from playground.community import QUESTIONS, community_goals

CHOICES = {
    "missed": "Give missed alerts more importance",
    "flagged": "Leave out flagged readings",
    "labels": "Edit labels or include readings",
    "location": "Leave out location information",
}
COLUMNS = {
    "example_id": "Reading", "included": "Use reading", "pm25": "PM2.5",
    "pm10": "PM10", "wind": "Wind", "site": "Site", "unreliable": "Flagged", "condition": "Label",
}


class Playground:
    def __init__(self, workspace: Path):
        self.manager = ExperimentManager(workspace)
        self.directory = self.manager.root / "comparisons"
        self.directory.mkdir(exist_ok=True)
        self.lock = threading.RLock()
        if not self.manager.records() and len(self.manager.goals()) == 1:
            self.manager.save_goals(community_goals())
        self.original = self.manager.datasets.all()[0]

    def editor(self):
        frame = self.manager.datasets.training_frame(self.original.id)[list(COLUMNS)].rename(columns=COLUMNS)
        frame["Flagged"] = frame["Flagged"].map({0: "No", 1: "Yes"})
        return frame

    def pairs(self):
        pairs = [json.loads(path.read_text(encoding="utf-8")) for path in self.directory.glob("*/comparison.json")]
        return sorted(pairs, key=lambda pair: pair["created_at"])

    def pair(self, identifier):
        for pair in self.pairs():
            if pair["id"] == identifier:
                return pair
        raise ValueError("Choose a saved comparison.")

    def goal_id(self):
        pairs = self.pairs()
        return pairs[0]["goal_id"] if pairs else self.manager.goals()[-1]["id"]

    def edited_frame(self, editor):
        if set(editor.columns) != set(COLUMNS.values()):
            raise ValueError("Keep the reading-table columns unchanged.")
        edited = editor.rename(columns={label: name for name, label in COLUMNS.items()}).copy()
        training = self.manager.datasets.training_frame(self.original.id)
        edited["example_id"] = edited.example_id.astype(str)
        if edited.example_id.duplicated().any() or not set(edited.example_id).issubset(set(training.example_id)):
            raise ValueError("Reading IDs are fixed. Only edit training readings; use Advanced to add new examples.")
        edited["condition"] = edited.condition.astype(str).str.strip()
        edited["unreliable"] = edited["unreliable"].map({"No": 0, "Yes": 1})
        fixed = ["pm25", "pm10", "wind", "site", "unreliable"]
        a = edited.set_index("example_id")[fixed]
        b = training.set_index("example_id").loc[a.index, fixed]
        if not a.eq(b).to_numpy().all():
            raise ValueError("Only labels and Use reading can change here. Advanced allows measurement edits.")
        if not edited.condition.isin(["normal", "alert"]).all():
            raise ValueError("Use normal or alert as the label. Advanced supports additional categories.")
        full = self.manager.datasets.frame(self.original.id).set_index("example_id")
        for column in edited.columns:
            if column != "example_id":
                full.loc[edited.example_id, column] = edited.set_index("example_id")[column]
        missing = set(training.example_id) - set(edited.example_id)
        full.loc[list(missing), "included"] = False
        from data.versions import normalize
        return normalize(full.reset_index())

    def variant(self, choice, editor, number):
        original_frame = self.manager.datasets.frame(self.original.id)
        edited = self.edited_frame(editor)
        changed = not edited.equals(original_frame)
        if choice != "labels" and changed:
            raise ValueError("You have edited readings. Choose Edit labels or include readings, or reset the table first.")
        features = list(self.original.features)
        if choice == "missed":
            return self.original
        frame = original_frame.copy()
        if choice == "labels":
            if not changed:
                raise ValueError("Change a label, reading value, or Use reading checkbox before training this comparison.")
            frame = edited
        elif choice == "flagged":
            training = self.manager.datasets.training_frame(self.original.id)
            flagged = training.loc[training.unreliable == 1, "example_id"]
            frame.loc[frame.example_id.isin(flagged), "included"] = False
        elif choice == "location":
            features = [feature for feature in features if feature not in GEO_FEATURES]
        else:
            raise ValueError("Choose one supported intervention.")
        return self.manager.datasets.save(
            frame, features, self.original.sensitive_features, f"Run {number}: {CHOICES[choice]}",
            CHOICES[choice], "Playground participant", parent_id=self.original.id,
            test_fraction=self.original.test_fraction, split_seed=self.original.split_seed,
        )

    def run(self, choice="missed", penalty=5, editor=None):
        if choice not in CHOICES:
            raise ValueError("Choose one supported intervention.")
        if not 1 <= penalty <= 10:
            raise ValueError("Alert-example importance must be between 1 and 10.")
        with self.lock:
            number = len(self.pairs()) + 1
            variant = self.variant(choice, self.editor() if editor is None else editor, number)
            goal_id = self.goal_id()
            baseline_config = ExperimentConfig(
                name="Original model", hypothesis="Measure the unchanged synthetic dataset with a 1x alert loss weight.",
                dataset_id=self.original.id, goal_id=goal_id, model_id="tiny_neural",
                seed=42, epochs=60, hidden_units=8, learning_rate=0.02,
            )
            original = None
            for previous in self.pairs():
                candidate = self.manager.record(previous["original_id"])
                if candidate["config"] == baseline_config.__dict__:
                    original = candidate
                    break
            reused = original is not None
            if original is None:
                original = run_experiment(self.manager, baseline_config)
            decision = f"Alert examples receive {penalty:g}x training loss weight" if choice == "missed" else CHOICES[choice]
            config = replace(baseline_config, name=f"Run {number}: {CHOICES[choice]}", hypothesis=decision,
                             dataset_id=variant.id, false_negative_cost=float(penalty) if choice == "missed" else 1.0)
            changed = run_experiment(self.manager, config)
            pair = {
                "id": f"comparison-{uuid4().hex[:12]}", "number": number,
                "created_at": datetime.now(timezone.utc).isoformat(), "choice": choice, "decision": decision,
                "penalty": float(penalty) if choice == "missed" else 1.0, "goal_id": goal_id,
                "community_question": QUESTIONS[choice],
                "original_id": original["id"], "changed_id": changed["id"], "baseline_reused": reused,
                "evaluation_id": original["evaluation_id"], "source": "Synthetic: seed 23; 720 fictional readings",
            }
            target = self.directory / pair["id"]
            target.mkdir()
            (target / "comparison.json").write_text(json.dumps(pair, indent=2), encoding="utf-8")
            return pair

    def records(self, pair):
        return self.manager.record(pair["original_id"]), self.manager.record(pair["changed_id"])

    def history(self):
        rows = []
        for pair in self.pairs():
            original, changed = self.records(pair)
            rows.append({"Run": pair["number"], "Decision": pair["decision"],
                         "Accuracy": f"{changed['metrics']['accuracy']:.1%}",
                         "Missed alerts": changed["metrics"]["false_negatives"],
                         "False alarms": changed["metrics"]["false_positives"],
                         "Changed predictions": prediction_changes(original, changed)["changed_predictions"]})
        return pd.DataFrame(rows, columns=["Run", "Decision", "Accuracy", "Missed alerts", "False alarms", "Changed predictions"])

    def export(self, identifier):
        pair = self.pair(identifier)
        original, changed = self.records(pair)
        goal = self.manager.goal(pair["goal_id"])
        path = self.directory / pair["id"] / "experiment-record.zip"
        payload = {"comparison": pair, "original": original, "your_choice": changed,
                   "limitations": ["Synthetic observations and labels; not health guidance.",
                     "Input removal does not erase older raw snapshots.",
                     "A visible holdout is not a final untouched test set.",
                     "Raw datasets and executable checkpoints are omitted; replay requires the local workspace."]}
        changed_predictions = prediction_changes(original, changed)
        summary = "\n".join([
            "# Small / Local AI experiment", "", pair["decision"], "",
            "Data: 720 fictional readings generated with seed 23; no live environmental feed.",
            f"Model: small CPU PyTorch network; seed 42; {original['metrics']['test_examples']} fixed test readings.", "",
            f"Original: accuracy {original['metrics']['accuracy']:.1%}; missed alerts {original['metrics']['false_negatives']}; false alarms {original['metrics']['false_positives']}.",
            f"Your choice: accuracy {changed['metrics']['accuracy']:.1%}; missed alerts {changed['metrics']['false_negatives']}; false alarms {changed['metrics']['false_positives']}.",
            f"Changed classifications: {changed_predictions['changed_predictions']}.", "",
            "The complete configurations, data-version hashes, learned weights, goals, and runtime are in experiment.json.",
            "Replay requires the locally saved dataset and checkpoint files; this export omits those files.",
        ])
        with ZipFile(path, "w", ZIP_DEFLATED) as archive:
            archive.writestr("summary.md", summary)
            archive.writestr("experiment.json", json.dumps(payload, indent=2, allow_nan=False))
            archive.writestr("metrics.csv", compare_records([original, changed]).to_csv(index=False))
            archive.writestr("original_goals.csv", compare_goals([original, changed], goal).to_csv(index=False))
        return path
