import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import joblib

from data.datasets import ROOT
from data.versions import DatasetRegistry
from values.schema import GoalSpec


class ExperimentManager:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or ROOT / "outputs" / "lab").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.datasets = DatasetRegistry(self.root / "datasets")
        self.experiment_dir = self.root / "experiments"
        self.experiment_dir.mkdir(exist_ok=True)
        if not self.goals():
            self.save_goals(GoalSpec())

    def goals(self):
        path = self.root / "goals.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    def save_goals(self, spec: GoalSpec):
        spec.validate()
        history = self.goals()
        revision = {"id": f"goals-{uuid4().hex[:12]}", "parent_id": history[-1]["id"] if history else None,
                    "created_at": datetime.now(timezone.utc).isoformat(), "spec": asdict(spec)}
        history.append(revision)
        destination = self.root / "goals.json"
        temporary = self.root / "goals.tmp"
        temporary.write_text(json.dumps(history, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(destination)
        return revision

    def goal(self, goal_id):
        for revision in self.goals():
            if revision["id"] == goal_id:
                return GoalSpec(**revision["spec"])
        raise ValueError("Unknown goal revision.")

    def records(self):
        records = [json.loads(path.read_text(encoding="utf-8")) for path in self.experiment_dir.glob("*/record.json")]
        return sorted(records, key=lambda record: record["created_at"])

    def record(self, experiment_id):
        for record in self.records():
            if record["id"] == experiment_id:
                return record
        raise ValueError("Choose a saved experiment.")

    def model(self, experiment_id):
        record = self.record(experiment_id)
        path = self.experiment_dir / record["id"] / "model.joblib"
        import hashlib
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["checkpoint_hash"]:
            raise ValueError("The local checkpoint changed on disk; it does not match its experiment.")
        # Only internally created local checkpoints are loaded; there is no pickle-upload interface.
        return joblib.load(path)

    def original_goal(self):
        records = self.records()
        goal_id = records[0]["config"]["goal_id"] if records else self.goals()[-1]["id"]
        return self.goal(goal_id)
