import copy
import json
import re
from dataclasses import asdict
from pathlib import Path

import gradio as gr
import pandas as pd

from data.datasets import GEO_FEATURES
from experiments.runner import run_experiment
from experiments.schema import ExperimentConfig
from outputs.design_record import export_design_record
from values.operationalization import suggested_false_negative_cost
from values.schema import GoalSpec, PRIORITIES, ValueProfile


class LabController:
    def __init__(self, manager):
        self.manager = manager

    def session(self):
        revision = self.manager.goals()[-1]
        goal = self.manager.goal(revision["id"])
        versions = self.manager.datasets.all()
        choices = [(version.name + " / " + version.id, version.id) for version in versions]
        records = self.manager.records()
        selected = records[-1]["id"] if records else None
        form = [goal.project, goal.profile["name"], goal.profile["author"], goal.problem, goal.hypothesis]
        form += [goal.profile["priorities"][name] for name in PRIORITIES]
        form += [goal.min_accuracy * 100, goal.max_false_negative_rate * 100, goal.max_false_positive_rate * 100,
                 goal.no_geography, goal.require_local, goal.require_offline, goal.require_inspectable,
                 goal.max_ram_gb, json.dumps(goal.tests, indent=2)]
        status = f"Loaded `{revision['id']}`. Saved goals and experiments remain available."
        return ([revision["id"]] + form + [status,
                gr.update(choices=choices, value=versions[-1].id),
                gr.update(choices=choices, value=versions[0].id),
                gr.update(choices=choices, value=versions[-1].id)] + list(self.refresh_run_choices(selected)) + [selected])

    def save_goals(self, project, profile, author, problem, hypothesis, *settings):
        importance = settings[:len(PRIORITIES)]
        accuracy, fnr, fpr, no_geo, local, offline, inspectable, ram, tests = settings[len(PRIORITIES):]
        value_profile = ValueProfile(profile, author, {name: int(value) for name, value in zip(PRIORITIES, importance)})
        spec = GoalSpec(project=project, problem=problem, hypothesis=hypothesis, profile=asdict(value_profile),
                        min_accuracy=accuracy / 100, max_false_negative_rate=fnr / 100,
                        max_false_positive_rate=fpr / 100, no_geography=no_geo, require_local=local,
                        require_offline=offline, require_inspectable=inspectable, max_ram_gb=ram,
                        tests=json.loads(tests))
        revision = self.manager.save_goals(spec)
        return revision["id"], f"Saved `{revision['id']}`. Earlier goals and tests remain unchanged."

    def load_dataset(self, identifier):
        version = self.manager.datasets.get(identifier)
        frame = self.manager.datasets.training_frame(identifier)
        choices = [c for c in frame if c not in {"example_id", "included", "condition"}]
        summary = f"**{len(frame)} training candidates** / `{identifier}`. {len(self.manager.datasets.reference(version.test_fraction, version.split_seed))} original held-out examples."
        return (gr.update(value=frame, headers=list(frame.columns)), gr.update(choices=choices, value=version.features),
                gr.update(choices=choices, value=version.sensitive_features), version.test_fraction * 100,
                version.split_seed, version.changes, summary, {}, f"Dataset: `{identifier}`")

    def save_dataset(self, identifier, editor, features, sensitive, name, note, author, fraction, seed, new_defaults):
        full = self.manager.datasets.frame(identifier)
        editor = editor.copy()
        for feature, value in (new_defaults or {}).items():
            if feature not in full:
                full[feature] = value
        original_training_ids = self.manager.datasets.training_frame(identifier).example_id
        keep = full.loc[~full.example_id.isin(original_training_ids)]
        merged = pd.concat([keep, editor], ignore_index=True)
        version = self.manager.datasets.save(merged, features, sensitive, name, note, author, parent_id=identifier,
                                             test_fraction=fraction / 100, split_seed=int(seed))
        choices = [(v.name + " / " + v.id, v.id) for v in self.manager.datasets.all()]
        return gr.update(choices=choices, value=version.id), gr.update(choices=choices), gr.update(choices=choices, value=version.id)

    def add_example(self, editor, text):
        inputs = json.loads(text)
        if not isinstance(inputs, dict) or "condition" not in inputs:
            raise ValueError("New examples need a JSON object with a condition label.")
        unknown = set(inputs) - set(editor.columns)
        if unknown:
            raise ValueError("Add new feature columns before adding their values: " + ", ".join(sorted(unknown)))
        from experiments.runner import feature_defaults
        features = [c for c in editor if c not in {"example_id", "included", "condition"}]
        row = {**feature_defaults(editor, features), "example_id": "", "included": True, **inputs}
        return pd.concat([editor, pd.DataFrame([row])], ignore_index=True)

    def add_feature(self, editor, name, text, features, sensitive, defaults):
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,39}", name or "") or name in editor:
            raise ValueError("Use a new lowercase feature name with letters, numbers, and underscores.")
        value = json.loads(text)
        if value is None or not isinstance(value, (str, float, int, bool)):
            raise ValueError("Initial feature values must be text, a number, or true/false.")
        editor = editor.copy()
        editor[name] = value
        choices = [c for c in editor if c not in {"example_id", "included", "condition"}]
        return (gr.update(value=editor, headers=list(editor.columns)), gr.update(choices=choices, value=list(features) + [name]),
                gr.update(choices=choices, value=sensitive), {**defaults, name: value})

    def remove_geo(self, features):
        return [feature for feature in features if feature not in GEO_FEATURES], "Geographic inputs deselected. Save a dataset variant to record this decision."

    def exclude_unreliable(self, editor):
        editor = editor.copy()
        if "unreliable" not in editor:
            raise ValueError("This dataset has no unreliable-observation flag.")
        mask = pd.to_numeric(editor.unreliable, errors="raise") == 1
        editor.loc[mask, "included"] = False
        return editor, f"Excluded {int(mask.sum())} flagged training observations. Save a variant to preserve the change."

    def compare_datasets(self, left, right):
        a, b = self.manager.datasets.get(left), self.manager.datasets.get(right)
        af, bf = self.manager.datasets.frame(left).set_index("example_id"), self.manager.datasets.frame(right).set_index("example_id")
        common = af.index.intersection(bf.index)
        columns = af.columns.intersection(bf.columns)
        return {"dataset_A": left, "dataset_B": right, "added_examples": len(bf.index.difference(af.index)),
                "excluded_A": int((~af.included).sum()), "excluded_B": int((~bf.included).sum()),
                "changed_labels": int(af.loc[common, "condition"].ne(bf.loc[common, "condition"]).sum()),
                "edited_cells": int(af.loc[common, columns].fillna("<missing>").ne(bf.loc[common, columns].fillna("<missing>")).to_numpy().sum()),
                "removed_inputs": sorted(set(a.features) - set(b.features)), "added_inputs": sorted(set(b.features) - set(a.features)),
                "same_holdout": self.manager.datasets.evaluation_id(a) == self.manager.datasets.evaluation_id(b)}

    def apply_priorities(self, goal_id):
        cost = suggested_false_negative_cost(self.manager.goal(goal_id).profile)
        return cost, f"Applied priority ratio to the training objective: alert examples receive **{cost:g}x loss weight**."

    def train(self, dataset_id, goal_id, name, hypothesis, model, seed, positive_label, penalty, threshold,
              balance, regularization, depth, trees, hidden, epochs, learning_rate, block_sensitive, parent):
        config = ExperimentConfig(name=name, hypothesis=hypothesis, dataset_id=dataset_id, goal_id=goal_id,
                                  model_id=model, seed=int(seed), positive_label=positive_label.strip(),
                                  false_negative_cost=penalty, threshold=threshold, balance=balance,
                                  regularization=regularization, tree_depth=int(depth), trees=int(trees),
                                  hidden_units=int(hidden), epochs=int(epochs), learning_rate=learning_rate,
                                  block_sensitive=block_sensitive, parent_id=parent)
        record = run_experiment(self.manager, config)
        return record["id"], f"Saved **{name}** / `{record['id']}`. Accuracy {record['metrics']['accuracy']:.1%}; missed alerts {record['metrics']['false_negatives']}."

    def weight_branch(self, source, parameter, name, hypothesis, row, column, value, edit, freeze_cell, frozen, reset, mode, epochs, learning_rate):
        parent = self.manager.record(source)
        config = ExperimentConfig(**copy.deepcopy(parent["config"]))
        config.name, config.hypothesis, config.parent_id, config.mode = name, hypothesis, source, mode
        config.epochs, config.learning_rate = int(epochs), learning_rate
        config.edits = [{"parameter": parameter, "row": int(row), "column": int(column), "value": value}] if edit else []
        config.frozen_cells = [{"parameter": parameter, "row": int(row), "column": int(column)}] if freeze_cell else []
        config.frozen_parameters, config.reset_parameters = frozen, reset
        record = run_experiment(self.manager, config)
        return record["id"], f"Saved checkpoint branch `{record['id']}`; parent `{source}` remains unchanged."

    def threshold_branch(self, source, threshold):
        parent = self.manager.record(source)
        config = ExperimentConfig(**copy.deepcopy(parent["config"]))
        config.name = parent["config"]["name"] + f" / threshold {threshold:g}"
        config.hypothesis = "A different alert threshold changes false negatives without learning new weights."
        config.parent_id, config.mode, config.threshold = source, "evaluate_only", threshold
        config.edits, config.frozen_parameters, config.frozen_cells, config.reset_parameters = [], [], [], []
        record = run_experiment(self.manager, config)
        return record["id"], f"Saved threshold experiment `{record['id']}`. No model weights were retrained."

    def refresh_run_choices(self, selected):
        records = self.manager.records()
        choices = [(r["config"]["name"] + " / " + r["id"], r["id"]) for r in records]
        neural = [r for r in records if r["config"]["model_id"] == "tiny_neural"]
        neural_choices = [(r["config"]["name"] + " / " + r["id"], r["id"]) for r in neural]
        latest_neural = neural[-1]["id"] if neural else None
        return (gr.update(choices=choices, value=selected), gr.update(choices=choices, value=None),
                gr.update(choices=neural_choices, value=latest_neural), gr.update(choices=choices, value=[r["id"] for r in records]))

    def probe(self, identifier, text):
        record = self.manager.record(identifier)
        model = self.manager.model(identifier)
        inputs = json.loads(text)
        if not isinstance(inputs, dict):
            raise ValueError("An observation must be a JSON object.")
        example = model.example(inputs)
        probabilities = model.probabilities(example)[0]
        prediction = model.predict(example, record["config"]["positive_label"], record["config"]["threshold"])[0]
        result = {"experiment": identifier, "prediction": str(prediction),
                  "probabilities": {label: float(value) for label, value in zip(model.classes, probabilities)},
                  "used_features": model.features, "ignored_inputs": sorted(set(inputs) - set(model.features)),
                  "defaulted_inputs": sorted(set(model.features) - set(inputs)),
                  "test_type": "Post-training exploratory observation; not an original acceptance test"}
        path = self.manager.experiment_dir / identifier / "exploratory_tests.jsonl"
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps({"inputs": inputs, "result": result}) + "\n")
        return result

    def export_dataset(self, identifier):
        version = self.manager.datasets.get(identifier)
        self.manager.datasets.frame(identifier)
        return gr.update(value=str(self.manager.datasets.directory / version.id / "data.csv"), visible=True)

    def export(self):
        return gr.update(value=str(export_design_record(self.manager)), visible=True)
