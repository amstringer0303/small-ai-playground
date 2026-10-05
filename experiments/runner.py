import copy
import hashlib
import importlib.metadata
import json
import platform
import time
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from evaluation.custom_tests import run_custom_tests
from evaluation.goals import assess_goals
from evaluation.metrics import evaluate_metrics, subgroup_metrics
from models.base import FittedModel, decisions, preprocessor
from models.neural_network import NeuralAdapter
from models.registry import get_model
from models.sklearn_models import SklearnAdapter
from models.weight_inspector import apply_interventions, parameter_change
from training.objectives import oversample, sample_weights
from training.trainer import ResourceMeasurement
from values.operationalization import enforce_constraints, values_trace


def feature_defaults(frame, features):
    result = {}
    for feature in features:
        column = frame[feature]
        if pd.api.types.is_numeric_dtype(column):
            result[feature] = float(column.median())
        else:
            result[feature] = str(column.mode().iloc[0])
    return result


def reference_for(manager, version):
    reference = manager.datasets.reference(version.test_fraction, version.split_seed)
    snapshot = manager.datasets.frame(version.id).set_index("example_id")
    for feature in version.features:
        if feature not in reference:
            reference[feature] = reference.example_id.map(snapshot[feature])
    return reference


def run_experiment(manager, config):
    config = copy.deepcopy(config)
    config.validate()
    version = manager.datasets.get(config.dataset_id)
    goal = manager.goal(config.goal_id)
    enforce_constraints(version, config, goal)
    reference = reference_for(manager, version)
    evaluation_id = manager.datasets.evaluation_id(version)
    training = manager.datasets.training_frame(version.id)
    training = training.loc[training.included].copy().reset_index(drop=True)
    if len(training) < 20 or training.condition.nunique() < 2:
        raise ValueError("Include at least 20 training examples across at least two categories.")
    if training.condition.nunique() > 8 or training.condition.value_counts().min() < 4:
        raise ValueError("Use at most 8 categories, with at least 4 included training examples per category.")
    classes = sorted(training.condition.unique().tolist())
    if config.positive_label not in classes:
        raise ValueError("The consequential/alert category must exist in the included training data.")
    missing_reference_categories = sorted(set(reference.condition) - set(classes))
    if missing_reference_categories:
        raise ValueError("Keep training examples for original evaluation categories: " + ", ".join(missing_reference_categories))
    parent = manager.record(config.parent_id) if config.parent_id else None
    if parent and config.mode == "evaluate_only" and parent["config"]["model_id"] != config.model_id:
        raise ValueError("Evaluate-only uses the parent's model family.")
    if parent and (config.mode == "evaluate_only" or config.model_id == "tiny_neural"):
        if parent["config"]["model_id"] != config.model_id:
            raise ValueError("Neural checkpoint continuation needs a neural parent.")
        if parent["features"] != version.features or parent["classes"] != classes:
            raise ValueError("A checkpoint branch needs the same features and categories. Start fresh after changing that schema.")
        if parent["evaluation_id"] != evaluation_id:
            raise ValueError("A checkpoint branch must use the parent's holdout; a new split could leak previously seen rows.")
    label_map = {label: index for index, label in enumerate(classes)}
    initial, starting, history = {}, {}, []
    captured_warnings = []
    with ResourceMeasurement() as resource:
        if config.model_id == "tiny_neural":
            fit_data, validation_data = train_test_split(training, test_size=0.2, random_state=config.seed,
                                                        stratify=training.condition)
        else:
            fit_data, validation_data = training, None
        if config.balance == "oversample" and config.mode == "train":
            fit_data = oversample(fit_data, "condition", config.seed)
        if parent and (config.mode == "evaluate_only" or config.model_id == "tiny_neural"):
            fitted = manager.model(parent["id"])
        else:
            preprocessing = preprocessor(fit_data[version.features])
            prepared = preprocessing.fit_transform(fit_data[version.features])
            adapter = (NeuralAdapter(prepared.shape[1], config.hidden_units, len(classes), config.seed)
                       if config.model_id == "tiny_neural" else SklearnAdapter(config.model_id, config))
            fitted = FittedModel(config.model_id, version.features, classes, preprocessing, adapter,
                                 feature_defaults(fit_data, version.features), preprocessing.get_feature_names_out().tolist())
        prepared = fitted.preprocessing.transform(fit_data[version.features])
        encoded_labels = np.array([label_map[label] for label in fit_data.condition])
        weights = sample_weights(fit_data.condition, config.positive_label, config.false_negative_cost, config.balance)
        if config.model_id == "tiny_neural":
            initial = fitted.adapter.parameters()
            handles = apply_interventions(fitted.adapter, config)
            starting = fitted.adapter.parameters()
        else:
            handles = []
        try:
            if config.mode == "train":
                validation = None
                if validation_data is not None:
                    validation = (fitted.preprocessing.transform(validation_data[version.features]),
                                  np.array([label_map[label] for label in validation_data.condition]))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    history = fitted.adapter.fit(prepared, encoded_labels, weights, config, validation)
                    captured_warnings = [str(warning.message) for warning in caught]
        finally:
            for handle in handles:
                handle.remove()
        final = fitted.adapter.parameters()
        if config.model_id != "tiny_neural":
            final["feature_names"] = fitted.encoded_features
            if "coefficients" in final:
                final["coefficient_targets"] = [classes[-1]] if len(classes) == 2 else classes
    probabilities = fitted.probabilities(reference)
    predicted = decisions(probabilities, classes, config.positive_label, config.threshold)
    metric_classes = sorted(set(classes) | set(reference.condition))
    metrics = evaluate_metrics(reference.condition, predicted, metric_classes, config.positive_label)
    latency_start = time.perf_counter()
    for _ in range(20):
        fitted.probabilities(reference.iloc[:1])
    metrics.update({"training_seconds": resource.seconds if config.mode == "train" else 0.0,
                    "inference_ms": (time.perf_counter() - latency_start) / 20 * 1000,
                    "peak_ram_mb": resource.peak_mb})
    if config.model_id == "tiny_neural":
        count = sum(p.numel() for p in fitted.adapter.network.parameters())
        trainable = sum(p.numel() for p in fitted.adapter.network.parameters() if p.requires_grad)
        for cell in config.frozen_cells:
            if cell["parameter"] not in config.frozen_parameters:
                trainable -= 1
    else:
        parameters = fitted.adapter.parameters()
        count = int(np.asarray(parameters["coefficients"]).size + len(parameters["bias"])) if "coefficients" in parameters else parameters["nodes"]
        trainable = count
    metrics["parameter_count"] = count
    metrics["parameter_count_kind"] = "tree nodes (structural size)" if config.model_id in {"tree", "forest"} else "scalar weights and biases"
    record = {
        "id": f"exp-{uuid4().hex[:12]}", "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_id": config.parent_id, "config": asdict(config), "dataset": asdict(version),
        "goal_snapshot": asdict(goal), "model": asdict(get_model(config.model_id)),
        "features": list(version.features), "encoded_features": fitted.encoded_features, "classes": classes,
        "evaluation_id": evaluation_id, "train_examples": len(fit_data),
        "validation_examples": len(validation_data) if validation_data is not None else 0,
        "objective": {"type": "weighted cross-entropy" if config.model_id in {"logistic", "tiny_neural"} else "weighted Gini impurity",
                      "consequential_label": config.positive_label, "false_negative_cost": config.false_negative_cost,
                      "balance": config.balance, "validation_loss": "unweighted cross-entropy (neural only)"},
        "trainable_parameters": trainable, "initial_parameters": initial, "starting_parameters": starting,
        "final_parameters": final, "parameter_changes": parameter_change(initial, final),
        "history": history, "warnings": captured_warnings,
        "runtime": {"device": "CPU", "python": platform.python_version(), "platform": platform.system(),
                    "versions": {package: importlib.metadata.version(package) for package in ["torch", "scikit-learn", "pandas", "gradio"]}},
        "metrics": metrics,
        "predictions": [{"example_id": identifier, "expected": expected, "predicted": prediction,
                         "alert_probability": float(probability[classes.index(config.positive_label)])}
                        for identifier, expected, prediction, probability in zip(reference.example_id, reference.condition, predicted, probabilities)],
        "subgroups": subgroup_metrics(reference, predicted, config.positive_label),
        "custom_tests": run_custom_tests(fitted, goal.tests, reference, config),
    }
    target = manager.experiment_dir / record["id"]
    target.mkdir()
    joblib.dump(fitted, target / "model.joblib", compress=3)
    checkpoint = (target / "model.joblib").read_bytes()
    metrics["model_bytes"] = len(checkpoint)
    record["checkpoint_hash"] = hashlib.sha256(checkpoint).hexdigest()
    record["goal_results"] = assess_goals(goal, record)
    record["values_trace"] = values_trace(version, config, goal, metrics)
    (target / "record.json").write_text(json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    return record
