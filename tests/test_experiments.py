from dataclasses import replace
import json
from zipfile import ZipFile

import numpy as np
import pytest

from experiments.comparison import compare_goals, prediction_changes
from experiments.runner import run_experiment
from outputs.design_record import export_design_record
from values.operationalization import suggested_false_negative_cost


@pytest.mark.parametrize("model", ["logistic", "tree", "forest", "tiny_neural"])
def test_all_models_train_and_restore(manager, config, model):
    record = run_experiment(manager, replace(config, model_id=model, epochs=12, trees=12))
    restored = manager.model(record["id"])
    assert record["metrics"]["test_examples"] == 144
    assert 0 <= record["metrics"]["accuracy"] <= 1
    assert record["metrics"]["model_bytes"] > 0
    assert len(record["custom_tests"]) == 3
    assert restored.features == record["features"]


def test_same_seed_reproduces_predictions_and_parameters(manager, config):
    a = run_experiment(manager, config)
    b = run_experiment(manager, replace(config, name="Replay"))
    assert a["predictions"] == b["predictions"]
    assert a["final_parameters"] == b["final_parameters"]


def test_penalty_changes_actual_learned_coefficients(manager, config):
    baseline = run_experiment(manager, config)
    weighted = run_experiment(manager, replace(config, false_negative_cost=5))
    assert not np.allclose(baseline["final_parameters"]["coefficients"], weighted["final_parameters"]["coefficients"])
    assert weighted["metrics"]["false_negatives"] < baseline["metrics"]["false_negatives"]
    assert prediction_changes(baseline, weighted)["changed_predictions"] > 0


def test_threshold_changes_outputs_without_changing_weights(manager, config):
    baseline = run_experiment(manager, config)
    branch = run_experiment(manager, replace(config, parent_id=baseline["id"], mode="evaluate_only", threshold=0.1))
    assert branch["final_parameters"] == baseline["final_parameters"]
    assert branch["metrics"]["training_seconds"] == 0
    assert prediction_changes(baseline, branch)["changed_predictions"] > 0


def test_goal_revisions_do_not_redefine_original_success(manager, config):
    baseline = run_experiment(manager, config)
    revised = replace(manager.goal(config.goal_id), min_accuracy=1, hypothesis="New deliberately strict goal")
    revision = manager.save_goals(revised)
    new_run = run_experiment(manager, replace(config, goal_id=revision["id"]))
    assert manager.record(baseline["id"])["goal_snapshot"]["min_accuracy"] == 0.8
    assert new_run["goal_snapshot"]["min_accuracy"] == 1
    assert manager.original_goal().min_accuracy == 0.8
    assert len(compare_goals([baseline, new_run], manager.original_goal())) == 16


def test_different_test_sets_are_not_called_comparable(manager, config):
    baseline = run_experiment(manager, config)
    original = manager.datasets.get(config.dataset_id)
    version = manager.datasets.save(manager.datasets.frame(original.id), original.features, [], "Split", "", "A", original.id, test_fraction=0.3)
    another = run_experiment(manager, replace(config, dataset_id=version.id))
    assert not prediction_changes(baseline, another)["comparable"]


def test_design_record_contains_lineage_goals_and_learned_state(manager, config):
    baseline = run_experiment(manager, config)
    path = export_design_record(manager)
    with ZipFile(path) as archive:
        assert set(archive.namelist()) == {"design_record.json", "design_record.md", "comparison.csv", "original_goal_assessment.csv"}
        data = json.loads(archive.read("design_record.json"))
    record = data["experiments"][0]
    assert record["id"] == baseline["id"]
    assert record["dataset"]["content_hash"]
    assert record["final_parameters"]["coefficients"]
    assert data["goal_history"]


def test_priorities_have_an_explicit_objective_mapping(manager):
    profile = manager.goal(manager.goals()[0]["id"]).profile
    assert suggested_false_negative_cost(profile) == 3


def test_duplicate_experiment_names_do_not_duplicate_exported_goal_results(manager, config):
    first = run_experiment(manager, config)
    second = run_experiment(manager, config)
    path = export_design_record(manager)
    with ZipFile(path) as archive:
        markdown = archive.read("design_record.md").decode()
    assert markdown.count("- Accuracy:") == 2
    comparison = compare_goals([first, second], manager.original_goal())
    assert set(comparison.id) == {first["id"], second["id"]}


@pytest.mark.parametrize("strategy", ["class_weight", "oversample"])
def test_rebalancing_is_a_recorded_training_intervention(manager, config, strategy):
    record = run_experiment(manager, replace(config, balance=strategy))
    assert record["objective"]["balance"] == strategy
    assert record["metrics"]["test_examples"] == 144
    if strategy == "oversample":
        assert record["train_examples"] > 576
