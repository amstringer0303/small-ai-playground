from dataclasses import replace

import numpy as np
import pytest

from experiments.comparison import prediction_changes
from experiments.runner import run_experiment


@pytest.fixture
def neural(manager, config):
    return run_experiment(manager, replace(config, model_id="tiny_neural", epochs=25))


def test_manual_weight_intervention_changes_outputs_and_preserves_parent(manager, config, neural):
    edit = {"parameter": "output.bias", "row": 0, "column": 0, "value": 100.0}
    branch = run_experiment(manager, replace(config, model_id="tiny_neural", parent_id=neural["id"],
                                            mode="evaluate_only", edits=[edit]))
    assert branch["final_parameters"]["output.bias"][0] == 100
    assert prediction_changes(neural, branch)["changed_predictions"] > 0
    assert manager.record(neural["id"])["final_parameters"] == neural["final_parameters"]
    assert manager.model(neural["id"]).adapter.parameters() == neural["final_parameters"]


def test_freeze_layer_and_retrain(manager, config, neural):
    branch = run_experiment(manager, replace(config, model_id="tiny_neural", parent_id=neural["id"], epochs=15,
                                            frozen_parameters=["hidden.weight", "hidden.bias"]))
    for parameter in ["hidden.weight", "hidden.bias"]:
        assert branch["final_parameters"][parameter] == neural["final_parameters"][parameter]
    assert not np.allclose(branch["final_parameters"]["output.weight"], neural["final_parameters"]["output.weight"])
    assert branch["trainable_parameters"] < branch["metrics"]["parameter_count"]


def test_freeze_individual_connection_and_retrain(manager, config, neural):
    cell = {"parameter": "hidden.weight", "row": 0, "column": 0}
    branch = run_experiment(manager, replace(config, model_id="tiny_neural", parent_id=neural["id"], epochs=10, frozen_cells=[cell]))
    assert branch["final_parameters"]["hidden.weight"][0][0] == neural["final_parameters"]["hidden.weight"][0][0]
    assert branch["metrics"]["parameter_count"] - branch["trainable_parameters"] == 1
    assert manager.model(branch["id"]).adapter.parameters() == branch["final_parameters"]


def test_reset_selected_parameter_is_reproducible(manager, config, neural):
    initial = manager.model(neural["id"]).adapter.reset_state
    branch = run_experiment(manager, replace(config, model_id="tiny_neural", parent_id=neural["id"],
                                            mode="evaluate_only", reset_parameters=["hidden.weight"]))
    assert branch["final_parameters"]["hidden.weight"] == initial["hidden.weight"]
    assert branch["config"]["reset_parameters"] == ["hidden.weight"]


def test_invalid_weight_index_is_rejected(manager, config, neural):
    with pytest.raises(ValueError, match="outside"):
        run_experiment(manager, replace(config, model_id="tiny_neural", parent_id=neural["id"], mode="evaluate_only",
                                        edits=[{"parameter": "hidden.weight", "row": 999, "column": 0, "value": 1}]))


def test_warm_start_cannot_leak_across_changed_holdout(manager, config, neural):
    original = manager.datasets.get(config.dataset_id)
    version = manager.datasets.save(manager.datasets.frame(original.id), original.features, [], "Different split", "", "A", original.id, test_fraction=0.3)
    with pytest.raises(ValueError, match="holdout"):
        run_experiment(manager, replace(config, model_id="tiny_neural", dataset_id=version.id, parent_id=neural["id"]))


def test_neural_seed_is_reproducible(manager, config, neural):
    replay = run_experiment(manager, replace(config, model_id="tiny_neural", epochs=25))
    assert replay["final_parameters"] == neural["final_parameters"]
    assert replay["predictions"] == neural["predictions"]
