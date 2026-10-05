from dataclasses import replace

import pandas as pd
import pytest

from data.datasets import GEO_FEATURES
from data.versions import normalize
from experiments.runner import run_experiment
from ui.controller import LabController


def test_location_variant_is_immutable_and_same_holdout(manager):
    original = manager.datasets.all()[0]
    frame = manager.datasets.frame(original.id)
    variant = manager.datasets.save(frame, [f for f in original.features if f not in GEO_FEATURES],
                                     original.sensitive_features, "No geography", "Privacy intervention", "Group A", original.id)
    assert variant.parent_id == original.id
    assert set(variant.changes["removed_features"]) == GEO_FEATURES
    assert manager.datasets.evaluation_id(variant) == manager.datasets.evaluation_id(original)
    assert manager.datasets.get(original.id).features == original.features


def test_holdout_cannot_be_relabeled(manager):
    original = manager.datasets.all()[0]
    frame = manager.datasets.frame(original.id)
    identifier = manager.datasets.reference(0.2, 42).example_id.iloc[0]
    frame.loc[frame.example_id == identifier, "condition"] = "community_alert"
    with pytest.raises(ValueError, match="holdout"):
        manager.datasets.save(frame, original.features, [], "Changed answers", "", "A", original.id)


def test_training_label_edits_and_new_examples_are_preserved(manager):
    original = manager.datasets.all()[0]
    training = manager.datasets.training_frame(original.id)
    chosen = training.example_id.head(4)
    frame = manager.datasets.frame(original.id)
    frame.loc[frame.example_id.isin(chosen), "condition"] = "community_alert"
    row = training.iloc[[0]].copy()
    row.example_id = "new"
    row.condition = "community_alert"
    frame = pd.concat([frame, row], ignore_index=True)
    variant = manager.datasets.save(frame, original.features, [], "Community labels", "Group A relabeled", "Group A", original.id)
    assert variant.changes["changed_labels"] == 4
    assert variant.changes["added_examples"] == 1
    assert "community_alert" in set(manager.datasets.frame(variant.id).condition)
    assert len(manager.datasets.frame(variant.id)) == 721


def test_snapshot_tampering_is_detected(manager):
    original = manager.datasets.all()[0]
    path = manager.datasets.directory / original.id / "data.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "pm25"] += 100
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="modified on disk"):
        manager.datasets.frame(original.id)


def test_sensitive_inputs_hard_constraint_blocks_training(manager, config):
    with pytest.raises(ValueError, match="Hard constraint"):
        run_experiment(manager, replace(config, block_sensitive=True))
    assert not manager.records()


def test_added_feature_survives_csv_round_trip_and_can_train(manager, config):
    controller = LabController(manager)
    original = manager.datasets.all()[0]
    editor = manager.datasets.training_frame(original.id)
    result = controller.add_feature(editor, "community_rating", "0.12345678912345", original.features,
                                     original.sensitive_features, {})
    frame = manager.datasets.frame(original.id)
    frame["community_rating"] = 0.12345678912345
    frame.loc[frame.example_id.isin(editor.example_id), "community_rating"] = result[0]["value"].community_rating.to_numpy()
    variant = manager.datasets.save(frame, original.features + ["community_rating"], [], "Extra feature", "", "A", original.id)
    assert "community_rating" in manager.datasets.frame(variant.id)
    record = run_experiment(manager, replace(config, dataset_id=variant.id))
    assert record["metrics"]["test_examples"] == 144


def test_numeric_training_editor_strings_normalize():
    frame = pd.DataFrame({"example_id": ["x"], "included": ["true"], "condition": ["alert"],
                          "pm25": ["12.5"], "community_rating": ["4"]})
    result = normalize(frame)
    assert result.pm25.iloc[0] == 12.5
    assert result.community_rating.iloc[0] == 4


def test_removing_a_training_row_is_a_saved_data_intervention(manager):
    controller = LabController(manager)
    original = manager.datasets.all()[0]
    training = manager.datasets.training_frame(original.id)
    removed_id = training.example_id.iloc[0]
    result = controller.save_dataset(original.id, training.iloc[1:], original.features,
                                     original.sensitive_features, "Removed example", "Participant removed misleading data",
                                     "Group A", 20, 42, {})
    version = manager.datasets.get(result[0]["value"])
    assert removed_id not in set(manager.datasets.frame(version.id).example_id)
    assert version.changes["removed_examples"] == 1
    assert len(manager.datasets.frame(original.id)) == 720
    assert len(manager.datasets.reference(version.test_fraction, version.split_seed)) == 144


def test_split_change_is_recorded(manager):
    original = manager.datasets.all()[0]
    variant = manager.datasets.save(manager.datasets.frame(original.id), original.features, [],
                                     "Different holdout", "Split intervention", "A", original.id, test_fraction=0.3)
    assert variant.changes["split_changed"]
    assert manager.datasets.evaluation_id(variant) != manager.datasets.evaluation_id(original)
