import json
from zipfile import ZipFile

import numpy as np
import pytest

from data.datasets import GEO_FEATURES
from playground.service import Playground
from playground.views import pair_view


@pytest.fixture
def playground(tmp_path):
    return Playground(tmp_path / "study")


def test_default_data_and_goals(playground):
    assert len(playground.editor()) == 576
    assert len(playground.manager.datasets.reference(0.2, 42)) == 144
    assert not playground.manager.goal(playground.goal_id()).no_geography
    assert playground.pairs() == []


def test_empty_interface_disables_actions_without_records(tmp_path):
    from app import build_app
    app, _ = build_app(tmp_path / "empty-ui")
    buttons = {c["props"].get("value"): c["props"] for c in app.config["components"] if c["type"] == "button"}
    assert buttons["Open comparison"]["interactive"] is False
    assert buttons["Download experiment record"]["interactive"] is False


def test_refresh_saved_does_not_replace_the_open_experiment(tmp_path):
    from app import build_app
    app, _ = build_app(tmp_path / "refresh-ui")
    callback = next(fn for fn in app.fns.values() if fn.fn.__name__ == "refresh_saved")
    assert [type(component).__name__ for component in callback.outputs] == ["Dataframe", "Dropdown", "Button"]
    assert len(callback.fn()) == 3


def test_weighting_changes_real_weights_and_outputs(playground):
    pair = playground.run(penalty=5)
    original, changed = playground.records(pair)
    assert original["evaluation_id"] == changed["evaluation_id"]
    assert original["features"] == changed["features"]
    assert original["config"]["seed"] == changed["config"]["seed"] == 42
    assert original["config"]["false_negative_cost"] == 1
    assert changed["config"]["false_negative_cost"] == 5
    assert not np.allclose(original["final_parameters"]["hidden.weight"], changed["final_parameters"]["hidden.weight"])
    assert original["predictions"] != changed["predictions"]
    assert len(pair_view(playground, pair["id"])) == 13


def test_no_intervention_is_a_reproducible_control(playground):
    pair = playground.run(penalty=1)
    original, changed = playground.records(pair)
    assert original["predictions"] == changed["predictions"]
    assert original["final_parameters"] == changed["final_parameters"]
    assert "No classifications changed" in pair_view(playground, pair["id"])[0]


def test_baseline_reused_and_goals_stay_frozen(playground):
    first = playground.run(penalty=3)
    before = playground.manager.goal(first["goal_id"])
    before.min_accuracy = 0.99
    playground.manager.save_goals(before)
    second = playground.run(penalty=5)
    assert second["baseline_reused"]
    assert second["original_id"] == first["original_id"]
    assert second["goal_id"] == first["goal_id"]
    assert second["number"] == 2
    restored = Playground(playground.manager.root)
    assert len(restored.pairs()) == 2
    assert restored.goal_id() == first["goal_id"]


@pytest.mark.parametrize("choice", ["flagged", "location"])
def test_data_interventions_preserve_holdout(playground, choice):
    pair = playground.run(choice)
    original, changed = playground.records(pair)
    assert original["evaluation_id"] == changed["evaluation_id"]
    assert [p["expected"] for p in original["predictions"]] == [p["expected"] for p in changed["predictions"]]
    version = playground.manager.datasets.get(changed["config"]["dataset_id"])
    if choice == "location":
        assert not set(changed["features"]) & GEO_FEATURES
        assert version.changes["changed_labels"] == 0
    else:
        frame = playground.manager.datasets.training_frame(version.id)
        assert not frame.loc[frame.unreliable == 1, "included"].any()
        assert changed["features"] == original["features"]


def test_label_edits_saved_without_changing_source(playground):
    editor = playground.editor()
    row = editor.index[editor.Label == "normal"][0]
    reading = editor.at[row, "Reading"]
    editor.at[row, "Label"] = "alert"
    pair = playground.run("labels", editor=editor)
    _, changed = playground.records(pair)
    variant = playground.manager.datasets.get(changed["config"]["dataset_id"])
    assert variant.changes["changed_labels"] == 1
    assert playground.manager.datasets.frame(playground.original.id).set_index("example_id").at[reading, "condition"] == "normal"
    assert playground.manager.datasets.frame(variant.id).set_index("example_id").at[reading, "condition"] == "alert"


def test_deleted_training_row_is_excluded(playground):
    editor = playground.editor()
    identifier = editor.iloc[0].Reading
    updated = playground.edited_frame(editor.iloc[1:].copy())
    assert not updated.set_index("example_id").at[identifier, "included"]
    assert len(updated) == 720


def test_nonselected_edits_are_not_silently_ignored(playground):
    editor = playground.editor()
    editor.at[0, "PM2.5"] += 10
    with pytest.raises(ValueError, match="Only labels and Use reading"):
        playground.run("missed", editor=editor)
    assert not playground.manager.records()


def test_label_edits_cannot_be_hidden_by_another_choice(playground):
    editor = playground.editor()
    editor.at[0, "Label"] = "alert" if editor.at[0, "Label"] == "normal" else "normal"
    with pytest.raises(ValueError, match="You have edited readings"):
        playground.run("missed", editor=editor)
    assert not playground.manager.records()


def test_reading_ids_and_labels_are_guarded(playground):
    editor = playground.editor()
    editor.at[0, "Reading"] = "unknown-reading"
    with pytest.raises(ValueError, match="Reading IDs are fixed"):
        playground.edited_frame(editor)
    editor = playground.editor()
    editor.at[0, "Label"] = "critical"
    with pytest.raises(ValueError, match="normal or alert"):
        playground.edited_frame(editor)
    with pytest.raises(ValueError, match="Change a label"):
        playground.run("labels")


def test_export_contains_parameters_and_data_lineage(playground):
    pair = playground.run(penalty=5)
    path = playground.export(pair["id"])
    with ZipFile(path) as archive:
        payload = json.loads(archive.read("experiment.json"))
        assert payload["comparison"]["id"] == pair["id"]
        assert payload["your_choice"]["final_parameters"]["hidden.weight"]
        assert payload["your_choice"]["dataset"]["content_hash"]
        assert {"summary.md", "experiment.json", "metrics.csv", "original_goals.csv"} == set(archive.namelist())


@pytest.mark.parametrize("choice,penalty", [("unknown", 5), ("missed", 0), ("missed", 11)])
def test_invalid_interventions_do_not_train(playground, choice, penalty):
    with pytest.raises(ValueError):
        playground.run(choice, penalty)
    assert not playground.manager.records()
