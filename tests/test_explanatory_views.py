from dataclasses import replace

import pytest

from playground.community import community_goals, decision_text
from playground.service import Playground
from playground.views import count_change, interpretation, metric_table, target_evidence


def test_reading_corrections_are_real_training_edits_with_counts(tmp_path):
    playground = Playground(tmp_path / "corrections")
    original = playground.editor()
    identifier = original.iloc[0].Reading
    updated = playground.correct_reading(original, identifier, "alert", False)
    assert original.iloc[0].Label == "normal"
    assert original.iloc[0]["Use reading"]
    assert playground.reading(updated, identifier).condition == "alert"
    assert not playground.reading(updated, identifier).included
    assert playground.corrections(updated) == {"labels": 1, "excluded": 1, "changed": True}
    assert playground.corrections(original) == {"labels": 0, "excluded": 0, "changed": False}


def test_correction_form_cannot_edit_evaluation_readings(tmp_path):
    playground = Playground(tmp_path / "guarded-corrections")
    identifier = playground.manager.datasets.reference(0.2, 42).iloc[0].example_id
    with pytest.raises(ValueError, match="training reading"):
        playground.correct_reading(playground.editor(), identifier, "alert", True)
    identifier = playground.editor().iloc[0].Reading
    with pytest.raises(ValueError, match="normal or alert"):
        playground.correct_reading(playground.editor(), identifier, "critical", True)


@pytest.mark.parametrize("before,after,expected", [(8, 3, "5 fewer"), (5, 13, "8 more"), (4, 4, "No change")])
def test_counts_are_explained_in_words(before, after, expected):
    assert count_change(before, after) == expected


def records():
    a = {"metrics": {"accuracy": 131 / 144, "false_negatives": 8, "true_positives": 29,
                     "false_positives": 5, "true_negatives": 102,
                     "false_negative_rate": 8 / 37, "false_positive_rate": 5 / 107}}
    b = {"metrics": {"accuracy": 128 / 144, "false_negatives": 3, "true_positives": 34,
                     "false_positives": 13, "true_negatives": 94,
                     "false_negative_rate": 3 / 37, "false_positive_rate": 13 / 107}}
    return a, b


def test_comparison_uses_the_right_denominators_and_explains_tradeoffs():
    a, b = records()
    table = metric_table(a, b).set_index("Result")
    assert table.loc["Missed alerts", "Your choice"] == "3 of 37"
    assert table.loc["False alarms", "Your choice"] == "13 of 107"
    text = interpretation(a, b, community_goals())
    assert "5 fewer missed alerts; 8 more false alarms" in text
    assert "All three prediction targets" in text
    assert "not evidence of reliable real-world" in text


def test_status_respects_original_limits_and_missing_evidence():
    _, b = records()
    goal = replace(community_goals(), max_false_negative_rate=0.01)
    evidence = target_evidence(b, goal)
    assert "Not met" in evidence
    assert "at most 1%" in evidence
    b["metrics"]["false_negative_rate"] = None
    assert "No supporting examples" in target_evidence(b, goal)


def test_decision_definitions_distinguish_weights_and_data_changes():
    corrections = {"labels": 2, "excluded": 3}
    assert "equal importance" in decision_text("missed", 5, corrections)
    assert "5x" in decision_text("missed", 5, corrections)
    assert "evaluation readings" in decision_text("flagged", 5, corrections)
    assert "not data erasure" in decision_text("location", 5, corrections)
    assert "2 labels changed; 3 training readings excluded" in decision_text("labels", 5, corrections)


def test_opening_view_keeps_table_optional_and_correction_fields_hidden(tmp_path):
    from app import build_app
    app, _ = build_app(tmp_path / "beginner-ui")
    assert app.config["title"] == "Small AI Playground"
    assert any("<h1>Small AI Playground</h1>" in c["props"].get("value", "")
               for c in app.config["components"] if c["type"] == "html")
    props = {c["props"].get("elem_id"): c["props"] for c in app.config["components"]}
    assert not props["training-readings"]["open"]
    assert not props["reading-correction"]["visible"]
    assert props["simple-run"]["interactive"]
    assert props["reset-readings"]["visible"]
    assert not props["reset-readings"]["interactive"]
