from dataclasses import replace
import socket

import pytest

from playground.community import check_candidate, check_metrics, community_goals, comparison_configs, QUESTIONS
from playground.study_report import conclusion
from experiments.schema import ExperimentConfig
from scripts.community_study import block_network, run_study


def test_proposed_goals_are_explicit_and_location_is_optional():
    goal = community_goals()
    goal.validate()
    assert goal.problem == QUESTIONS["missed"]
    assert not goal.no_geography
    assert goal.max_ram_gb == 1.0


def test_pair_changes_only_training_importance_and_name():
    a, b = comparison_configs(ExperimentConfig(model_id="tiny_neural"), seed=101)
    assert a == replace(b, name=a.name, false_negative_cost=1.0)
    assert a.seed == b.seed == 101
    assert b.false_negative_cost == 5.0


def test_rate_targets_use_separate_denominators_and_missing_rates_fail():
    goal = community_goals()
    metrics = {"accuracy": 0.8, "false_negative_rate": 0.1, "false_positive_rate": 0.2}
    assert all(check_metrics(metrics, goal).values())
    assert not check_metrics(dict(metrics, false_negative_rate=None), goal)["Missed-alert rate"]
    assert not check_metrics(dict(metrics, false_positive_rate=0.201), goal)["False-alarm rate"]


def test_offline_guard_blocks_connections_and_restores_sockets():
    connect = socket.socket.connect
    with block_network() as attempts:
        with pytest.raises(RuntimeError, match="Network access is blocked"):
            socket.create_connection(("127.0.0.1", 1))
        with socket.socket() as client:
            with pytest.raises(RuntimeError, match="Network access is blocked"):
                client.connect(("127.0.0.1", 1))
        assert len(attempts) == 2
    assert socket.socket.connect is connect


def test_claimed_offline_and_resource_limits_are_not_automatic_passes():
    metrics = {"accuracy": 0.9, "false_negative_rate": 0.05, "false_positive_rate": 0.1,
               "peak_ram_mb": 1025, "training_seconds": 61, "inference_ms": 51, "model_bytes": 1048577}
    record = {"metrics": metrics, "runtime": {"device": "CPU"}, "final_parameters": {"weights": [1]}}
    checks = check_candidate(record, community_goals(), offline_verified=False)
    assert not checks["Offline training and inference"]
    assert not checks["Observed process RAM"]
    assert not checks["Training time"]
    assert not checks["Mean inference time"]
    assert not checks["Saved model size"]


def test_conclusion_does_not_hide_failed_seeds():
    assert "not consistently successful" in conclusion({"changed_runs_passing": 4, "runs": [None] * 5})


def test_study_preserves_protocol_and_generates_real_artifacts(tmp_path):
    workspace = tmp_path / "community"
    result = run_study(workspace)
    assert [row["seed"] for row in result["runs"]] == [42, 7, 23, 101, 202]
    assert result["test_readings"] == result["alert_examples"] + result["normal_examples"] == 144
    assert result["offline"]["attempts"] == 0
    assert result["offline"]["checkpoint_predictions_reproduced"]
    assert result["runs"][0]["original"]["final_parameters"] != result["runs"][0]["changed"]["final_parameters"]
    for name in ["protocol.json", "study-results.json", "community-study.html", "community-study.md"]:
        assert (workspace / name).exists()
    assert (workspace / "protocol.json").stat().st_mtime_ns <= (workspace / "study-results.json").stat().st_mtime_ns
    assert "data:image/png;base64," in (workspace / "community-study.html").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="fresh workspace"):
        run_study(workspace)
