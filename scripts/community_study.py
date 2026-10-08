"""Run a prespecified, synthetic community-question study in a fresh workspace."""
import argparse
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from data.datasets import ROOT
from evaluation.metrics import evaluate_metrics
from experiments.runner import run_experiment
from experiments.schema import ExperimentConfig
from playground.community import LIMITS, QUESTIONS, SEEDS, check_candidate, check_metrics, comparison_configs
from playground.service import Playground
from playground.study_report import write_report


@contextmanager
def block_network():
    attempts = []

    def denied(*args, **kwargs):
        attempts.append("Attempted Python socket network operation")
        raise RuntimeError("Network access is blocked for this local-model study.")

    with ExitStack() as stack:
        for target in ["socket.socket.connect", "socket.socket.connect_ex", "socket.socket.sendto",
                       "socket.create_connection", "socket.getaddrinfo"]:
            stack.enter_context(patch(target, denied))
        yield attempts


def run_study(workspace):
    workspace = Path(workspace).resolve()
    if (workspace / "protocol.json").exists() or (workspace / "goals.json").exists():
        raise ValueError("Use a fresh workspace; existing study criteria and results will not be overwritten.")
    playground = Playground(workspace)
    goal_id = playground.goal_id()
    goal = playground.manager.goal(goal_id)
    base = ExperimentConfig(dataset_id=playground.original.id, goal_id=goal_id, model_id="tiny_neural",
                            name="Original priorities", hypothesis=goal.hypothesis,
                            epochs=60, hidden_units=8, learning_rate=0.02)
    protocol = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "question": QUESTIONS["missed"], "potential_questions": QUESTIONS,
        "additional_question": "Can it run on an ordinary laptop without internet or a paid AI service?",
        "criteria_status": "Playground criteria; not approved by a community",
        "goal_id": goal_id, "goals": asdict(goal), "resource_limits": LIMITS,
        "source": "Synthetic: 720 invented readings, generation seed 23; not health or regulatory labels",
        "dataset_id": playground.original.id, "dataset_hash": playground.original.content_hash,
        "test_fraction": 0.2, "test_split_seed": 42, "training_seeds": list(SEEDS),
        "original_config": asdict(base), "intervention": "Alert-example training loss weight 1x -> 5x",
        "comparison_rule": "Keep features, architecture, threshold, split and epochs unchanged within each seed pair",
        "robustness_criterion": "All five changed-model runs must meet every recorded performance and resource target",
        "no_ai_rule": "Flag when simulated PM2.5 >= 42. Arbitrary demonstration rule, not an EPA or health threshold",
        "offline_check": "Python socket connections, UDP sends and DNS blocked during training, inference and checkpoint loading",
    }
    (workspace / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    runs = []
    with block_network() as attempts:
        reference = playground.manager.datasets.reference(0.2, 42)
        rule_metrics = evaluate_metrics(reference.condition, np.where(reference.pm25 >= 42, "alert", "normal"),
                                        ["alert", "normal"], "alert")
        rule = {"name": "Non-AI demonstration rule", "metrics": rule_metrics,
                "checks": check_metrics(rule_metrics, goal)}
        logistic = run_experiment(playground.manager, ExperimentConfig(
            dataset_id=base.dataset_id, goal_id=goal_id, model_id="logistic", seed=42,
            name="Simpler learned benchmark", hypothesis="Check whether a linear classifier is already sufficient."))
        pair = playground.run("missed", 5)
        original, changed = playground.records(pair)
        runs.append({"seed": 42, "original": original, "changed": changed})
        for seed in SEEDS[1:]:
            a_config, b_config = comparison_configs(base, seed)
            a = run_experiment(playground.manager, a_config)
            b = run_experiment(playground.manager, b_config)
            runs.append({"seed": seed, "original": a, "changed": b})
            print(f"Seed {seed}: missed {a['metrics']['false_negatives']} -> {b['metrics']['false_negatives']}; "
                  f"false alarms {a['metrics']['false_positives']} -> {b['metrics']['false_positives']}", flush=True)
        all_records = [logistic] + [row[variant] for row in runs for variant in ["original", "changed"]]
        for record in all_records:
            assert record["evaluation_id"] == original["evaluation_id"]
            assert [p["example_id"] for p in record["predictions"]] == reference.example_id.tolist()
            assert [p["expected"] for p in record["predictions"]] == reference.condition.tolist()
            fitted = playground.manager.model(record["id"])
            reproduced = fitted.probabilities(reference)[:, fitted.classes.index("alert")]
            np.testing.assert_allclose(reproduced, [p["alert_probability"] for p in record["predictions"]])
            if record["config"]["model_id"] == "tiny_neural":
                assert all(parameter.device.type == "cpu" for parameter in fitted.adapter.network.parameters())
        assert not attempts
    for row in runs:
        for variant in ["original", "changed"]:
            row[variant + "_checks"] = check_candidate(row[variant], goal, offline_verified=True)
    result = {
        "protocol": protocol, "evaluation_id": original["evaluation_id"], "comparison_id": pair["id"],
        "test_readings": len(reference), "alert_examples": int(reference.condition.eq("alert").sum()),
        "normal_examples": int(reference.condition.eq("normal").sum()), "rule": rule,
        "logistic": logistic, "logistic_checks": check_candidate(logistic, goal, True), "runs": runs,
        "offline": {"blocked_python_socket_operations": True, "attempts": len(attempts),
                    "checkpoint_predictions_reproduced": True},
        "changed_runs_passing": sum(all(row["changed_checks"].values()) for row in runs),
        "original_runs_passing": sum(all(row["original_checks"].values()) for row in runs),
        "limitations": [
            "Invented data and labels; no real residents participated or approved these targets.",
            "The development evaluation set was already visible in this prototype. Repeated seeds are not new test populations.",
            "There are only 37 alert examples. These checks use point estimates, not statistical guarantees about an underlying population.",
            "No claim of health protection, regulatory suitability, subgroup fairness, or deployment readiness.",
            "A small saved model is not a small installation. Python, PyTorch and the app have additional storage/RAM overhead.",
            "RAM is sampled process RSS; latency is a mean, not a worst-case guarantee. Results apply to this computer.",
            "The offline guard covers Python socket operations in this process, not every possible native-library network path or the whole device.",
            "Location inputs remain in this case. Local computation is not anonymization or a complete privacy/governance assessment.",
        ],
    }
    (workspace / "study-results.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    write_report(result, workspace)
    print(json.dumps({"question": protocol["question"], "changed_runs_passing": result["changed_runs_passing"],
                      "original_runs_passing": result["original_runs_passing"],
                      "report": str(workspace / "community-study.html")}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=ROOT / "outputs" / "community-case")
    args = parser.parse_args()
    run_study(args.workspace)
