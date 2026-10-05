from dataclasses import replace

from values.schema import GoalSpec


QUESTIONS = {
    "missed": "Can we catch more readings needing review without too many false alarms?",
    "flagged": "Should we exclude suspect sensor readings?",
    "labels": "What changes when residents correct training labels?",
    "location": "Can we remove location inputs without losing useful predictions?",
}
FOCUS = {"missed": "Missed reviews", "flagged": "Suspect readings",
         "labels": "Community corrections", "location": "Location inputs"}
SEEDS = (42, 7, 23, 101, 202)
LIMITS = {"training_seconds": 60.0, "inference_ms": 50.0, "model_bytes": 1024 * 1024}


def community_goals():
    return GoalSpec(
        project="Community sensor review: small / local AI",
        problem=QUESTIONS["missed"],
        hypothesis="Giving alert examples 5x loss weight reduces missed reviews without exceeding the false-alarm budget.",
        no_geography=False,
        max_ram_gb=1.0,
    )


def goal_text(goal):
    return (f"**Recorded prototype targets:** miss <= {goal.max_false_negative_rate:.0%} of alert examples; "
            f"falsely flag <= {goal.max_false_positive_rate:.0%} of normal examples; "
            f"accuracy >= {goal.min_accuracy:.0%}; local CPU; observed RAM <= {goal.max_ram_gb:g} GB. "
            "These are proposed targets, not community-approved standards.")


def check_metrics(metrics, goal):
    checks = {
        "Accuracy": metrics["accuracy"] >= goal.min_accuracy,
        "Missed-alert rate": metrics["false_negative_rate"] is not None
        and metrics["false_negative_rate"] <= goal.max_false_negative_rate,
        "False-alarm rate": metrics["false_positive_rate"] is not None
        and metrics["false_positive_rate"] <= goal.max_false_positive_rate,
    }
    return checks


def check_candidate(record, goal, offline_verified):
    checks = check_metrics(record["metrics"], goal)
    metrics = record["metrics"]
    checks.update({
        "Local CPU": record["runtime"]["device"] == "CPU",
        "Offline training and inference": bool(offline_verified),
        "Inspectable learned parameters": bool(record["final_parameters"]),
        "Observed process RAM": metrics["peak_ram_mb"] <= goal.max_ram_gb * 1024,
        "Training time": metrics["training_seconds"] <= LIMITS["training_seconds"],
        "Mean inference time": metrics["inference_ms"] <= LIMITS["inference_ms"],
        "Saved model size": metrics["model_bytes"] <= LIMITS["model_bytes"],
    })
    return checks


def comparison_configs(base, seed):
    original = replace(base, seed=seed, name=f"Original priorities / seed {seed}", false_negative_cost=1.0)
    changed = replace(base, seed=seed, name=f"Review-first priorities / seed {seed}", false_negative_cost=5.0)
    return original, changed
