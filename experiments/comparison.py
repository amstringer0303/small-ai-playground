import numpy as np
import pandas as pd

from evaluation.goals import assess_goals


def compare_records(records):
    rows = []
    for record in records:
        metrics, config = record["metrics"], record["config"]
        rows.append({"run": len(rows) + 1, "experiment": config["name"], "id": record["id"], "dataset": config["dataset_id"],
                     "model": config["model_id"], "test_cohort": record["evaluation_id"],
                     "accuracy": metrics["accuracy"], "precision": metrics["precision"], "recall": metrics["recall"],
                     "false_negatives": metrics["false_negatives"], "false_positives": metrics["false_positives"],
                     "FN_rate": metrics["false_negative_rate"], "training_s": metrics["training_seconds"],
                     "model_kb": metrics["model_bytes"] / 1024, "RAM_mb": metrics["peak_ram_mb"],
                     "penalty": config["false_negative_cost"], "threshold": config["threshold"]})
    return pd.DataFrame(rows)


def compare_goals(records, goal):
    rows = []
    for record in records:
        for result in assess_goals(goal, record):
            rows.append({"experiment": record["config"]["name"], "id": record["id"], "goal": result["goal"],
                         "criterion": result["criterion"], "observed": str(result["observed"]),
                         "status": "PASS" if result["passed"] else "FAIL"})
    return pd.DataFrame(rows)


def prediction_changes(left, right):
    if left["evaluation_id"] != right["evaluation_id"]:
        return {"comparable": False, "message": "Different held-out cohorts. Metrics are not a controlled comparison."}
    old = {p["example_id"]: p for p in left["predictions"]}
    new = {p["example_id"]: p for p in right["predictions"]}
    common = sorted(set(old) & set(new))
    changes = [{"example_id": identifier, "expected": old[identifier]["expected"],
                "before": old[identifier]["predicted"], "after": new[identifier]["predicted"],
                "probability_change": new[identifier]["alert_probability"] - old[identifier]["alert_probability"]}
               for identifier in common]
    return {"comparable": True, "changed_predictions": sum(row["before"] != row["after"] for row in changes),
            "mean_probability_shift": float(np.mean([abs(row["probability_change"]) for row in changes])),
            "rows": changes}
