from data.datasets import GEO_FEATURES


def assess_goals(goal, record):
    metrics = record["metrics"]
    inputs = record["features"]
    criteria = [
        ("Accuracy", f">= {goal.min_accuracy:.0%}", metrics["accuracy"], metrics["accuracy"] >= goal.min_accuracy),
        ("False-negative rate", f"<= {goal.max_false_negative_rate:.0%}", metrics["false_negative_rate"],
         metrics["false_negative_rate"] is not None and metrics["false_negative_rate"] <= goal.max_false_negative_rate),
        ("False-positive rate", f"<= {goal.max_false_positive_rate:.0%}", metrics["false_positive_rate"],
         metrics["false_positive_rate"] is not None and metrics["false_positive_rate"] <= goal.max_false_positive_rate),
        ("No geographic inputs", "No coordinates, site, or sensor" if goal.no_geography else "Not required",
         ", ".join(sorted(set(inputs) & GEO_FEATURES)) or "None",
         not goal.no_geography or not set(inputs) & GEO_FEATURES),
        ("Local operation", "Local" if goal.require_local else "Not required", "Local CPU", True),
        ("Offline inference", "Offline" if goal.require_offline else "Not required", "No network calls", True),
        ("Inspectable parameters", "Inspectable" if goal.require_inspectable else "Not required", "Accessible", True),
        ("Observed process RAM", f"<= {goal.max_ram_gb:g} GB", f"{metrics['peak_ram_mb'] / 1024:.3f} GB",
         metrics["peak_ram_mb"] <= goal.max_ram_gb * 1024),
    ]
    return [{"goal": name, "criterion": criterion, "observed": value, "passed": bool(passed)}
            for name, criterion, value, passed in criteria]
