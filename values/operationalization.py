from data.datasets import GEO_FEATURES


def suggested_false_negative_cost(profile: dict) -> float:
    priorities = profile["priorities"]
    return round(max(0.1, (priorities["false_negatives"] + 1) /
                     (priorities["false_positives"] + 1)), 2)


def enforce_constraints(version, config, goal):
    if config.block_sensitive and set(version.features) & set(version.sensitive_features):
        raise ValueError("Hard constraint: remove sensitive inputs before running this experiment.")
    if config.mode != "train" and not config.parent_id:
        raise ValueError("A checkpoint intervention needs a saved parent experiment.")
    if config.model_id != "tiny_neural" and (config.edits or config.frozen_parameters or config.frozen_cells or config.reset_parameters):
        raise ValueError("Learned-weight editing and freezing currently require the small neural network.")


def values_trace(version, config, goal, results) -> list[dict]:
    priorities = goal.profile["priorities"]
    geo = sorted(set(version.features) & GEO_FEATURES)
    fnr = results["false_negative_rate"]
    observed_fnr = f"{fnr:.1%}" if fnr is not None else "no supporting examples"
    return [
        {"decision": "Privacy", "priority": priorities["privacy"],
         "technical_choice": f"Geographic inputs: {', '.join(geo) if geo else 'none'}",
         "evidence": "Goal failed" if goal.no_geography and geo else "Goal passed",
         "consequence": "Only selected features enter preprocessing/training; raw source versions remain local."},
        {"decision": "Missed alerts", "priority": priorities["false_negatives"],
         "technical_choice": f"Alert training penalty {config.false_negative_cost:g}x; inference threshold {config.threshold:g}",
         "evidence": f"False-negative rate {observed_fnr}",
         "consequence": "Penalty changes sample/loss weighting; threshold changes decisions without learning new weights."},
        {"decision": "Interpretability", "priority": priorities["interpretability"],
         "technical_choice": config.model_id,
         "evidence": "Coefficients, tree rules/importances, or all neural tensors are inspectable.",
         "consequence": "Accessible parameters do not guarantee that a model's behavior is easy to understand."},
        {"decision": "Compute / cost", "priority": priorities["low_compute"],
         "technical_choice": "CPU only, one training thread; no paid APIs",
         "evidence": f"{results['training_seconds']:.3f}s; observed process RAM {results['peak_ram_mb']:.1f} MB",
         "consequence": "Epochs, hidden units, tree count, and depth are explicit interventions. Electricity cost is not measured."},
        {"decision": "Local operation / community control", "priority": priorities["local_operation"],
         "technical_choice": "Local files, editable labels, local weights; no network model calls",
         "evidence": f"Dataset {version.id}, authored by {version.label_author}",
         "consequence": "Different profiles/label authors remain distinct; priorities are never averaged into a single ethics score."},
    ]
