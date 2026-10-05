import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from experiments.comparison import compare_goals, compare_records, prediction_changes
from models.weight_inspector import parameter_change, weight_table

COLORS = ["#17785a", "#b13b61", "#2b75a0", "#a07715"]


def base_figure(width=8, height=4):
    figure = Figure(figsize=(width, height), layout="constrained", facecolor="#ffffff")
    return figure


def confusion_plot(record):
    figure = base_figure(5, 3.6)
    axis = figure.subplots()
    matrix = np.asarray(record["metrics"]["confusion_matrix"])
    axis.imshow(matrix, cmap="GnBu")
    for index in np.ndindex(matrix.shape):
        axis.text(index[1], index[0], str(matrix[index]), ha="center", va="center",
                  color="white" if matrix[index] > matrix.max() * 0.55 else "#202a31", fontsize=13)
    labels = record["metrics"]["classes"]
    axis.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels,
             yticklabels=labels, xlabel="Predicted", ylabel="Original label")
    return figure


def training_plot(record):
    history = record["history"]
    if not history:
        return None
    frame = pd.DataFrame(history)
    figure = base_figure(10, 5.3)
    axes = figure.subplots(2, 2)
    axes[0, 0].plot(frame.epoch, frame.loss, color=COLORS[0], label="Weighted training loss")
    axes[0, 0].plot(frame.epoch, frame.validation_loss, color=COLORS[1], label="Unweighted validation loss")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title("Loss")
    axes[0, 1].plot(frame.epoch, frame.gradient_mean, color=COLORS[2])
    axes[0, 1].set_title("Mean absolute gradient")
    axes[1, 0].plot(frame.epoch, frame.weight_mean, color=COLORS[0])
    axes[1, 0].set_title("Weight mean over time")
    if "weight_histogram" in frame:
        edges = history[0]["histogram_edges"]
        axes[1, 1].imshow(np.asarray(frame.weight_histogram.tolist()).T, origin="lower", aspect="auto", cmap="GnBu",
                          extent=(0.5, len(frame) + 0.5, edges[0], edges[-1]))
        axes[1, 1].set_title("Weight distribution over time")
        axes[1, 1].set_ylabel("Parameter value")
    else:
        axes[1, 1].plot(frame.epoch, frame.weight_std, color=COLORS[1])
        axes[1, 1].set_title("Weight standard deviation over time")
    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    return figure


def evaluation_view(record):
    if not record:
        return "Run an experiment to see held-out evidence.", pd.DataFrame(), None, None, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
    metrics = record["metrics"]
    rows = []
    for key in ["accuracy", "precision", "recall", "false_negative_rate", "false_positive_rate", "false_negatives",
                "false_positives", "training_seconds", "inference_ms", "model_bytes", "peak_ram_mb", "parameter_count"]:
        value = metrics[key]
        if key in {"accuracy", "precision", "recall", "false_negative_rate", "false_positive_rate"}:
            value = f"{value:.1%}" if value is not None else "No supporting examples"
        elif isinstance(value, float):
            value = f"{value:.3f}"
        label = "tree node count" if key == "parameter_count" and record["config"]["model_id"] in {"tree", "forest"} else key.replace("_", " ")
        rows.append({"metric": label, "value": str(value)})
    goals = pd.DataFrame([{**r, "observed": str(r["observed"]), "status": "PASS" if r["passed"] else "FAIL"}
                          for r in record["goal_results"]]).drop(columns="passed")
    tests = pd.DataFrame([{**t, "status": "NO EVIDENCE" if t["passed"] is None else "PASS" if t["passed"] else "FAIL"}
                          for t in record["custom_tests"]])
    if "passed" in tests:
        tests = tests.drop(columns="passed")
    summary = f"**{record['config']['name']}** / `{record['id']}`. Dataset `{record['dataset']['id']}`; {metrics['test_examples']} held-out examples."
    if record["warnings"]:
        summary += "\n\nTraining notes: " + "; ".join(record["warnings"])
    return summary, pd.DataFrame(rows), confusion_plot(record), training_plot(record), goals, tests, pd.DataFrame(record["subgroups"]), record["final_parameters"]


def weight_view(record, parameter):
    if not record or record["config"]["model_id"] != "tiny_neural":
        return "Train a small PyTorch network first.", None, pd.DataFrame(), None
    initial, final = record["initial_parameters"], record["final_parameters"]
    table = weight_table(initial, final, parameter)
    start, finish = np.asarray(initial[parameter]), np.asarray(final[parameter])
    if start.ndim == 1:
        start, finish = start.reshape(-1, 1), finish.reshape(-1, 1)
    figure = base_figure(11, 3.5)
    axes = figure.subplots(1, 3)
    limit = max(float(np.abs(start).max()), float(np.abs(finish).max()), 0.01)
    for axis, data, title in zip(axes, [start, finish, finish - start], ["Before intervention / training", "After", "Difference"]):
        bound = max(float(np.abs(data).max()), 0.01) if title == "Difference" else limit
        image = axis.imshow(data, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
        axis.set(title=title, xlabel="Input index", ylabel="Output index")
        figure.colorbar(image, ax=axis, shrink=0.7)
    histogram = base_figure(8, 3)
    axis = histogram.subplots()
    before = np.concatenate([np.asarray(tensor).flatten() for tensor in initial.values()])
    after = np.concatenate([np.asarray(tensor).flatten() for tensor in final.values()])
    bins = np.linspace(min(before.min(), after.min()), max(before.max(), after.max()) + 1e-6, 25)
    axis.hist(before, bins=bins, alpha=0.6, color=COLORS[2], label="Before")
    axis.hist(after, bins=bins, alpha=0.6, color=COLORS[1], label="After")
    axis.set(xlabel="Learned parameter value", ylabel="Connections")
    axis.legend()
    summary = f"**{record['metrics']['parameter_count']} parameters**, {record['trainable_parameters']} trainable in this run."
    summary += f" `{parameter}` shape: {np.asarray(final[parameter]).shape}."
    if parameter == "hidden.weight":
        summary += " Input columns: " + ", ".join(f"{i}: {feature}" for i, feature in enumerate(record["encoded_features"]))
    elif parameter.startswith("output"):
        summary += " Output rows: " + ", ".join(f"{i}: {label}" for i, label in enumerate(record["classes"]))
    return summary, figure, table.round(6), histogram


def comparison_view(manager, identifiers):
    records = [manager.record(identifier) for identifier in identifiers or []]
    if not records:
        return "Choose saved experiments to compare.", pd.DataFrame(), None, pd.DataFrame(), pd.DataFrame()
    table = compare_records(records).round(4)
    same_cohort = len({r["evaluation_id"] for r in records}) == 1
    summary = "Same held-out cohort." if same_cohort else "**Different held-out cohorts: this is not a controlled comparison.**"
    summary += " All goal assessments use the original goals from the first saved experiment."
    figure = base_figure(11, 3.4)
    axes = figure.subplots(1, 3)
    ticks = list(range(0, len(records), max(1, int(np.ceil(len(records) / 10)))))
    labels = [str(index + 1) for index in ticks]
    for axis, metric, title, color in zip(axes, ["accuracy", "recall", "false_negative_rate"],
                                         ["Accuracy", "Alert recall", "Missed-alert rate"], COLORS):
        values = [record["metrics"][metric] for record in records]
        axis.plot(range(len(values)), values, "o-", color=color, markersize=6)
        axis.set(xticks=ticks, xticklabels=labels, ylim=(0, 1.03), title=title, xlabel="Run number")
        axis.tick_params(axis="x", labelsize=9)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    changes = pd.DataFrame()
    if len(records) >= 2:
        difference = prediction_changes(records[0], records[-1])
        if difference["comparable"]:
            changes = pd.DataFrame(difference["rows"])
            summary += f" {difference['changed_predictions']} class decisions changed between the first and last selections."
            summary += f" Mean absolute alert-probability change: {difference['mean_probability_shift']:.3f}."
            weight_delta = parameter_change(records[0]["final_parameters"], records[-1]["final_parameters"])
            if weight_delta:
                summary += " Parameter change norms: " + ", ".join(f"{name}={value:.3f}" for name, value in weight_delta.items())
    return summary, table, figure, compare_goals(records, manager.original_goal()), changes


def lineage_view(manager, selected_id=None):
    datasets = [{"dataset": d.id, "name": d.name, "parent": d.parent_id or "Original", "label_author": d.label_author,
                 "decision": d.note, "features": ", ".join(d.features)} for d in manager.datasets.all()]
    experiments = []
    for record in manager.records():
        c = record["config"]
        experiments.append({"experiment": c["name"], "id": record["id"], "parent": record["parent_id"] or "Fresh initialization",
                            "dataset": c["dataset_id"], "goal_revision": c["goal_id"], "intervention": c["hypothesis"],
                            "weight_edits": len(c["edits"]), "frozen": ", ".join(c["frozen_parameters"]),
                            "penalty": c["false_negative_cost"], "threshold": c["threshold"],
                            "accuracy": record["metrics"]["accuracy"], "missed_alerts": record["metrics"]["false_negatives"]})
    goals = [{"revision": revision["id"], "parent": revision["parent_id"] or "Original",
              "profile": revision["spec"]["profile"]["name"], "author": revision["spec"]["profile"]["author"],
              "hypothesis": revision["spec"]["hypothesis"], "created": revision["created_at"]} for revision in manager.goals()]
    record = manager.record(selected_id) if selected_id else None
    return pd.DataFrame(datasets), pd.DataFrame(experiments), pd.DataFrame(goals), pd.DataFrame(record["values_trace"] if record else [])
