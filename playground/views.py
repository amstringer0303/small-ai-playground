import pandas as pd
from matplotlib.figure import Figure

from experiments.comparison import compare_goals, prediction_changes
from ui.views import weight_view


def targets(playground):
    goal = playground.manager.goal(playground.goal_id())
    return (f"Accuracy >= {goal.min_accuracy:.0%}  |  Missed-alert rate <= {goal.max_false_negative_rate:.0%}  |  "
            f"False-alarm rate <= {goal.max_false_positive_rate:.0%}")


def metric_table(original, changed):
    a, b = original["metrics"], changed["metrics"]
    return pd.DataFrame([
        {"Result": "Accuracy", "Original": f"{a['accuracy']:.1%}", "Your choice": f"{b['accuracy']:.1%}",
         "Difference": f"{(b['accuracy'] - a['accuracy']) * 100:+.1f} percentage points"},
        {"Result": "Missed alerts", "Original": str(a["false_negatives"]), "Your choice": str(b["false_negatives"]),
         "Difference": f"{b['false_negatives'] - a['false_negatives']:+d}"},
        {"Result": "False alarms", "Original": str(a["false_positives"]), "Your choice": str(b["false_positives"]),
         "Difference": f"{b['false_positives'] - a['false_positives']:+d}"},
    ])


def outcome_plot(original, changed):
    figure = Figure(figsize=(5, 2.8), layout="constrained", facecolor="white")
    axes = figure.subplots(1, 2)
    for axis, key, label, color in zip(axes, ["false_negatives", "false_positives"],
                                     ["Missed alerts", "False alarms"], ["#b13b61", "#2b75a0"]):
        values = [original["metrics"][key], changed["metrics"][key]]
        bars = axis.barh(["Original", "Your choice"], values, color=["#d9dfe3", color], height=0.52)
        axis.bar_label(bars, padding=5, fontsize=12)
        axis.invert_yaxis()
        axis.set(title=label, xlabel="Test readings", xlim=(0, max(values + [1]) * 1.25))
        axis.spines[["top", "right", "left"]].set_visible(False)
        axis.grid(axis="x", alpha=0.15)
        axis.set_axisbelow(True)
    return figure


def pair_view(playground, identifier):
    if not identifier:
        return ("No saved comparisons.", "", None, pd.DataFrame(), pd.DataFrame(),
                None, pd.DataFrame(), None, "", None, pd.DataFrame(), None, "")
    pair = playground.pair(identifier)
    original, changed = playground.records(pair)
    difference = prediction_changes(original, changed)
    reference = playground.manager.datasets.reference(playground.original.test_fraction, playground.original.split_seed)
    alerts = int(reference.condition.eq("alert").sum())
    normals = len(reference) - alerts
    summary = (f"**Run {pair['number']}** / {pair['decision']}\n\n"
               f"{len(reference)} identical test readings: {alerts} alert, {normals} normal. "
               f"**{difference['changed_predictions']} predictions changed.**")
    if not difference["changed_predictions"]:
        summary += " No classifications changed; probabilities may still differ."
    goal_rows = compare_goals([original, changed], playground.manager.goal(pair["goal_id"]))
    goals = pd.DataFrame([
        {"Goal": rows.iloc[0]["goal"], "Criterion": rows.iloc[0]["criterion"],
         "Original": rows.iloc[0]["status"], "Your choice": rows.iloc[1]["status"]}
        for _, rows in goal_rows.groupby("goal", sort=False)
    ])
    predictions = pd.DataFrame(difference["rows"])
    predictions = predictions.loc[predictions.before.ne(predictions.after)].head(12)
    predictions = predictions.rename(columns={"example_id": "Reading", "expected": "Original test label",
        "before": "Original prediction", "after": "Your prediction", "probability_change": "Alert probability change"})
    if "Alert probability change" in predictions:
        predictions["Alert probability change"] = predictions["Alert probability change"].round(3)
    a = weight_view(original, "hidden.weight")
    b = weight_view(changed, "hidden.weight")
    table = metric_table(original, changed).to_html(index=False, border=0, classes="outcome-table")
    return (summary, table, outcome_plot(original, changed), goals, predictions,
            a[1], a[2], a[3], a[0], b[1], b[2], b[3], b[0])
