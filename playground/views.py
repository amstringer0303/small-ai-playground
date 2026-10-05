import pandas as pd
from html import escape
from matplotlib.figure import Figure

from experiments.comparison import compare_goals, prediction_changes
from ui.views import weight_view
from playground.community import check_metrics, goal_text


def targets(playground):
    goal = playground.manager.goal(playground.goal_id())
    return goal_text(goal)


def metric_table(original, changed):
    a, b = original["metrics"], changed["metrics"]
    return pd.DataFrame([
        {"Result": "Accuracy", "Original": f"{a['accuracy']:.1%}", "Your choice": f"{b['accuracy']:.1%}",
         "Difference": f"{(b['accuracy'] - a['accuracy']) * 100:+.1f} percentage points"},
        {"Result": "Missed alerts", "Original": f"{a['false_negatives']} of {a['false_negatives'] + a['true_positives']}",
         "Your choice": f"{b['false_negatives']} of {b['false_negatives'] + b['true_positives']}",
         "Difference": count_change(a["false_negatives"], b["false_negatives"])},
        {"Result": "False alarms", "Original": f"{a['false_positives']} of {a['false_positives'] + a['true_negatives']}",
         "Your choice": f"{b['false_positives']} of {b['false_positives'] + b['true_negatives']}",
         "Difference": count_change(a["false_positives"], b["false_positives"])},
    ])


def count_change(before, after):
    return f"{abs(after - before)} {'fewer' if after < before else 'more'}" if after != before else "No change"


def target_evidence(record, goal):
    metrics, checks = record["metrics"], check_metrics(record["metrics"], goal)
    entries = [
        ("Missed-alert rate", metrics["false_negative_rate"], "at most", goal.max_false_negative_rate),
        ("False-alarm rate", metrics["false_positive_rate"], "at most", goal.max_false_positive_rate),
        ("Accuracy", metrics["accuracy"], "at least", goal.min_accuracy),
    ]
    items = []
    for label, observed, direction, limit in entries:
        status = "Met" if checks[label] else "Not met"
        value = f"{observed:.1%}" if observed is not None else "No supporting examples"
        items.append(f'<div class="target-evidence"><span>{escape(label)}</span><strong>{value}</strong>'
                     f'<span class="target-status {"met" if checks[label] else "not-met"}">{status}</span>'
                     f'<small>Target: {direction} {limit:.0%}</small></div>')
    return '<div class="target-evidence-grid" aria-label="Prediction target results">' + "".join(items) + "</div>"


def interpretation(original, changed, goal):
    a, b = original["metrics"], changed["metrics"]
    misses = count_change(a["false_negatives"], b["false_negatives"]).lower()
    alarms = count_change(a["false_positives"], b["false_positives"]).lower()
    misses = "no change in missed alerts" if misses == "no change" else f"{misses} missed alerts"
    alarms = "no change in false alarms" if alarms == "no change" else f"{alarms} false alarms"
    passed = sum(check_metrics(b, goal).values())
    result = f"**The tradeoff: {misses}; {alarms}.** "
    result += ("All three prediction targets met in this run." if passed == 3 else
               f"{passed} of 3 prediction targets met in this run.")
    return result + " One run on fictional data is not evidence of reliable real-world performance."


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
        return ("No saved comparison yet.", "", None, pd.DataFrame(), pd.DataFrame(),
                None, pd.DataFrame(), None, "", None, pd.DataFrame(), None, "")
    pair = playground.pair(identifier)
    original, changed = playground.records(pair)
    difference = prediction_changes(original, changed)
    reference = playground.manager.datasets.reference(playground.original.test_fraction, playground.original.split_seed)
    alerts = int(reference.condition.eq("alert").sum())
    normals = len(reference) - alerts
    goal = playground.manager.goal(pair["goal_id"])
    decision = f"Alert-error importance: {pair['penalty']:g}x" if pair["choice"] == "missed" else pair["decision"]
    summary = (f"**Saved Run {pair['number']}** / {decision}\n\n"
               f"{interpretation(original, changed, goal)}\n\n"
               f"{len(reference)} identical test readings: {alerts} alert, {normals} normal. "
               f"**{difference['changed_predictions']} predictions changed.**")
    if not difference["changed_predictions"]:
        summary += " No classifications changed; probabilities may still differ."
    goal_rows = compare_goals([original, changed], goal)
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
    table += target_evidence(changed, goal)
    return (summary, table, outcome_plot(original, changed), goals, predictions,
            a[1], a[2], a[3], a[0], b[1], b[2], b[3], b[0])
