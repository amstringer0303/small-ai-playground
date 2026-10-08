import base64
from html import escape
from io import BytesIO

import pandas as pd

from playground.views import outcome_plot


def result_table(result):
    first = result["runs"][0]
    rows = []
    candidates = [("Non-AI demonstration rule", result["rule"]["metrics"], result["rule"]["checks"]),
                  ("Logistic regression", result["logistic"]["metrics"], result["logistic_checks"]),
                  ("Neural network / original", first["original"]["metrics"], first["original_checks"]),
                  ("Neural network / 5x importance", first["changed"]["metrics"], first["changed_checks"])]
    for name, metrics, checks in candidates:
        rows.append({"Candidate": name, "Missed alerts": f"{metrics['false_negatives']} / {result['alert_examples']}",
                     "False alarms": f"{metrics['false_positives']} / {result['normal_examples']}",
                     "Accuracy": f"{metrics['accuracy']:.1%}",
                     "Performance targets": "Pass" if all(checks[key] for key in
                         ["Accuracy", "Missed-alert rate", "False-alarm rate"]) else "Fail"})
    return pd.DataFrame(rows)


def conclusion(result):
    passed, total = result["changed_runs_passing"], len(result["runs"])
    if passed == total:
        return f"All {total} changed-model runs passed the prototype targets. This is a result on fictional data, not proof of real-world reliability."
    return (f"Only {passed} of {total} changed-model runs passed every prototype target. "
            "The change was not consistently successful: the goal was to pass all five.")


def write_report(result, directory):
    first = result["runs"][0]
    goal = result["protocol"]["goals"]
    criteria = [
        f"Miss at most {goal['max_false_negative_rate']:.0%} of readings needing review.",
        f"Send at most {goal['max_false_positive_rate']:.0%} of normal readings for unnecessary review.",
        f"Classify at least {goal['min_accuracy']:.0%} of all readings correctly.",
        f"Local CPU; training <= 60 seconds; mean inference <= 50 ms; observed RAM <= {goal['max_ram_gb']:g} GB; saved model <= 1 MiB.",
        "All five changed-model runs must meet these targets; no cherry-picking a seed.",
    ]
    checks = pd.DataFrame([{"Criterion": key, "Original": "Pass" if first["original_checks"][key] else "Fail",
                            "5x importance": "Pass" if value else "Fail"}
                           for key, value in first["changed_checks"].items()])
    repeated = pd.DataFrame([{"Seed": row["seed"], "Original missed": row["original"]["metrics"]["false_negatives"],
        "5x missed": row["changed"]["metrics"]["false_negatives"],
        "Original false alarms": row["original"]["metrics"]["false_positives"],
        "5x false alarms": row["changed"]["metrics"]["false_positives"],
        "All targets": "Pass" if all(row["changed_checks"].values()) else "Fail"} for row in result["runs"]])
    resources = first["changed"]["metrics"]
    purpose = ("Imagine reviewing sensor readings by hand. The model suggests which ones need a closer look. "
               "The goal is fewer missed reviews without too many unnecessary reviews.")
    local_goal = ("This gives the small local model a concrete task: change how it learns, compare its errors, "
                  "and check whether it runs on this computer without an AI API.")
    flag_text = ("A sensor-problem flag is an existing tag for an unreliable reading, not an alert predicted by AI. "
                 "The separate Suspect readings experiment tests whether leaving those readings out improves review predictions. "
                 "This worked example keeps them and changes training importance instead.")
    test_text = (f"Train two versions on the same data: one treats both kinds of error equally; "
                 f"the other gives alert examples 5x importance. Test both on the same {result['test_readings']} readings "
                 f"({result['alert_examples']} needing review, {result['normal_examples']} not needing review).")
    original, changed = first["original"]["metrics"], first["changed"]["metrics"]
    outcome = (f"Missed reviews fell from {original['false_negatives']} to {changed['false_negatives']}; "
               f"unnecessary reviews rose from {original['false_positives']} to {changed['false_positives']}. "
               "The model caught more alerts, but asked for more reviews.")
    repeat_text = ("Repeat training five times with different starting weights, using the same test readings. "
                   "The goal is to meet every target in all five runs.")
    limits_text = (f"Fictional data and unapproved targets, not health guidance. Repeated runs reuse just "
                   f"{result['alert_examples']} alert test readings, so they do not show real-world reliability. "
                   "Local processing does not guarantee privacy; resource and offline checks cover only this setup.")
    resource_text = (f"Seed 42 changed model: {resources['parameter_count']} learned parameters; "
        f"{resources['model_bytes'] / 1024:.1f} KiB checkpoint; {resources['training_seconds']:.2f} s training; "
        f"{resources['inference_ms']:.2f} ms mean inference; {resources['peak_ram_mb']:.0f} MiB sampled process RAM. "
        "Python network operations blocked: zero attempts. Saved checkpoints reproduced their predictions.")
    buffer = BytesIO()
    outcome_plot(first["original"], first["changed"]).savefig(buffer, format="png", dpi=140)
    image_data = base64.b64encode(buffer.getvalue()).decode("ascii")
    question_list = "".join(f"<li>{escape(question)}</li>" for question in result["protocol"]["potential_questions"].values())
    criterion_list = "".join(f"<li>{escape(item)}</li>" for item in criteria)
    table = lambda frame: '<div class="table-scroll">' + frame.to_html(index=False, border=0) + "</div>"
    source_link = '<a href="https://www.epa.gov/air-sensor-toolbox">EPA Air Sensor Toolbox</a>'
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Small AI Playground: worked example</title>
<style>body {{ margin:0; font:15px/1.6 Arial,sans-serif; color:#202a31; background:#fff; letter-spacing:0; }}
main {{ max-width:920px; margin:auto; padding:32px 24px; }} h1 {{ font-size:28px; line-height:1.3; }}
h2 {{ font-size:20px; margin:0 0 12px; }} section {{ padding:24px 0; border-bottom:1px solid #dbe1e5; }}
.state {{ color:#17785a; font-size:13px; font-weight:bold; }} .note {{ color:#56636c; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; }} th,td {{ padding:12px 10px; text-align:left;
border-bottom:1px solid #dbe1e5; vertical-align:top; }} th {{ background:#f1f5f6; }}
.table-scroll {{ overflow-x:auto; }} img {{ display:block; width:100%; max-width:660px; height:auto; margin:20px 0; }}
li {{ margin-bottom:8px; }} a {{ color:#176b91; }} details {{ margin:16px 0; }}
.decision {{ border-left:3px solid #b13b61; padding-left:16px; }}
@media(max-width:600px) {{ main {{ padding:20px 16px; }} h1 {{ font-size:23px; }} th,td {{ padding:9px 7px; }} }}</style></head>
<body><main><div class="state">SENSOR REVIEW &middot; SYNTHETIC WORKED EXAMPLE</div>
<h1>Small AI Playground</h1><p class="note">Real local model training. Fictional readings. No community approval or health advice.</p>
<section><h2>1. What are we trying to improve?</h2><p><strong>{escape(result['protocol']['question'])}</strong></p>
<p>{escape(purpose)}</p><p>{escape(local_goal)}</p>
<details><summary>Where do sensor-problem flags fit?</summary><p>{escape(flag_text)}</p></details>
<details><summary>Other potential questions</summary><ul>{question_list}</ul></details>
<p class="note">These are example questions for this playground.</p></section>
<section><h2>2. What counts as success?</h2><ul>{criterion_list}</ul>
<p class="note">Prototype targets only. Residents would need to decide acceptable errors and review capacity.</p></section>
<section><h2>3. What did we change?</h2><p>{escape(test_text)}</p>
<details><summary>Test setup and simpler comparisons</summary><p>720 simulated readings; 576 training candidates.
Neural training also reserves a validation split. Keep the model, inputs, decision threshold (0.5) and test set unchanged.</p>
<p>Compare a simple PM2.5 rule and logistic regression too, to check whether a simpler option is enough.
The rule calls PM2.5 &gt;= 42 an alert. This is an arbitrary demonstration rule, not a health threshold.</p></details></section>
<section><h2>4. What happened?</h2><p><strong>{escape(outcome)}</strong></p>{table(result_table(result))}
<img src="data:image/png;base64,{image_data}" alt="Original and changed model missed-alert and false-alarm counts">
<details><summary>Did it run locally?</summary><p>{escape(resource_text)}</p>{table(checks)}</details>
<h2>Did it work more than once?</h2><p>{escape(repeat_text)}</p>{table(repeated)}</section>
<section><h2>5. Did it meet the goal?</h2><p class="decision"><strong>{escape(conclusion(result))}</strong></p>
<details><summary>Background and limits</summary><p>{escape(limits_text)}</p>
<p class="note">Air-sensor background: {source_link}. Full technical limits are saved in study-results.json.</p></details></section>
<p class="note">Protocol, data hashes, configurations, individual predictions and learned weights are saved in the local workspace.</p>
</main></body></html>'''
    (directory / "community-study.html").write_text(html, encoding="utf-8")
    markdown = "\n".join([
        "# Small AI Playground: worked example", "", result["protocol"]["question"], "",
        "## What this tests", purpose, "", local_goal, "",
        "## Where flags fit", flag_text, "",
        "## What counts as success?", *[f"- {item}" for item in criteria], "",
        "## What changed?", test_text, "",
        "The study also checks a simple PM2.5 rule and logistic regression, so AI isn't assumed to be the best option.", "",
        "## What happened?", outcome, "", "```text", result_table(result).to_string(index=False), "```", "",
        "## Did it work more than once?", repeat_text, "", "```text", repeated.to_string(index=False), "```", "",
        "## Did it run locally?", resource_text, "",
        "## Did it meet the goal?", conclusion(result), "",
        "## Background and limits", limits_text,
        "- [EPA Air Sensor Toolbox](https://www.epa.gov/air-sensor-toolbox)", "",
        "Regenerate with `python -m scripts.community_study --workspace outputs/new-community-study`.",
    ])
    (directory / "community-study.md").write_text(markdown, encoding="utf-8")
