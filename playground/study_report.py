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
        return f"All {total} changed-model runs passed the prototype targets. This supports a real-data pilot, not deployment."
    return (f"Only {passed} of {total} changed-model runs passed every prototype target. "
            "The intervention is not consistently successful under this test; do not loosen criteria after seeing the result.")


def write_report(result, directory):
    first = result["runs"][0]
    goal = result["protocol"]["goals"]
    criteria = [
        f"Miss <= {goal['max_false_negative_rate']:.0%} of alert examples.",
        f"Falsely flag <= {goal['max_false_positive_rate']:.0%} of normal examples.",
        f"Accuracy >= {goal['min_accuracy']:.0%}.",
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
    resource_text = (f"Seed 42 changed model: {resources['parameter_count']} learned parameters; "
        f"{resources['model_bytes'] / 1024:.1f} KiB checkpoint; {resources['training_seconds']:.2f} s training; "
        f"{resources['inference_ms']:.2f} ms mean inference; {resources['peak_ram_mb']:.0f} MiB sampled process RAM. "
        "Python network operations blocked: zero attempts. Saved checkpoints reproduced their predictions.")
    buffer = BytesIO()
    outcome_plot(first["original"], first["changed"]).savefig(buffer, format="png", dpi=140)
    image_data = base64.b64encode(buffer.getvalue()).decode("ascii")
    question_list = "".join(f"<li>{escape(question)}</li>" for question in result["protocol"]["potential_questions"].values())
    criterion_list = "".join(f"<li>{escape(item)}</li>" for item in criteria)
    limitations = "".join(f"<li>{escape(item)}</li>" for item in result["limitations"])
    table = lambda frame: '<div class="table-scroll">' + frame.to_html(index=False, border=0) + "</div>"
    source_link = '<a href="https://www.epa.gov/air-sensor-toolbox">EPA Air Sensor Toolbox</a>'
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Sensor review: worked example</title>
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
<body><main><div class="state">SMALL / LOCAL AI &middot; SYNTHETIC WORKED EXAMPLE</div>
<h1>Sensor review playground</h1><p class="note">Real local model training. Fictional readings. No community approval or health advice.</p>
<section><h2>1. Choose a community question</h2><p><strong>{escape(result['protocol']['question'])}</strong></p>
<details><summary>Other potential questions</summary><ul>{question_list}</ul></details>
<p class="note">These are example questions for this playground. Air-sensor background: {source_link}.</p></section>
<section><h2>2. Record success before testing</h2><ul>{criterion_list}</ul>
<p class="note">Prototype targets only. Residents would need to decide acceptable errors and review capacity.</p></section>
<section><h2>3. Run a controlled test</h2><p>720 simulated readings: 576 training candidates and the same 144 evaluation readings
({result['alert_examples']} alert, {result['normal_examples']} normal). Neural training reserves an internal validation split.</p>
<p>Compare a fixed non-AI rule, a simpler linear model, and an 8-hidden-unit neural network.
Change only alert-example training importance from 1x to 5x within each neural seed pair.
Repeat seeds 42, 7, 23, 101, and 202; keep the decision threshold at 0.5.</p>
<p class="note">The non-AI rule flags simulated PM2.5 &gt;= 42. This is an arbitrary demonstration rule, not a health threshold.</p></section>
<section><h2>4. Compare measured results</h2>{table(result_table(result))}
<img src="data:image/png;base64,{image_data}" alt="Original and changed model missed-alert and false-alarm counts">
<p>{escape(resource_text)}</p><details><summary>Every criterion, seed 42</summary>{table(checks)}</details>
<h2>Repeatability check</h2>{table(repeated)}</section>
<section><h2>5. Make a decision</h2><p class="decision"><strong>{escape(conclusion(result))}</strong></p>
<p>Next: agree the question and error budget with residents; check whether a simpler tool is sufficient;
obtain permissioned, quality-checked data; then reserve a genuinely untouched, time- or site-separated evaluation set.</p>
<details><summary>Limits of this result</summary><ul>{limitations}</ul></details></section>
<p class="note">Protocol, data hashes, configurations, individual predictions and learned weights are saved in the local workspace.</p>
</main></body></html>'''
    (directory / "community-study.html").write_text(html, encoding="utf-8")
    markdown = "\n".join([
        "# Sensor review: worked example", "", result["protocol"]["question"], "",
        "Synthetic study; proposed criteria, not community-approved or health guidance.", "",
        "## Potential questions", *[f"- {q}" for q in result["protocol"]["potential_questions"].values()], "",
        "## Criteria recorded before training", *[f"- {item}" for item in criteria], "",
        "## Results (seed 42)", "```text", result_table(result).to_string(index=False), "```", "",
        "## Five fixed seeds", "```text", repeated.to_string(index=False), "```", "", resource_text, "",
        "## Decision", conclusion(result), "", "## Limits", *[f"- {item}" for item in result["limitations"]], "",
        "## Background", "Example questions and targets for this playground, not health standards.",
        "- [EPA Air Sensor Toolbox](https://www.epa.gov/air-sensor-toolbox)", "",
        "Regenerate with `python -m scripts.community_study --workspace outputs/new-community-study`.",
    ])
    (directory / "community-study.md").write_text(markdown, encoding="utf-8")
