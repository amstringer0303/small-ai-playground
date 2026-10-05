import json

import gradio as gr

from ui.project import project_fields
from values.schema import PRIORITIES

LABELS = {
    "accuracy": "Accuracy", "privacy": "Privacy", "false_negatives": "Avoid missed alerts",
    "false_positives": "Avoid false alarms", "interpretability": "Interpretability",
    "community_control": "Community control", "local_operation": "Local operation",
    "low_compute": "Low compute", "low_cost": "Low cost", "retrainability": "Ability to retrain",
    "accessibility": "Accessibility",
}


def render(goal, revision_id):
    controls = project_fields(goal)
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### WHAT WE WEIGHT")
            controls["priorities"] = {}
            for name in PRIORITIES:
                controls["priorities"][name] = gr.Slider(0, 5, value=goal.profile["priorities"][name], step=1,
                                                       label=LABELS[name])
        with gr.Column(scale=1):
            gr.Markdown("### Testable goals")
            controls["accuracy"] = gr.Slider(0, 100, value=goal.min_accuracy * 100, step=1,
                                            label="Minimum accuracy (%)", info="Agreement with the original held-out labels.")
            controls["fnr"] = gr.Slider(0, 100, value=goal.max_false_negative_rate * 100, step=1,
                                       label="Maximum missed-alert rate (%)", info="Missed alerts divided by actual alerts.")
            controls["fpr"] = gr.Slider(0, 100, value=goal.max_false_positive_rate * 100, step=1,
                                       label="Maximum false-alarm rate (%)", info="False alarms divided by actual normal observations.")
            controls["no_geo"] = gr.Checkbox(value=goal.no_geography, label="No geographic inputs",
                                            info="Counts coordinates, site, and sensor identifiers, which may reveal location.")
            controls["local"] = gr.Checkbox(value=goal.require_local, label="Must run locally")
            controls["offline"] = gr.Checkbox(value=goal.require_offline, label="Inference must work offline")
            controls["inspectable"] = gr.Checkbox(value=goal.require_inspectable, label="Learned parameters must be inspectable")
            controls["ram"] = gr.Slider(0.25, 16, value=goal.max_ram_gb, step=0.25, label="Maximum process RAM (GB)",
                                       info="Compared with sampled process memory; this does not impose an operating-system limit.")
            with gr.Accordion("Predefined acceptance tests", open=False):
                controls["tests"] = gr.Code(value=json.dumps(goal.tests, indent=2), language="json", label="Custom tests",
                                           lines=16, interactive=True)
                gr.Markdown("Each test uses `features` + `expected` + `min_probability`, or `where` + `expected` + `min_rate`.")
    controls["save"] = gr.Button("Save goal revision", variant="primary")
    controls["status"] = gr.Markdown(f"Active goals: `{revision_id}`. Revisions preserve earlier goals and tests.", elem_id="goal-status")
    return controls
