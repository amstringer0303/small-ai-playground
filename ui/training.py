import gradio as gr

from ui.model_lab import render as model_controls


def render(manager):
    controls = {}
    controls["dataset_status"] = gr.Markdown(f"Dataset: `{manager.datasets.all()[-1].id}`. Goals: `{manager.goals()[-1]['id']}`.")
    with gr.Row():
        controls["name"] = gr.Textbox(value="A / Baseline", label="Experiment name")
        controls["seed"] = gr.Number(value=42, precision=0, label="Model seed",
                                     info="Controls initialization and training randomness, independently of the data split.")
    controls["hypothesis"] = gr.Textbox(value="Measure the starting point before removing location.", label="Expected change", lines=2)
    with gr.Row():
        with gr.Column(scale=1):
            controls.update(model_controls())
        with gr.Column(scale=1):
            gr.Markdown("### Training objective")
            controls["positive_label"] = gr.Textbox(value="alert", label="Consequential category",
                                                    info="The category whose missed detections count as false negatives.")
            controls["penalty"] = gr.Slider(0.1, 30, value=1, step=0.1, label="False-negative penalty (class / loss weight)",
                                            info="Multiplies training loss for alert examples; changes what the model learns.")
            controls["apply_priorities"] = gr.Button("Apply missed-alert / false-alarm priorities")
            controls["objective_note"] = gr.Markdown("The baseline uses a 1x penalty. Priorities take effect when explicitly applied.")
            controls["balance"] = gr.Radio([("Original class mix", "none"), ("Inverse-frequency class weights", "class_weight"),
                                              ("Oversample minority classes", "oversample")], value="none", label="Training representation")
            gr.Markdown("### Inference parameter")
            controls["threshold"] = gr.Slider(0.01, 0.99, value=0.5, step=0.01, label="Alert probability threshold",
                                              info="Lower thresholds issue more alerts. This changes outputs, not learned weights.")
            controls["block_sensitive"] = gr.Checkbox(False, label="Block sensitive inputs (hard constraint)",
                                                      info="Goals are assessed after a run; this constraint prevents a run using marked inputs.")
            choices = [(r["config"]["name"] + " / " + r["id"], r["id"]) for r in manager.records()]
            controls["parent"] = gr.Dropdown(choices, value=None, label="Starting checkpoint (optional)",
                                             info="Neural runs continue the saved weights; other model families train fresh.")
    controls["run"] = gr.Button("Run and save experiment", variant="primary", elem_id="run-experiment")
    controls["status"] = gr.Markdown("CPU training. Data, checkpoints, and records stay on this computer.", elem_id="training-status")
    return controls
