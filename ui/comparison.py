import gradio as gr


def render(manager):
    controls = {}
    choices = [(r["config"]["name"] + " / " + r["id"], r["id"]) for r in manager.records()]
    controls["runs"] = gr.CheckboxGroup(choices, value=[r[1] for r in choices], label="Experiments to compare")
    controls["refresh"] = gr.Button("Compare selected experiments", variant="primary")
    controls["summary"] = gr.Markdown("Comparisons use the original goal revision from the first experiment.", elem_id="comparison-summary")
    controls["table"] = gr.Dataframe(interactive=False, label="Measured tradeoffs", max_height=400)
    controls["plot"] = gr.Plot(label="Accuracy, recall, and missed-alert rate", show_label=False)
    controls["goals"] = gr.Dataframe(interactive=False, label="Every experiment against original goals", max_height=450)
    controls["changes"] = gr.Dataframe(interactive=False, label="Prediction changes: first versus last selection", max_height=300)
    return controls
