import gradio as gr


def render(manager):
    choices = [(r["config"]["name"] + " / " + r["id"], r["id"]) for r in manager.records()]
    controls = {}
    controls["run"] = gr.Dropdown(choices, value=choices[-1][1] if choices else None, label="Saved experiment")
    controls["summary"] = gr.Markdown("Run an experiment to see held-out results and goal evidence.")
    with gr.Row():
        controls["metrics"] = gr.Dataframe(headers=["metric", "value"], interactive=False, label="Evaluation metrics", scale=1)
        controls["confusion"] = gr.Plot(label="Confusion matrix", scale=1, show_label=False)
    controls["training"] = gr.Plot(label="Training dynamics", show_label=False)
    controls["goals"] = gr.Dataframe(headers=["goal", "criterion", "observed", "status"],
                                      interactive=False, label="Goals saved with this experiment")
    controls["tests"] = gr.Dataframe(headers=["name", "expected", "observed", "examples", "status"],
                                      interactive=False, label="Acceptance tests saved before training")
    with gr.Accordion("Subgroup evidence", open=False):
        controls["subgroups"] = gr.Dataframe(interactive=False, label="Held-out subgroups")
    with gr.Accordion("Inspect all learned parameters", open=False):
        controls["parameters"] = gr.JSON(label="Learned parameters")
    with gr.Accordion("Test an observation / threshold intervention", open=False):
        controls["example"] = gr.Code(value='{"pm25": 80, "pm10": 110, "wind": 1, "hour": 8}',
                                        language="json", label="Observation", lines=3, interactive=True)
        controls["probe"] = gr.Button("Test observation")
        controls["probe_result"] = gr.JSON(label="Prediction and class probabilities")
        controls["threshold"] = gr.Slider(0.01, 0.99, value=0.3, step=0.01, label="New decision threshold",
                                          info="Creates an evaluation-only experiment using the saved checkpoint.")
        controls["branch_threshold"] = gr.Button("Save threshold experiment")
    return controls
