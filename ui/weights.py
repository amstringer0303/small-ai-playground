import gradio as gr

PARAMETERS = ["hidden.weight", "hidden.bias", "output.weight", "output.bias"]


def render(manager):
    neural = [r for r in manager.records() if r["config"]["model_id"] == "tiny_neural"]
    controls = {}
    gr.Markdown("### MODEL WEIGHTS")
    with gr.Row():
        controls["source"] = gr.Dropdown([(r["config"]["name"] + " / " + r["id"], r["id"]) for r in neural],
                                          value=neural[-1]["id"] if neural else None, label="Neural checkpoint", scale=2)
        controls["parameter"] = gr.Dropdown(PARAMETERS, value="hidden.weight", label="Learned tensor", scale=1, elem_id="weight-parameter")
    controls["summary"] = gr.Markdown("Train a small PyTorch network to inspect its learned parameters.")
    controls["heatmap"] = gr.Plot(label="Before / after / difference", show_label=False)
    controls["connections"] = gr.Dataframe(headers=["row", "column", "before", "after", "change"],
                                            interactive=False, label="Individual connections", max_height=240)
    controls["distribution"] = gr.Plot(label="Learned-weight distributions", show_label=False)
    gr.Markdown("### Controlled weight intervention")
    with gr.Row():
        controls["name"] = gr.Textbox(value="Weight intervention", label="Branch name")
        controls["hypothesis"] = gr.Textbox(value="Changing this connection changes alert predictions.", label="Expected change")
    with gr.Row():
        controls["row"] = gr.Number(value=0, precision=0, label="Output neuron / bias index")
        controls["column"] = gr.Number(value=0, precision=0, label="Input index", info="Bias tensors use column 0.")
        controls["value"] = gr.Number(value=0, label="Replacement weight")
    with gr.Row():
        controls["edit"] = gr.Checkbox(value=True, label="Replace selected connection")
        controls["freeze_cell"] = gr.Checkbox(value=False, label="Freeze selected connection during retraining")
    controls["freeze"] = gr.CheckboxGroup(PARAMETERS, label="Freeze entire parameters", value=[],
                                          elem_id="freeze-parameters",
                                          info="Hidden weight + bias freeze the hidden layer; output weight + bias freeze the output layer.")
    controls["reset"] = gr.CheckboxGroup(PARAMETERS, label="Reset parameters to original initialization", value=[])
    with gr.Row():
        controls["mode"] = gr.Radio([("Evaluate without training", "evaluate_only"), ("Retrain from this checkpoint", "train")],
                                     value="evaluate_only", label="Intervention mode")
        controls["epochs"] = gr.Slider(1, 200, value=30, step=1, label="Retraining epochs")
        controls["learning_rate"] = gr.Slider(0.001, 0.1, value=0.01, step=0.001, label="Retraining learning rate")
    controls["run"] = gr.Button("Save intervention experiment", variant="primary")
    controls["status"] = gr.Markdown("A branch preserves the parent checkpoint and original tests.", elem_id="weight-status")
    return controls
