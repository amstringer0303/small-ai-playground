import gradio as gr

from models.registry import MODELS


def render():
    controls = {}
    with gr.Row():
        controls["model"] = gr.Radio([(m.name, m.id) for m in MODELS.values()], value="logistic", label="Local model family")
    with gr.Accordion("Architecture, weight access, and provenance", open=False):
        controls["metadata"] = gr.JSON(value=MODELS["logistic"].__dict__, label="Model metadata")
    with gr.Column(visible=True) as linear:
        controls["regularization"] = gr.Slider(0.01, 10, value=1, step=0.01, label="Regularization C",
                                              info="Smaller values keep logistic coefficients closer to zero.")
    with gr.Column(visible=False) as tree:
        controls["depth"] = gr.Slider(1, 20, value=5, step=1, label="Maximum tree depth",
                                     info="Limits how many successive decisions a tree can make.")
        controls["trees"] = gr.Slider(1, 300, value=80, step=1, label="Number of forest trees",
                                     info="Used only by random forest; more trees cost more time and memory.")
    with gr.Column(visible=False) as neural:
        controls["hidden"] = gr.Slider(2, 32, value=8, step=1, label="Hidden units",
                                       info="Small enough to inspect every learned connection.")
        controls["epochs"] = gr.Slider(1, 300, value=60, step=1, label="Training epochs",
                                       info="Number of full passes over the included training data.")
        controls["learning_rate"] = gr.Slider(0.001, 0.2, value=0.02, step=0.001, label="Learning rate",
                                              info="Size of each update to the neural weights.")
    controls.update({"linear_section": linear, "tree_section": tree, "neural_section": neural})
    return controls
