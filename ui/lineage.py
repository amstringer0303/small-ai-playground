import gradio as gr


def render():
    controls = {}
    controls["refresh"] = gr.Button("Refresh lineage")
    controls["datasets"] = gr.Dataframe(interactive=False, label="Dataset provenance", max_height=300)
    controls["experiments"] = gr.Dataframe(interactive=False, label="Data -> training -> parameter -> output decisions", max_height=400)
    controls["goals"] = gr.Dataframe(interactive=False, label="Goal and value-profile revisions", max_height=300)
    controls["trace"] = gr.Dataframe(interactive=False, label="Values trace for selected experiment", max_height=400, wrap=True)
    controls["export"] = gr.Button("Export Small AI Playground Design Record", variant="primary")
    controls["download"] = gr.File(label="Design record ZIP", interactive=False, visible=False)
    return controls
