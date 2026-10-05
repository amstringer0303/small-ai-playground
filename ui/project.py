import gradio as gr


def project_fields(goal):
    with gr.Row():
        project = gr.Textbox(value=goal.project, label="Project", scale=2)
        profile = gr.Textbox(value=goal.profile["name"], label="Value profile", scale=1)
        author = gr.Textbox(value=goal.profile["author"], label="Participant / label author", scale=1)
    problem = gr.Textbox(value=goal.problem, label="Classification problem", lines=2)
    hypothesis = gr.Textbox(value=goal.hypothesis, label="Research hypothesis", lines=2,
                           info="State a change you expect to observe across experiments.")
    return {"project": project, "profile": profile, "author": author, "problem": problem, "hypothesis": hypothesis}
