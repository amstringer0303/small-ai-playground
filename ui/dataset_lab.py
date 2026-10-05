import gradio as gr


def render(manager):
    version = manager.datasets.all()[-1]
    frame = manager.datasets.training_frame(version.id)
    features = [c for c in frame.columns if c not in {"example_id", "included", "condition"}]
    controls = {}
    with gr.Row():
        controls["version"] = gr.Dropdown([(v.name + " / " + v.id, v.id) for v in manager.datasets.all()],
                                          value=version.id, label="Dataset version", scale=3, elem_id="dataset-version")
        controls["name"] = gr.Textbox(value="Location removed", label="New variant name", scale=2)
    controls["summary"] = gr.Markdown(f"**{len(frame)} training candidates**; holdout remains separate. Source: `{version.id}`.", elem_id="dataset-summary")
    controls["editor"] = gr.Dataframe(value=frame, type="pandas", datatype="auto", interactive=True,
                                      label="Training examples", max_height=440, wrap=False,
                                      buttons=["fullscreen", "copy"], show_search="filter",
                                      pinned_columns=2,
                                      elem_id="training-data")
    with gr.Row():
        controls["remove_geo"] = gr.Button("Remove location inputs")
        controls["exclude_unreliable"] = gr.Button("Exclude unreliable observations")
    with gr.Row():
        with gr.Column():
            controls["features"] = gr.CheckboxGroup(features, value=version.features, label="Model input features",
                                                    info="Unchecked columns stay in the local source snapshot but are never model inputs.")
        with gr.Column():
            controls["sensitive"] = gr.CheckboxGroup(features, value=version.sensitive_features,
                                                     label="Sensitive variables", info="Can be blocked with the hard constraint in Training.")
    with gr.Row():
        controls["test_fraction"] = gr.Slider(10, 40, value=version.test_fraction * 100, step=5, label="Held-out share (%)",
                                             info="Changing this creates a different evaluation cohort.")
        controls["split_seed"] = gr.Number(value=version.split_seed, precision=0, label="Split seed",
                                           info="Same seed/share preserves the same original held-out rows.")
    with gr.Accordion("Add examples or features", open=False):
        with gr.Row():
            controls["new_example"] = gr.Code(value='{"pm25": 90, "pm10": 130, "condition": "alert"}',
                                              language="json", label="New example", lines=3, interactive=True)
            controls["add_example"] = gr.Button("Add example", scale=0)
        with gr.Row():
            controls["feature_name"] = gr.Textbox(label="New feature name", placeholder="community_rating")
            controls["feature_default"] = gr.Textbox(value="0", label="Initial value (JSON)",
                                                     info="A number, text, or true/false; edit the new column per observation.")
            controls["add_feature"] = gr.Button("Add feature", scale=0)
    controls["note"] = gr.Textbox(label="Data decision / provenance note", lines=2,
                                  placeholder="Removed location identifiers; labels provided by the community profile.")
    with gr.Row():
        controls["save"] = gr.Button("Save dataset variant", variant="primary")
        controls["export"] = gr.Button("Export selected dataset")
    controls["download"] = gr.File(label="Dataset CSV", interactive=False, visible=False)
    controls["changes"] = gr.JSON(value=version.changes, label="Variant changes")
    with gr.Accordion("Compare dataset versions", open=False):
        choices = [(v.name + " / " + v.id, v.id) for v in manager.datasets.all()]
        controls["left"] = gr.Dropdown(choices, value=manager.datasets.all()[0].id, label="Dataset A")
        controls["right"] = gr.Dropdown(choices, value=version.id, label="Dataset B")
        controls["compare"] = gr.Button("Compare datasets")
        controls["difference"] = gr.JSON(label="Dataset differences")
    controls["new_defaults"] = gr.State({})
    return controls
