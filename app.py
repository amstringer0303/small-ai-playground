import argparse
import os
from pathlib import Path

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import gradio as gr
from starlette.middleware import Middleware

from advanced_lab import build_app as build_advanced, checked
from data.datasets import ROOT
from playground.service import Playground
from playground.community import FOCUS, QUESTIONS, decision_text
from playground.views import pair_view, targets
from ui.offline import LocalAssetsMiddleware


def build_app(workspace=None):
    playground = Playground(Path(workspace or ROOT / "outputs" / "simple-lab"))
    initial_editor = playground.editor()
    first_reading = initial_editor.iloc[0].Reading
    first_label = initial_editor.iloc[0].Label

    def reading_text(row):
        flag = "simulated sensor-problem flag" if row.unreliable else "no sensor-problem flag"
        return f"**{row.site}** / simulated PM2.5 **{row.pm25:g}** / PM10 **{row.pm10:g}** / {flag}."

    advanced, _, advanced_sync = build_advanced(playground.manager.root, expose_sync=True)
    with gr.Blocks(title="Small AI Playground", analytics_enabled=False) as demo:
        gr.HTML('<header id="simple-header"><div><h1>Small AI Playground</h1><span>Air-quality playground</span></div>'
                '<div class="local-state">LOCAL CPU / NO AI API</div></header>')
        with gr.Tabs() as tabs:
            with gr.Tab("Playground", id="playground"):
                with gr.Row(elem_id="simple-controls"):
                    with gr.Column(scale=3, min_width=280):
                        gr.Markdown("### 1. Community question")
                        choice = gr.Dropdown([(focus, mode) for mode, focus in FOCUS.items()], value="missed",
                                             label="Question", elem_id="simple-choice")
                        question_text = gr.Markdown(f"**{QUESTIONS['missed']}**", elem_id="community-question-text")
                        gr.Markdown("**720 fictional readings.** Simulated labels: **alert** = needs review; "
                                    "**normal** = not flagged. Not health guidance.")
                    with gr.Column(scale=2, min_width=280):
                        gr.Markdown("### 2. Training choice")
                        penalty = gr.Slider(1, 10, value=5, step=1, label="Alert-error importance",
                                            elem_id="simple-penalty")
                        decision = gr.Markdown(decision_text("missed", 5, {"labels": 0, "excluded": 0}), elem_id="simple-decision")
                        with gr.Group(visible=False, elem_id="reading-correction") as correction_form:
                            reading = gr.Dropdown([(f"{row.Reading} / {row.Site}", row.Reading)
                                                   for row in initial_editor.itertuples()], value=first_reading,
                                                  label="Reading", elem_id="correction-reading")
                            reading_details = gr.Markdown(reading_text(playground.reading(initial_editor, first_reading)))
                            corrected_label = gr.Radio([("Normal (not flagged)", "normal"), ("Alert (needs review)", "alert")],
                                                       value=first_label, label="Training label", elem_id="correction-label")
                            included = gr.Checkbox(value=True, label="Use reading in training", elem_id="correction-included")
                        gr.Markdown("**Model:** small CPU neural network")
                        with gr.Row(elem_id="training-actions"):
                            run = gr.Button("Train and compare", variant="primary", elem_id="simple-run", scale=2)
                            reset = gr.Button("Reset readings", size="sm", scale=0, interactive=False,
                                              elem_id="reset-readings")
                        status = gr.Markdown("Ready / original priorities vs. your training choice", elem_id="simple-status")
                with gr.Accordion("Recorded success targets", open=False):
                    goal_status = gr.Markdown(targets(playground), elem_id="community-targets")
                with gr.Accordion("Training readings", open=False, elem_id="training-readings"):
                    gr.Markdown("**Learning examples:** 576 readings with target labels. "
                                "**Evaluation examples:** 144 separate readings with unchanged original labels. "
                                "The neural network also reserves some learning examples for validation.")
                    editor = gr.Dataframe(value=initial_editor, type="pandas", interactive=True,
                                          datatype=["str", "bool", "number", "number", "number", "str", "str", "str"],
                                          static_columns=[0, 2, 3, 4, 5, 6],
                                          label="Training readings", max_height=280, pinned_columns=2,
                                          show_search="filter", buttons=["fullscreen"], elem_id="simple-training-data")
                    with gr.Row():
                        gr.DownloadButton("Source CSV", value=str(ROOT / "data" / "air_quality.csv"), size="sm", scale=0)
                    with gr.Accordion("Data source", open=False):
                        gr.Markdown((ROOT / "data" / "PROVENANCE.md").read_text(encoding="utf-8"))
                gr.Markdown("### 3. Saved comparison")
                summary = gr.Markdown("No saved comparison yet.", elem_id="simple-summary")
                gr.Markdown("**Missed alert:** an alert reading classified as normal. "
                            "**False alarm:** a normal reading classified as alert. "
                            "**Accuracy:** the share of all readings classified correctly.", elem_id="comparison-terms")
                with gr.Row(elem_id="simple-results"):
                    with gr.Column(scale=3, min_width=280):
                        metrics = gr.HTML(elem_id="simple-metrics")
                    with gr.Column(scale=2, min_width=280):
                        plot = gr.Plot(label="Missed alerts and false alarms", show_label=False)
                with gr.Accordion("Original goals and changed predictions", open=False):
                    goals = gr.Dataframe(interactive=False, label="Original acceptance criteria", wrap=True)
                    predictions = gr.Dataframe(interactive=False, label="Readings whose classification changed", wrap=True)
                    gr.Markdown("An alert score is a model-assigned probability, not a measured health risk.")
                with gr.Accordion("Learned weights", open=False):
                    gr.Markdown("Neural weights are learned numbers connecting inputs to predictions. "
                                "A changed weight does not establish a causal relationship in the real environment.")
                    with gr.Tabs():
                        weight_outputs = []
                        for title in ["Original model", "Your changed model"]:
                            with gr.Tab(title):
                                weight_summary = gr.Markdown()
                                heatmap = gr.Plot(label="Weights before training, after training, and difference", show_label=False)
                                connections = gr.Dataframe(interactive=False, label="Individual connections", max_height=200)
                                distribution = gr.Plot(label="Learned-weight distributions", show_label=False)
                                weight_outputs += [heatmap, connections, distribution, weight_summary]
                pair_id = gr.State(None)
                export = gr.Button("Download experiment record", size="sm", interactive=False)
                download = gr.File(label="Experiment record", visible=False, interactive=False, elem_id="simple-download")
                study_path = playground.manager.root / "community-study.html"
                if study_path.exists():
                    gr.DownloadButton("Download worked example", value=str(study_path), size="sm", scale=0,
                                      elem_id="community-study")
            with gr.Tab("Saved runs", id="saved"):
                history = gr.Dataframe(value=playground.history(), interactive=False, label="Saved comparisons", wrap=True)
                saved = gr.Dropdown(label="Comparison", choices=[], elem_id="simple-saved")
                refresh = gr.Button("Refresh saved runs", size="sm")
                reopen = gr.Button("Open comparison", size="sm", interactive=bool(playground.pairs()))
                saved_summary = gr.Markdown()
                saved_metrics = gr.HTML()
                saved_plot = gr.Plot(label="Missed alerts and false alarms", show_label=False)
            with gr.Tab("Advanced"):
                advanced_refresh = gr.Button("Refresh advanced experiments", size="sm")
                advanced.render()

        display = [summary, metrics, plot, goals, predictions] + weight_outputs

        def show(identifier):
            return pair_view(playground, identifier)

        def choices():
            return [(f"Run {p['number']}: {p['decision']}", p["id"]) for p in playground.pairs()]

        def load():
            pairs = playground.pairs()
            identifier = pairs[-1]["id"] if pairs else None
            return (identifier, playground.history(), gr.update(choices=choices(), value=identifier),
                    gr.update(interactive=bool(identifier)), targets(playground), gr.update(interactive=bool(identifier)))

        def train(mode, importance, readings):
            pair = playground.run(mode, importance, readings)
            reuse = " Original model reused." if pair["baseline_reused"] else " Both models trained locally."
            return pair["id"], f"Saved Run {pair['number']}.{reuse}", gr.update(interactive=True), gr.update(value=None, visible=False)

        def refresh_saved():
            pairs = playground.pairs()
            identifier = pairs[-1]["id"] if pairs else None
            return playground.history(), gr.update(choices=choices(), value=identifier), gr.update(interactive=bool(identifier))

        def decision_view(mode, importance, readings):
            corrections = playground.corrections(readings)
            ready = corrections["changed"] if mode == "labels" else not corrections["changed"]
            pending = "Waiting for a label or inclusion correction." if mode == "labels" else "Unapplied data corrections."
            return (gr.update(visible=mode == "missed"), decision_text(mode, importance, corrections),
                    f"**{QUESTIONS[mode]}**", gr.update(visible=mode == "labels"), gr.update(interactive=ready),
                    "Ready / original priorities vs. your training choice" if ready else pending,
                    gr.update(interactive=corrections["changed"]))

        def reading_view(identifier, readings):
            row = playground.reading(readings, identifier)
            return row.condition, bool(row.included), reading_text(row)

        draft_inputs = [choice, penalty, editor]
        draft_outputs = [penalty, decision, question_text, correction_form, run, status, reset]
        choice.change(checked(decision_view), draft_inputs, draft_outputs)
        editor.input(lambda: "labels", [], [choice])
        penalty.change(checked(decision_view), draft_inputs, draft_outputs)
        editor.change(checked(decision_view), draft_inputs, draft_outputs)
        reading.change(checked(reading_view), [reading, editor], [corrected_label, included, reading_details])
        editor.change(checked(reading_view), [reading, editor], [corrected_label, included, reading_details])
        def correct(identifier, label, use, readings):
            return playground.correct_reading(readings, identifier, label, use), "labels"

        correction_inputs = [reading, corrected_label, included, editor]
        checked_correct = checked(correct)
        corrected_label.input(checked_correct, correction_inputs, [editor, choice], concurrency_id="reading-correction")
        included.input(checked_correct, correction_inputs, [editor, choice], concurrency_id="reading-correction")
        reset.click(lambda: playground.editor(), [], [editor])
        run.click(checked(train), [choice, penalty, editor], [pair_id, status, export, download],
                  api_name="train_comparison", concurrency_id="model-training") \
            .success(checked(show), [pair_id], display) \
            .then(load, [], [pair_id, history, saved, export, goal_status, reopen]) \
            .then(advanced_sync["refresh"], [], advanced_sync["outputs"])
        advanced_refresh.click(advanced_sync["refresh"], [], advanced_sync["outputs"])
        saved.change(checked(lambda identifier: show(identifier)[:3]), [saved], [saved_summary, saved_metrics, saved_plot])
        refresh.click(refresh_saved, [], [history, saved, reopen])
        reopen.click(lambda identifier: identifier, [saved], [pair_id]) \
            .then(checked(show), [pair_id], display) \
            .then(lambda: (gr.update(selected="playground"), gr.update(visible=False), gr.update(interactive=True)),
                  [], [tabs, download, export])
        export.click(checked(lambda identifier: gr.update(value=str(playground.export(identifier)), visible=True)), [pair_id], [download])
        demo.load(load, [], [pair_id, history, saved, export, goal_status, reopen]).then(checked(show), [pair_id], display)
    return demo, playground


def main():
    parser = argparse.ArgumentParser(description="Simple local air-quality experiments")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args()
    demo, playground = build_app(args.workspace)
    theme = gr.themes.Base(primary_hue="emerald", secondary_hue="sky", neutral_hue="gray",
                           font=["Arial", "sans-serif"], font_mono=["Consolas", "monospace"])
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=args.port, share=False, inbrowser=False,
        theme=theme, css_paths=[ROOT / "ui" / "style.css", ROOT / "playground" / "style.css"], footer_links=[],
        allowed_paths=[str(playground.manager.root), str(playground.manager.root.parent / "exports"),
                       str(ROOT / "data" / "air_quality.csv")],
        app_kwargs={"middleware": [Middleware(LocalAssetsMiddleware)]},
    )


if __name__ == "__main__":
    main()
