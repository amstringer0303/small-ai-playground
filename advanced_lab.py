import argparse
import os
from functools import wraps
from pathlib import Path

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import gradio as gr
from starlette.middleware import Middleware

from data.datasets import ROOT
from experiments.manager import ExperimentManager
from models.registry import get_model
from ui import comparison, dataset_lab, evaluation, goals, lineage, training, weights
from ui.controller import LabController
from ui.offline import LocalAssetsMiddleware
from ui.views import comparison_view, evaluation_view, lineage_view, weight_view
from values.schema import PRIORITIES


def checked(function):
    @wraps(function)
    def callback(*args):
        try:
            return function(*args)
        except (ValueError, TypeError, KeyError) as error:
            raise gr.Error(str(error)) from error
    return callback


def build_app(workspace=None, expose_sync=False):
    manager = ExperimentManager(Path(workspace) if workspace else None)
    controller = LabController(manager)
    active_goal = manager.goals()[-1]
    with gr.Blocks(title="Small / Local AI Playground", analytics_enabled=False) as demo:
        gr.HTML('<header id="lab-header"><h1>Small / Local AI Playground</h1><div class="state">LOCAL LAB &nbsp; / &nbsp; Air quality &nbsp; / &nbsp; v0.1</div></header>')
        goal_id = gr.State(active_goal["id"])
        last_run = gr.State(None)
        with gr.Tabs():
            with gr.Tab("Project & goals"):
                g = goals.render(manager.goal(active_goal["id"]), active_goal["id"])
            with gr.Tab("Dataset lab"):
                d = dataset_lab.render(manager)
            with gr.Tab("Train & models"):
                t = training.render(manager)
            with gr.Tab("Model weights"):
                w = weights.render(manager)
            with gr.Tab("Evaluation"):
                e = evaluation.render(manager)
            with gr.Tab("Compare"):
                c = comparison.render(manager)
            with gr.Tab("Lineage & export"):
                l = lineage.render()

        goal_inputs = [g[k] for k in ["project", "profile", "author", "problem", "hypothesis"]]
        goal_inputs += list(g["priorities"].values())
        goal_inputs += [g[k] for k in ["accuracy", "fnr", "fpr", "no_geo", "local", "offline", "inspectable", "ram", "tests"]]
        g["save"].click(checked(controller.save_goals), goal_inputs, [goal_id, g["status"]])

        dataset_outputs = [d[k] for k in ["editor", "features", "sensitive", "test_fraction", "split_seed", "changes", "summary", "new_defaults"]] + [t["dataset_status"]]
        d["version"].change(checked(controller.load_dataset), [d["version"]], dataset_outputs)
        d["remove_geo"].click(checked(controller.remove_geo), [d["features"]], [d["features"], d["summary"]])
        d["exclude_unreliable"].click(checked(controller.exclude_unreliable), [d["editor"]], [d["editor"], d["summary"]])
        d["add_example"].click(checked(controller.add_example), [d["editor"], d["new_example"]], [d["editor"]])
        d["add_feature"].click(checked(controller.add_feature), [d[k] for k in ["editor", "feature_name", "feature_default", "features", "sensitive", "new_defaults"]],
                               [d[k] for k in ["editor", "features", "sensitive", "new_defaults"]])
        d["save"].click(checked(controller.save_dataset), [d[k] for k in ["version", "editor", "features", "sensitive", "name", "note"]] + [g["author"]] +
                        [d[k] for k in ["test_fraction", "split_seed", "new_defaults"]], [d["version"], d["left"], d["right"]], api_name="save_dataset")
        d["compare"].click(checked(controller.compare_datasets), [d["left"], d["right"]], [d["difference"]])
        d["export"].click(checked(controller.export_dataset), [d["version"]], [d["download"]])

        t["model"].change(lambda model: (get_model(model).__dict__, gr.update(visible=model == "logistic"),
                                         gr.update(visible=model in {"tree", "forest"}), gr.update(visible=model == "tiny_neural")),
                           [t["model"]], [t[k] for k in ["metadata", "linear_section", "tree_section", "neural_section"]])
        t["apply_priorities"].click(checked(controller.apply_priorities), [goal_id], [t["penalty"], t["objective_note"]])

        eval_outputs = [e[k] for k in ["summary", "metrics", "confusion", "training", "goals", "tests", "subgroups", "parameters"]]
        weight_outputs = [w[k] for k in ["summary", "heatmap", "connections", "distribution"]]
        comparison_outputs = [c[k] for k in ["summary", "table", "plot", "goals", "changes"]]
        lineage_outputs = [l[k] for k in ["datasets", "experiments", "goals", "trace"]]

        def show_evaluation(identifier):
            return evaluation_view(manager.record(identifier) if identifier else None)

        def show_weights(identifier, parameter):
            return weight_view(manager.record(identifier) if identifier else None, parameter)

        def after_run(event):
            event.success(checked(controller.refresh_run_choices), [last_run], [e["run"], t["parent"], w["source"], c["runs"]]) \
                 .then(checked(show_evaluation), [last_run], eval_outputs) \
                 .then(checked(lambda identifiers: comparison_view(manager, identifiers)), [c["runs"]], comparison_outputs) \
                 .then(checked(lambda identifier: lineage_view(manager, identifier)), [last_run], lineage_outputs) \
                 .then(checked(show_weights), [w["source"], w["parameter"]], weight_outputs)

        train_inputs = [d["version"], goal_id] + [t[k] for k in ["name", "hypothesis", "model", "seed", "positive_label", "penalty", "threshold",
                         "balance", "regularization", "depth", "trees", "hidden", "epochs", "learning_rate", "block_sensitive", "parent"]]
        after_run(t["run"].click(checked(controller.train), train_inputs, [last_run, t["status"]], api_name="train_experiment", concurrency_id="model-training"))
        branch_inputs = [w[k] for k in ["source", "parameter", "name", "hypothesis", "row", "column", "value", "edit", "freeze_cell", "freeze", "reset", "mode", "epochs", "learning_rate"]]
        after_run(w["run"].click(checked(controller.weight_branch), branch_inputs, [last_run, w["status"]], api_name="weight_intervention", concurrency_id="model-training"))
        after_run(e["branch_threshold"].click(checked(controller.threshold_branch), [e["run"], e["threshold"]], [last_run, t["status"]], concurrency_id="model-training"))
        e["run"].change(checked(show_evaluation), [e["run"]], eval_outputs)
        w["source"].change(checked(show_weights), [w["source"], w["parameter"]], weight_outputs)
        w["parameter"].change(checked(show_weights), [w["source"], w["parameter"]], weight_outputs)
        e["probe"].click(checked(controller.probe), [e["run"], e["example"]], [e["probe_result"]])
        c["refresh"].click(checked(lambda identifiers: comparison_view(manager, identifiers)), [c["runs"]], comparison_outputs)
        l["refresh"].click(checked(lambda identifier: lineage_view(manager, identifier)), [e["run"]], lineage_outputs)
        l["export"].click(checked(controller.export), [], [l["download"]])
        session_outputs = [goal_id] + goal_inputs + [g["status"], d["version"], d["left"], d["right"],
                                                   e["run"], t["parent"], w["source"], c["runs"], last_run]
        demo.load(checked(controller.session), [], session_outputs) \
            .then(checked(controller.load_dataset), [d["version"]], dataset_outputs) \
            .then(checked(show_evaluation), [e["run"]], eval_outputs) \
            .then(checked(show_weights), [w["source"], w["parameter"]], weight_outputs) \
            .then(checked(lambda identifiers: comparison_view(manager, identifiers)), [c["runs"]], comparison_outputs) \
            .then(checked(lambda identifier: lineage_view(manager, identifier)), [e["run"]], lineage_outputs)
    if expose_sync:
        def refresh_choices():
            records = manager.records()
            selected = records[-1]["id"] if records else None
            return controller.refresh_run_choices(selected)
        return demo, manager, {"refresh": refresh_choices, "outputs": [e["run"], t["parent"], w["source"], c["runs"]]}
    return demo, manager


def main():
    parser = argparse.ArgumentParser(description="Local intervention lab for small AI models")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--workspace", type=Path, default=None, help="Separate persistent project directory")
    args = parser.parse_args()
    demo, manager = build_app(args.workspace)
    theme = gr.themes.Base(primary_hue="emerald", secondary_hue="sky", neutral_hue="gray",
                           font=["Arial", "sans-serif"], font_mono=["Consolas", "monospace"])
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=args.port, share=False, inbrowser=False,
        theme=theme, css_paths=ROOT / "ui" / "style.css", footer_links=[],
        allowed_paths=[str(manager.root), str(manager.root.parent / "exports")],
        app_kwargs={"middleware": [Middleware(LocalAssetsMiddleware)]},
    )


if __name__ == "__main__":
    main()
