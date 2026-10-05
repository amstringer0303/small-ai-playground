import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

from experiments.comparison import compare_goals, compare_records


def export_design_record(manager, selected_ids=None, destination: Path | None = None):
    records = [manager.record(identifier) for identifier in selected_ids] if selected_ids else manager.records()
    if not records:
        raise ValueError("Run at least one experiment before exporting a design record.")
    goal = manager.original_goal()
    comparison = compare_records(records)
    assessment = compare_goals(records, goal)
    payload = {"format_version": "0.1", "title": "Small / Local AI Design Record",
               "exported_at": datetime.now(timezone.utc).isoformat(),
               "product_thesis": "Participation means consequential control over what the model learns.",
               "dataset_provenance": "Synthetic: seed 23; see data/PROVENANCE.md",
               "goal_history": manager.goals(), "original_goals": goal.__dict__,
               "experiments": records,
               "limitations": ["Synthetic labels are not a health or regulatory standard.",
                 "RAM is sampled process RSS, not an enforced OS limit.",
                 "Changing holdout cohorts breaks controlled comparability.",
                 "Raw training data and executable checkpoints are omitted from this export; local lab snapshots are needed for replay."]}
    lines = ["# Small / Local AI Design Record", "", f"Project: {goal.project}", "",
             f"Original hypothesis: {goal.hypothesis}", "", "## Experiments", ""]
    for record in records:
        config, metrics = record["config"], record["metrics"]
        lines += [f"### {config['name']} ({record['id']})", "", config["hypothesis"], "",
                  f"Dataset: {config['dataset_id']}; model: {config['model_id']}; parent: {record['parent_id'] or 'none'}.",
                  f"Accuracy: {metrics['accuracy']:.1%}; missed alerts: {metrics['false_negatives']}; false alarms: {metrics['false_positives']}.",
                  f"Alert penalty: {config['false_negative_cost']:g}x; inference threshold: {config['threshold']:g}.",
                  f"Features: {', '.join(record['features'])}.", "",
                  "Original-goal results:", ""]
        for result in assessment.loc[assessment.id == record["id"]].to_dict("records"):
            lines.append(f"- {result['goal']}: {result['status']} ({result['observed']}; {result['criterion']})")
        lines.append("")
    lines += ["## Interpretation", "", "Dimensions are separate; there is no combined ethics score.",
              "Priority ratings influence training only through explicit operational choices, recorded in each run.",
              "Goals and custom tests are versioned. Revisions do not overwrite past success criteria.", "",
              "## Reproduction", "", "Use the saved local lab and `python -m scripts.replay EXPERIMENT_ID`.",
              "The JSON contains learned parameters and configurations, but omits raw data and executable checkpoints."]
    destination = Path(destination or manager.root.parent / "exports")
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"design-record-{uuid4().hex[:10]}.zip"
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("design_record.json", json.dumps(payload, indent=2, allow_nan=False))
        archive.writestr("design_record.md", "\n".join(lines))
        archive.writestr("comparison.csv", comparison.to_csv(index=False))
        archive.writestr("original_goal_assessment.csv", assessment.to_csv(index=False))
    return path
