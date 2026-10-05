import argparse
from pathlib import Path

from experiments.manager import ExperimentManager
from experiments.runner import run_experiment
from experiments.schema import ExperimentConfig


def main():
    parser = argparse.ArgumentParser(description="Replay a saved local experiment")
    parser.add_argument("experiment_id")
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args()
    manager = ExperimentManager(args.workspace)
    original = manager.record(args.experiment_id)
    config = ExperimentConfig(**original["config"])
    config.name += " / replay"
    replay = run_experiment(manager, config)
    exact = original["predictions"] == replay["predictions"]
    print(f"Original: {original['id']} | replay: {replay['id']} | predictions identical: {exact}")
    if not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
