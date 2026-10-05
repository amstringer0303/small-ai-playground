import argparse
from pathlib import Path

from data.datasets import ROOT
from playground.service import Playground


def build_demo(workspace):
    playground = Playground(workspace)
    for choice in ["missed", "flagged", "location"]:
        pair = playground.run(choice)
        original, changed = playground.records(pair)
        print(f"Run {pair['number']}: {pair['decision']}")
        print(f"Original: missed={original['metrics']['false_negatives']}, alarms={original['metrics']['false_positives']}")
        print(f"Changed: missed={changed['metrics']['false_negatives']}, alarms={changed['metrics']['false_positives']}")
    print(playground.history().to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=ROOT / "outputs" / "demo-lab")
    args = parser.parse_args()
    build_demo(args.workspace.resolve())
