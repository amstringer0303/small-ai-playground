import pytest

from experiments.manager import ExperimentManager
from experiments.schema import ExperimentConfig


@pytest.fixture
def manager(tmp_path):
    return ExperimentManager(tmp_path / "lab")


@pytest.fixture
def config(manager):
    return ExperimentConfig(dataset_id=manager.datasets.all()[0].id, goal_id=manager.goals()[-1]["id"])
