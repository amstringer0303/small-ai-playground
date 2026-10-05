import math
from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    name: str = "Baseline"
    hypothesis: str = "Measure the starting model before intervening."
    dataset_id: str = ""
    goal_id: str = ""
    model_id: str = "logistic"
    seed: int = 42
    positive_label: str = "alert"
    false_negative_cost: float = 1.0
    threshold: float = 0.5
    balance: str = "none"
    regularization: float = 1.0
    tree_depth: int = 5
    trees: int = 80
    hidden_units: int = 8
    epochs: int = 60
    learning_rate: float = 0.02
    block_sensitive: bool = False
    parent_id: str | None = None
    mode: str = "train"
    edits: list[dict] = field(default_factory=list)
    frozen_parameters: list[str] = field(default_factory=list)
    frozen_cells: list[dict] = field(default_factory=list)
    reset_parameters: list[str] = field(default_factory=list)

    def validate(self):
        from models.registry import get_model
        get_model(self.model_id)
        if not self.name.strip() or not self.hypothesis.strip():
            raise ValueError("Name the experiment and state what you expect to change.")
        for value, low, high in [(self.false_negative_cost, 0.1, 30), (self.threshold, 0.01, 0.99),
                                 (self.regularization, 0.001, 100), (self.learning_rate, 0.0001, 0.2)]:
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError("A training/threshold setting is outside its supported range.")
        for value, low, high in [(self.seed, 0, 2**31 - 1), (self.tree_depth, 1, 20), (self.trees, 1, 300),
                                 (self.hidden_units, 2, 32), (self.epochs, 0, 300)]:
            if not isinstance(value, int) or not low <= value <= high:
                raise ValueError("Seeds, counts, and sizes must be whole numbers in the supported range.")
        if self.mode not in {"train", "evaluate_only"} or self.balance not in {"none", "class_weight", "oversample"}:
            raise ValueError("Choose a supported run mode and balancing strategy.")
        if self.mode == "train" and self.model_id == "tiny_neural" and self.epochs == 0:
            raise ValueError("Training needs at least one epoch; use evaluate-only for a weight/threshold intervention.")
