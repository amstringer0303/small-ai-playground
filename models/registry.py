from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    family: str
    architecture: str
    parameters: str
    weight_availability: str
    license: str
    representation: str
    hardware: str
    provenance: str
    capabilities: tuple[str, ...]
    access_category: str = "Open-source architecture implementation; locally accessible learned weights"
    pretrained: bool = False
    checkpoint_license: str = "Undecided: private research prototype"


MODELS = {
    model.id: model for model in [
        ModelInfo("logistic", "Logistic regression", "Linear classifier", "Multinomial linear + softmax",
                  "Classes x encoded features, plus biases", "Locally learned; all coefficients accessible",
                  "scikit-learn BSD-3-Clause; checkpoint licensing undecided", "float64; no quantization",
                  "CPU; typically a small coefficient matrix", "https://scikit-learn.org/stable/modules/linear_model.html",
                  ("train", "inspect", "class_weight", "threshold")),
        ModelInfo("tree", "Decision tree", "Tree classifier", "CART decision tree",
                  "Depends on fitted tree nodes", "Locally learned; every split accessible",
                  "scikit-learn BSD-3-Clause; checkpoint licensing undecided", "Tree thresholds; no quantization",
                  "CPU; depth controls size", "https://scikit-learn.org/stable/modules/tree.html",
                  ("train", "inspect", "class_weight", "threshold")),
        ModelInfo("forest", "Random forest", "Tree ensemble", "Bootstrap ensemble of CART trees",
                  "Depends on tree count and depth", "Locally learned; every tree accessible",
                  "scikit-learn BSD-3-Clause; checkpoint licensing undecided", "Tree ensemble; no quantization",
                  "CPU; tree count controls memory", "https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees",
                  ("train", "inspect", "class_weight", "threshold")),
        ModelInfo("tiny_neural", "Small PyTorch network", "Neural classifier", "Linear -> ReLU -> Linear -> softmax",
                  "(Inputs + 1) x hidden units + (hidden units + 1) x classes",
                  "Locally learned; every weight and bias editable",
                  "PyTorch BSD-3-Clause; checkpoint licensing undecided", "float32; no quantization",
                  "CPU; 2-32 hidden units", "https://pytorch.org/docs/stable/nn.html",
                  ("train", "inspect", "class_weight", "threshold", "edit_weights", "freeze", "reset", "warm_start")),
    ]
}


def get_model(model_id: str) -> ModelInfo:
    if model_id not in MODELS:
        raise ValueError("Select a supported local model.")
    return MODELS[model_id]


def registry_rows() -> list[dict]:
    return [asdict(model) for model in MODELS.values()]
