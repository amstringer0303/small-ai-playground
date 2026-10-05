from dataclasses import asdict, dataclass, field

PRIORITIES = {
    "accuracy": 4, "privacy": 5, "false_negatives": 5, "false_positives": 1,
    "interpretability": 4, "community_control": 5, "local_operation": 5,
    "low_compute": 3, "low_cost": 4, "retrainability": 5, "accessibility": 4,
}
DEFAULT_TESTS = [
    {"name": "Extreme pollution", "features": {"pm25": 100, "pm10": 140, "humidity": 60,
      "temperature": 22, "wind": 1, "hour": 8}, "expected": "alert", "min_probability": 0.8},
    {"name": "Clean conditions", "features": {"pm25": 3, "pm10": 5, "humidity": 45,
      "temperature": 20, "wind": 7, "hour": 13}, "expected": "normal", "min_probability": 0.6},
    {"name": "High PM2.5 subgroup", "where": {"feature": "pm25", "operator": ">=", "value": 70},
     "expected": "alert", "min_rate": 0.9},
]


@dataclass
class ValueProfile:
    name: str = "Community profile"
    author: str = "Local participant"
    priorities: dict[str, int] = field(default_factory=lambda: dict(PRIORITIES))

    def validate(self):
        if set(self.priorities) != set(PRIORITIES):
            raise ValueError("Keep each priority as a separate dimension.")
        if any(not isinstance(v, int) or not 0 <= v <= 5 for v in self.priorities.values()):
            raise ValueError("Priority importance must be an integer from 0 to 5.")


@dataclass
class GoalSpec:
    project: str = "Community air-quality lab"
    problem: str = "Classify synthetic observations as normal or alert."
    hypothesis: str = "Removing location protects privacy; weighting missed alerts may recover recall."
    profile: dict = field(default_factory=lambda: asdict(ValueProfile()))
    min_accuracy: float = 0.8
    max_false_negative_rate: float = 0.1
    max_false_positive_rate: float = 0.2
    no_geography: bool = True
    require_local: bool = True
    require_offline: bool = True
    require_inspectable: bool = True
    max_ram_gb: float = 4.0
    tests: list[dict] = field(default_factory=lambda: [dict(test) for test in DEFAULT_TESTS])

    def validate(self):
        ValueProfile(**self.profile).validate()
        for value in [self.min_accuracy, self.max_false_negative_rate, self.max_false_positive_rate]:
            if not 0 <= value <= 1:
                raise ValueError("Goal rates must be between zero and one.")
        if not 0.25 <= self.max_ram_gb <= 64:
            raise ValueError("RAM goal must be between 0.25 and 64 GB.")
        if not self.project.strip() or not self.hypothesis.strip():
            raise ValueError("Give the project a name and state a hypothesis.")
        from evaluation.custom_tests import validate_tests
        validate_tests(self.tests)
