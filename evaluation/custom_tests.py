import math

OPERATORS = {">=": lambda series, value: series >= value, "<=": lambda series, value: series <= value,
             "==": lambda series, value: series == value, ">": lambda series, value: series > value,
             "<": lambda series, value: series < value}


def validate_tests(tests):
    if not isinstance(tests, list) or len(tests) > 50:
        raise ValueError("Custom tests must be a JSON list with at most 50 entries.")
    names = set()
    for test in tests:
        if not isinstance(test, dict) or not all(k in test for k in ["name", "expected"]):
            raise ValueError("Every test needs a name and expected label.")
        if not isinstance(test["expected"], str) or not test["expected"].strip():
            raise ValueError("Expected labels must be nonempty text.")
        if test["name"] in names:
            raise ValueError("Give each custom test a unique name.")
        names.add(test["name"])
        if ("features" in test) == ("where" in test):
            raise ValueError("Each test needs either features for one example or where for a scenario.")
        key = "min_probability" if "features" in test else "min_rate"
        value = test.get(key, 0)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Acceptance probabilities/rates must be between zero and one.")
        if "features" in test and not isinstance(test["features"], dict):
            raise ValueError("Example features must be a JSON object.")
        if "where" in test:
            where = test["where"]
            if not isinstance(where, dict) or not all(k in where for k in ["feature", "operator", "value"]):
                raise ValueError("Scenario where needs feature, operator, and value.")
            if where["operator"] not in OPERATORS:
                raise ValueError("Use >=, <=, ==, >, or < in a scenario test.")


def run_custom_tests(model, tests, reference, config):
    results = []
    for test in tests:
        base = {"name": test["name"], "expected": test["expected"]}
        if test["expected"] not in model.classes:
            results.append({**base, "passed": False, "observed": "Expected category was not learned", "examples": 0})
            continue
        try:
            if "features" in test:
                example = model.example(test["features"])
                probabilities = model.probabilities(example)[0]
                prediction = model.predict(example, config.positive_label, config.threshold)[0]
                probability = float(probabilities[model.classes.index(test["expected"])])
                passed = prediction == test["expected"] and probability >= test.get("min_probability", 0)
                results.append({**base, "passed": bool(passed),
                    "observed": f"{prediction}; expected-label probability {probability:.1%}", "examples": 1})
            else:
                where = test["where"]
                if where["feature"] not in reference:
                    raise ValueError("Scenario feature is absent from the fixed evaluation data")
                mask = OPERATORS[where["operator"]](reference[where["feature"]], where["value"])
                subset = reference.loc[mask]
                if subset.empty:
                    results.append({**base, "passed": None, "observed": "No held-out examples match", "examples": 0})
                    continue
                rate = float((model.predict(subset, config.positive_label, config.threshold) == test["expected"]).mean())
                results.append({**base, "passed": rate >= test.get("min_rate", 1),
                    "observed": f"Expected-label prediction rate {rate:.1%}", "examples": len(subset)})
        except (ValueError, TypeError) as exc:
            results.append({**base, "passed": False, "observed": str(exc), "examples": 0})
    return results
