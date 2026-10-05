import pytest

from evaluation.custom_tests import validate_tests
from evaluation.metrics import evaluate_metrics
from models.base import decisions


def test_binary_error_rates_are_not_interchanged():
    result = evaluate_metrics(["alert", "alert", "normal", "normal"],
                              ["alert", "normal", "normal", "normal"], ["alert", "normal"], "alert")
    assert result["false_negatives"] == 1
    assert result["false_positives"] == 0
    assert result["false_negative_rate"] == 0.5
    assert result["precision"] == 1


def test_no_positive_examples_is_not_reported_as_zero_misses():
    result = evaluate_metrics(["normal"], ["normal"], ["alert", "normal"], "alert")
    assert result["false_negative_rate"] is None


def test_multiclass_threshold_uses_best_non_alert_category():
    prediction = decisions([[0.2, 0.1, 0.7]], ["alert", "normal", "uncertain"], "alert", 0.5)
    assert prediction[0] == "uncertain"


def test_ambiguous_custom_test_is_rejected():
    with pytest.raises(ValueError, match="either"):
        validate_tests([{"name": "Ambiguous", "expected": "alert", "features": {}, "where": {}}])
