import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score


def evaluate_metrics(expected, predicted, classes, positive_label):
    expected, predicted = np.asarray(expected), np.asarray(predicted)
    positive_true = expected == positive_label
    positive_predicted = predicted == positive_label
    tp = int((positive_true & positive_predicted).sum())
    fn = int((positive_true & ~positive_predicted).sum())
    fp = int((~positive_true & positive_predicted).sum())
    tn = int((~positive_true & ~positive_predicted).sum())
    return {"accuracy": float(accuracy_score(expected, predicted)),
            "precision": float(precision_score(positive_true, positive_predicted, zero_division=0)),
            "recall": float(recall_score(positive_true, positive_predicted, zero_division=0)),
            "false_positives": fp, "false_negatives": fn, "true_positives": tp, "true_negatives": tn,
            "false_negative_rate": fn / (fn + tp) if fn + tp else None,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None,
            "confusion_matrix": confusion_matrix(expected, predicted, labels=classes).tolist(),
            "classes": classes, "test_examples": len(expected)}


def subgroup_metrics(reference, predicted, positive_label):
    rows = []
    for feature in ["site", "sensor", "unreliable"]:
        if feature not in reference:
            continue
        for name, group in reference.groupby(feature):
            indices = group.index.to_numpy()
            metrics = evaluate_metrics(group.condition, np.asarray(predicted)[indices],
                                       sorted(set(reference.condition) | set(predicted)), positive_label)
            rows.append({"feature": feature, "group": str(name), "examples": len(group),
                         "accuracy": metrics["accuracy"], "false_negative_rate": metrics["false_negative_rate"],
                         "false_positives": metrics["false_positives"], "false_negatives": metrics["false_negatives"]})
    return rows
