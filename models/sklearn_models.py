from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text

from models.base import ModelAdapter


class SklearnAdapter(ModelAdapter):
    def __init__(self, model_id, config):
        if model_id == "logistic":
            self.estimator = LogisticRegression(C=config.regularization, max_iter=1000, random_state=config.seed)
        elif model_id == "tree":
            self.estimator = DecisionTreeClassifier(max_depth=config.tree_depth, random_state=config.seed)
        elif model_id == "forest":
            self.estimator = RandomForestClassifier(n_estimators=config.trees, max_depth=config.tree_depth,
                                                    random_state=config.seed, n_jobs=1)
        else:
            raise ValueError("Unsupported scikit-learn model.")

    def fit(self, x, y, sample_weight, config, validation=None):
        self.estimator.fit(x, y, sample_weight=sample_weight)
        return []

    def predict_proba(self, x):
        return self.estimator.predict_proba(x)

    def parameters(self):
        if hasattr(self.estimator, "coef_"):
            return {"coefficients": self.estimator.coef_.tolist(), "bias": self.estimator.intercept_.tolist()}
        result = {"feature_importances": self.estimator.feature_importances_.tolist()}
        if hasattr(self.estimator, "tree_"):
            result["tree_rules"] = export_text(self.estimator)
            result["nodes"] = int(self.estimator.tree_.node_count)
        else:
            result["nodes"] = sum(tree.tree_.node_count for tree in self.estimator.estimators_)
        return result
