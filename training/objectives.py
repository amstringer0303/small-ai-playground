import numpy as np


def sample_weights(labels, positive_label, penalty, balance):
    labels = np.asarray(labels)
    weights = np.ones(len(labels), dtype=float)
    if balance == "class_weight":
        for label in np.unique(labels):
            mask = labels == label
            weights[mask] *= len(labels) / (len(np.unique(labels)) * mask.sum())
    weights[labels == positive_label] *= penalty
    return weights


def oversample(frame, label_column, seed):
    rng = np.random.default_rng(seed)
    groups = [group for _, group in frame.groupby(label_column, sort=True)]
    largest = max(map(len, groups))
    indices = np.concatenate([np.concatenate([group.index.to_numpy(),
        rng.choice(group.index.to_numpy(), largest - len(group), replace=True)]) for group in groups])
    rng.shuffle(indices)
    return frame.loc[indices].reset_index(drop=True)
