import numpy as np
import pandas as pd
import torch


def cell_index(parameter, row, column):
    if parameter.ndim == 1:
        if column != 0 or not 0 <= row < parameter.shape[0]:
            raise ValueError("For a bias, use its row index and column 0.")
        return (row,)
    if parameter.ndim != 2 or not 0 <= row < parameter.shape[0] or not 0 <= column < parameter.shape[1]:
        raise ValueError("Weight row/column is outside this tensor.")
    return row, column


def apply_interventions(adapter, config):
    parameters = dict(adapter.network.named_parameters())
    handles = []
    unknown = (set(config.frozen_parameters) | set(config.reset_parameters)) - set(parameters)
    if unknown:
        raise ValueError(f"Unknown parameters: {', '.join(sorted(unknown))}")
    with torch.no_grad():
        for name in config.reset_parameters:
            parameters[name].copy_(torch.tensor(adapter.reset_state[name], dtype=parameters[name].dtype))
        for edit in config.edits:
            if edit["parameter"] not in parameters or not np.isfinite(edit["value"]):
                raise ValueError("Choose an existing parameter and a finite weight value.")
            parameter = parameters[edit["parameter"]]
            index = cell_index(parameter, int(edit["row"]), int(edit["column"]))
            parameter[index] = float(edit["value"])
    for name, parameter in parameters.items():
        parameter.requires_grad_(name not in config.frozen_parameters)
    for cell in config.frozen_cells:
        if cell["parameter"] not in parameters:
            raise ValueError("Choose an existing parameter to freeze.")
        parameter = parameters[cell["parameter"]]
        index = cell_index(parameter, int(cell["row"]), int(cell["column"]))
        if parameter.requires_grad:
            mask = torch.ones_like(parameter)
            mask[index] = 0
            handles.append(parameter.register_hook(lambda gradient, mask=mask: gradient * mask))
    return handles


def weight_table(before, after, parameter):
    if parameter not in after:
        return pd.DataFrame(columns=["row", "column", "before", "after", "change"])
    start = np.asarray(before[parameter])
    finish = np.asarray(after[parameter])
    if start.shape != finish.shape:
        raise ValueError("These checkpoints have different tensor shapes.")
    rows = []
    for index in np.ndindex(finish.shape):
        rows.append({"row": index[0], "column": index[1] if len(index) > 1 else 0,
                     "before": float(start[index]), "after": float(finish[index]),
                     "change": float(finish[index] - start[index])})
    return pd.DataFrame(rows)


def parameter_change(before, after):
    changes = {}
    for name in set(before) & set(after):
        left, right = np.asarray(before[name]), np.asarray(after[name])
        if left.shape == right.shape and left.dtype.kind in "fiu":
            changes[name] = float(np.linalg.norm(right - left))
    return changes
