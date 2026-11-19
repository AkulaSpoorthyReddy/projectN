
# STEP 43: CONVERT PREPARED DATA INTO PYTORCH TENSORS

import torch
import numpy as np


def dataframe_to_tensor(dataframe, device="cpu"):
    if dataframe.empty:
        raise ValueError("Cannot convert an empty dataframe.")

    values = dataframe.to_numpy(dtype=np.float32, copy=True)

    if not np.isfinite(values).all():
        raise ValueError("Input contains NaN or infinite values.")

    return torch.tensor(
        values,
        dtype=torch.float32,
        device=device
    )


def labels_to_tensor(labels, device="cpu"):
    values = np.asarray(labels)

    if values.ndim != 1:
        raise ValueError("Labels must be one-dimensional.")

    if not np.isin(values, [0, 1]).all():
        raise ValueError("Labels must contain only 0 and 1.")

    return torch.tensor(
        values,
        dtype=torch.long,
        device=device
    )


def prepare_model_tensors(prepared_data, device="cpu"):
    print("\n========== STEP 43: MODEL TENSOR PREPARATION ==========")

    tensors = {
        "shared_numeric": dataframe_to_tensor(
            prepared_data["shared_numeric"], device
        ),
        "shared_categorical": dataframe_to_tensor(
            prepared_data["shared_categorical"], device
        ),
        "specific_numeric": dataframe_to_tensor(
            prepared_data["specific_numeric"], device
        ),
        "specific_categorical": dataframe_to_tensor(
            prepared_data["specific_categorical"], device
        ),
        "labels": labels_to_tensor(
            prepared_data["labels"], device
        )
    }

    for name, tensor in tensors.items():
        print(f"{name}: {tuple(tensor.shape)}")

    return tensors
