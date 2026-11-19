
# STEP 8: SELECTIVE ALIGNMENT LOSS
# Align shared numerical feature representations across UNSW and NSL-KDD.

import torch


def rbf_kernel(x, y, gamma=None):
    """Calculate the RBF kernel between two batches."""
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("Kernel inputs must be 2D tensors.")

    if x.shape[1] != y.shape[1]:
        raise ValueError("Feature dimensions must match.")

    if gamma is None:
        gamma = 1.0 / x.shape[1]

    distances = torch.cdist(x.float(), y.float(), p=2).pow(2)

    return torch.exp(-gamma * distances)


def mmd_loss(x, y, gamma=None):
    """Calculate biased RBF Maximum Mean Discrepancy."""
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("MMD inputs must be 2D tensors.")

    if x.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("MMD inputs cannot be empty.")

    if x.shape[1] != y.shape[1]:
        raise ValueError("Feature dimensions must match.")

    k_xx = rbf_kernel(x, x, gamma).mean()
    k_yy = rbf_kernel(y, y, gamma).mean()
    k_xy = rbf_kernel(x, y, gamma).mean()

    loss = k_xx + k_yy - (2.0 * k_xy)

    return loss.clamp_min(0.0)


def selective_alignment_loss(
    unsw_shared_tokens,
    nsl_shared_tokens,
    feature_indices=(0, 1, 2),
    gamma=None
):
    """
    Align only the selected shared numerical feature tokens.

    Expected input shape:
        [batch_size, number_of_shared_tokens, embedding_dim]

    Token order:
        0 = duration
        1 = source_bytes
        2 = destination_bytes
    """

    if unsw_shared_tokens.ndim != 3:
        raise ValueError("UNSW tokens must be a 3D tensor.")

    if nsl_shared_tokens.ndim != 3:
        raise ValueError("NSL tokens must be a 3D tensor.")

    if unsw_shared_tokens.shape[2] != nsl_shared_tokens.shape[2]:
        raise ValueError("Embedding dimensions must match.")

    total_loss = unsw_shared_tokens.new_zeros(())

    for index in feature_indices:
        if index >= unsw_shared_tokens.shape[1]:
            raise ValueError(f"UNSW is missing token index {index}.")

        if index >= nsl_shared_tokens.shape[1]:
            raise ValueError(f"NSL is missing token index {index}.")

        unsw_feature = unsw_shared_tokens[:, index, :]
        nsl_feature = nsl_shared_tokens[:, index, :]

        total_loss = total_loss + mmd_loss(
            unsw_feature,
            nsl_feature,
            gamma
        )

    return total_loss / len(feature_indices)
