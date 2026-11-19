
# STEP 10: COMBINED CLASSIFICATION AND ALIGNMENT LOSS

import torch
import torch.nn as nn

from src.training.alignment import selective_alignment_loss


def combined_training_loss(
    unsw_outputs,
    unsw_labels,
    nsl_outputs,
    nsl_labels,
    alignment_weight=0.1
):
    """
    Combine classification losses from both datasets
    with selective shared-feature alignment loss.

    Model output format:
        logits, shared_tokens, gates
    """

    criterion = nn.CrossEntropyLoss()

    # Extract model outputs
    unsw_logits, unsw_shared, _ = unsw_outputs
    nsl_logits, nsl_shared, _ = nsl_outputs

    # Classification losses
    unsw_classification = criterion(
        unsw_logits,
        unsw_labels.long()
    )

    nsl_classification = criterion(
        nsl_logits,
        nsl_labels.long()
    )

    classification_loss = (
        unsw_classification + nsl_classification
    ) / 2

    # Selective alignment loss
    alignment_loss = selective_alignment_loss(
        unsw_shared,
        nsl_shared
    )

    # Combined objective
    total_loss = (
        classification_loss
        + alignment_weight * alignment_loss
    )

    return total_loss, {
        "classification_loss": classification_loss,
        "unsw_classification_loss": unsw_classification,
        "nsl_classification_loss": nsl_classification,
        "alignment_loss": alignment_loss
    }

