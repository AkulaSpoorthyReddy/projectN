
# STEP 46D: MODEL FORWARD PASS VERIFICATION

import torch
from src.models.novel_ids_model import NOVELIDSModel

print("========== STEP 46D: MODEL FORWARD TEST ==========")

# Temporary vocabulary sizes and feature counts for testing
model = NOVELIDSModel(
    categorical_vocab_sizes={
        "protocol": 10,
        "service": 10
    },
    unsw_specific_numeric_features=4,
    nsl_specific_numeric_features=3,
    unsw_category_count=6,
    nsl_category_count=7
)

batch_size = 8

# Shared inputs: 3 numerical + 2 categorical features
shared_numeric = torch.randn(batch_size, 3)
shared_categorical = torch.randint(0, 10, (batch_size, 2))

# UNSW inputs
unsw_specific_numeric = torch.randn(batch_size, 4)
unsw_specific_categorical = torch.randint(0, 6, (batch_size, 1))

# NSL inputs
nsl_specific_numeric = torch.randn(batch_size, 3)
nsl_specific_categorical = torch.randint(0, 7, (batch_size, 1))

model.eval()

with torch.no_grad():

    unsw_output = model(
        shared_numeric,
        shared_categorical,
        unsw_specific_numeric,
        unsw_specific_categorical,
        "UNSW"
    )

    nsl_output = model(
        shared_numeric,
        shared_categorical,
        nsl_specific_numeric,
        nsl_specific_categorical,
        "NSL"
    )

print("\nUNSW logits shape:", unsw_output[0].shape)
print("UNSW shared tokens:", unsw_output[1].shape)
print("UNSW gates:", unsw_output[2].shape)

print("\nNSL logits shape:", nsl_output[0].shape)
print("NSL shared tokens:", nsl_output[1].shape)
print("NSL gates:", nsl_output[2].shape)

print("\nTotal parameters:", sum(p.numel() for p in model.parameters()))

print("\nModel forward tests completed successfully!")
