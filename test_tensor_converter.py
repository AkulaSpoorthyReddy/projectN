
# STEP 44: TEST PYTORCH TENSOR CONVERSION

import pandas as pd
import torch

from src.data.tensor_converter import prepare_model_tensors

# Sample prepared data
prepared_data = {
    "shared_numeric": pd.DataFrame({
        "duration": [1.0, 2.0, 3.0],
        "source_bytes": [100.0, 200.0, 300.0]
    }),

    "shared_categorical": pd.DataFrame({
        "protocol": [0.0, 1.0, 0.0],
        "service": [1.0, 2.0, 3.0]
    }),

    "specific_numeric": pd.DataFrame({
        "ct_srv_src": [5.0, 6.0, 7.0]
    }),

    "specific_categorical": pd.DataFrame({
        "is_sm_ips_ports": [0.0, 1.0, 0.0]
    }),

    "labels": pd.Series([0, 1, 0])
}

# Convert into tensors
tensors = prepare_model_tensors(
    prepared_data,
    device="cpu"
)

# Verification
assert tensors["shared_numeric"].shape == (3, 2)
assert tensors["shared_categorical"].shape == (3, 2)
assert tensors["specific_numeric"].shape == (3, 1)
assert tensors["specific_categorical"].shape == (3, 1)
assert tensors["labels"].shape == (3,)

assert tensors["labels"].dtype == torch.long
assert tensors["shared_numeric"].dtype == torch.float32

print("\n========== STEP 44: VERIFICATION ==========")
print("PyTorch tensor conversion test passed!")
