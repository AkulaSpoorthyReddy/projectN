
# STEP 38: TEST TRAINING DATA PREPARATION

import pandas as pd

from src.data.training_data_preparer import prepare_training_data

# Sample cleaned UNSW-like dataset
data = pd.DataFrame({
    "duration": [1, 2, 3, 4],
    "source_bytes": [100, 200, 300, 400],
    "protocol": ["tcp", "udp", "tcp", "udp"],
    "service": ["http", "dns", "ftp", "http"],
    "ct_srv_src": [5, 6, 7, 8],
    "is_sm_ips_ports": [0, 1, 0, 1],
    "attack": [0, 0, 1, 1]
})

# Prepare dataset inputs
results = prepare_training_data(
    dataframe=data,
    target_column="attack",
    common_numeric=["duration", "source_bytes"],
    common_categorical=["protocol", "service"],
    dataset_specific_numeric=["ct_srv_src"],
    dataset_specific_categorical=["is_sm_ips_ports"],
    dataset_name="UNSW"
)

# Verification
assert results["shared_numeric"].shape == (4, 2)
assert results["shared_categorical"].shape == (4, 2)
assert results["specific_numeric"].shape == (4, 1)
assert results["specific_categorical"].shape == (4, 1)
assert results["labels"].tolist() == [0, 0, 1, 1]

print("\n========== STEP 38: VERIFICATION ==========")
print("Training data preparation test passed!")
print("Shared numeric shape:", results["shared_numeric"].shape)
print("Shared categorical shape:", results["shared_categorical"].shape)
print("Specific numeric shape:", results["specific_numeric"].shape)
print("Specific categorical shape:", results["specific_categorical"].shape)
print("Labels shape:", results["labels"].shape)
