
# STEP 36: TEST INTEGRATED PREPROCESSING PIPELINE

import pandas as pd
import numpy as np

from src.preprocessing.preprocessing_pipeline import (
    preprocess_training_dataset
)

# Create sample network traffic dataset
data = pd.DataFrame({
    "duration": [1, 2, 3, 4, 5, 6, 7, 8],
    "source_bytes": [100, 200, 300, 400, 500, 600, 700, 800],
    "destination_bytes": [50, 100, 150, 200, 250, 300, 350, 400],
    "protocol": ["tcp", "udp", "tcp", "udp", "tcp", "udp", "tcp", "udp"],
    "packet_count": [10, 20, 30, 40, 50, 60, 70, 80],
    "service": ["http", "dns", "http", "ftp", "http", "dns", "ftp", "http"],
    "attack": [0, 0, 0, 0, 1, 1, 1, 1]
})

# Introduce a missing value
data.loc[2, "duration"] = np.nan

# Run complete preprocessing pipeline
results = preprocess_training_dataset(
    dataframe=data,
    dataset_name="Sample UNSW",
    target_column="attack"
)

cleaned_data = results["cleaned_data"]

# Verification
assert cleaned_data.isna().sum().sum() == 0
assert "attack" in cleaned_data.columns
assert "duration" in results["profile"]

print("\n========== STEP 36: VERIFICATION ==========")
print("Integrated preprocessing test passed!")
print("Cleaned dataset shape:", cleaned_data.shape)
print("Profiled features:", len(results["profile"]))
