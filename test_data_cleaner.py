
# STEP 34: TEST AUTOMATED DATA CLEANING

import pandas as pd
import numpy as np

from src.preprocessing.data_cleaner import clean_dataset

# Create sample network traffic data
data = pd.DataFrame({
    "duration": [1.0, 2.0, np.nan, 4.0, 4.0],
    "source_bytes": [100, 200, 300, 400, 400],
    "protocol": ["tcp", "udp", "tcp", None, None],
    "attack": [0, 0, 1, 1, 1]
})

print("Original dataset:")
print(data)

# Clean the dataset
cleaned = clean_dataset(data, "Sample Network Dataset")

print("\nCleaned dataset:")
print(cleaned)

# Verification
assert cleaned.isna().sum().sum() == 0
assert cleaned.duplicated().sum() == 0
assert len(cleaned) == 4

print("\n========== STEP 34: VERIFICATION ==========")
print("Data cleaning test passed!")
