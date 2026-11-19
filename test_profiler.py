
# STEP 25: TEST AUTOMATED DATA PROFILING

import pandas as pd

from src.preprocessing.data_profiler import profile_dataset

# Small sample dataset for testing
sample_data = pd.DataFrame({
    "duration": [0.5, 1.2, 2.5, None, 3.0],
    "source_bytes": [120, 250, 300, 180, 500],
    "protocol": ["tcp", "udp", "tcp", "tcp", "udp"],
    "service": ["http", "dns", "http", None, "ftp"]
})

profile = profile_dataset(
    sample_data,
    dataset_name="Sample Network Dataset"
)

print("\n========== STEP 25: PROFILE RESULTS ==========")

for feature, details in profile.items():
    print(f"\nFeature: {feature}")
    print(details)
