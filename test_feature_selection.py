
# STEP 32: TEST AUTOMATED FEATURE SELECTION

import pandas as pd

from src.preprocessing.feature_selection import (
    analyze_feature_selection
)

# Create sample network traffic data
data = pd.DataFrame({
    "duration": [1, 2, 3, 4, 5, 6, 7, 8],
    "source_bytes": [100, 200, 300, 400, 500, 600, 700, 800],
    "destination_bytes": [50, 100, 150, 200, 250, 300, 350, 400],
    "protocol": ["tcp", "udp", "tcp", "udp", "tcp", "udp", "tcp", "udp"],
    "packet_count": [10, 20, 30, 40, 50, 60, 70, 80],
    "attack": [0, 0, 0, 0, 1, 1, 1, 1]
})

# Run feature selection analysis
results = analyze_feature_selection(
    dataframe=data,
    target_column="attack"
)

# Verify results
assert len(results["all_features"]) == 5
assert not results["relevance"].empty

print("\n========== STEP 32: VERIFICATION ==========")
print("Feature selection test passed!")
print("Total features analyzed:", len(results["all_features"]))
