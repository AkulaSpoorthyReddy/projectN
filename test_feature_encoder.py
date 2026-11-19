
# STEP 40: TEST FEATURE ENCODING

import pandas as pd

from src.preprocessing.feature_encoder import FeatureEncoder

# Sample training data
train_data = pd.DataFrame({
    "duration": [1.0, 2.0, 3.0, 4.0],
    "source_bytes": [100, 200, 300, 400],
    "protocol": ["tcp", "udp", "tcp", "udp"],
    "service": ["http", "dns", "ftp", "http"]
})

# Sample testing data with an unseen category
test_data = pd.DataFrame({
    "duration": [5.0, 6.0],
    "source_bytes": [500, 600],
    "protocol": ["tcp", "icmp"],
    "service": ["dns", "unknown_service"]
})

# Fit encoder using training data only
encoder = FeatureEncoder()

train_encoded = encoder.fit_transform(
    train_data,
    "UNSW"
)

# Transform test data using existing encoders
test_encoded = encoder.transform(test_data)

# Verification
assert train_encoded.shape == (4, 4)
assert test_encoded.shape == (2, 4)
assert test_encoded.isna().sum().sum() == 0

print("\n========== STEP 40: VERIFICATION ==========")
print("Training encoded shape:", train_encoded.shape)
print("Testing encoded shape:", test_encoded.shape)
print("Unseen categories handled successfully!")
print("Feature encoding test passed!")
