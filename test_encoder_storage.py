
# STEP 42: TEST ENCODER SAVE AND LOAD

import pandas as pd

from src.preprocessing.feature_encoder import FeatureEncoder
from src.preprocessing.encoder_storage import save_encoder, load_encoder

# Sample training data
train_data = pd.DataFrame({
    "duration": [1.0, 2.0, 3.0, 4.0],
    "source_bytes": [100, 200, 300, 400],
    "protocol": ["tcp", "udp", "tcp", "udp"],
    "service": ["http", "dns", "ftp", "http"]
})

# Fit encoder
encoder = FeatureEncoder()

original_output = encoder.fit_transform(
    train_data,
    "UNSW"
)

# Save encoder
save_encoder(encoder, "UNSW")

# Load encoder
loaded_encoder = load_encoder("UNSW")

# Transform the same data using loaded encoder
loaded_output = loaded_encoder.transform(train_data)

# Verify identical transformations
pd.testing.assert_frame_equal(
    original_output,
    loaded_output
)

print("\n========== STEP 42: VERIFICATION ==========")
print("Encoder save and load test passed!")
print("Original and loaded outputs are identical.")
