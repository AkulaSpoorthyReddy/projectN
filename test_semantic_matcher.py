

# STEP 30: TEST COMMON AND DATASET-SPECIFIC FEATURES

from src.preprocessing.semantic_matcher import (
    match_feature_profiles
)

unsw_profile = {
    "dur": {"semantic_type": "numeric"},
    "sbytes": {"semantic_type": "numeric"},
    "dbytes": {"semantic_type": "numeric"},
    "proto": {"semantic_type": "categorical"},
    "service": {"semantic_type": "categorical"},
    "ct_srv_src": {"semantic_type": "numeric"},
    "is_sm_ips_ports": {"semantic_type": "numeric"}
}

nsl_profile = {
    "duration": {"semantic_type": "numeric"},
    "src_bytes": {"semantic_type": "numeric"},
    "dst_bytes": {"semantic_type": "numeric"},
    "protocol_type": {"semantic_type": "categorical"},
    "service": {"semantic_type": "categorical"},
    "wrong_fragment": {"semantic_type": "numeric"},
    "urgent": {"semantic_type": "numeric"}
}

result = match_feature_profiles(
    unsw_profile,
    nsl_profile
)

print("\n========== STEP 30: FEATURE SEPARATION ==========")

print("\nCOMMON FEATURES:")
for item in result["common_features"]:
    print(
        item["feature_a"],
        "<->",
        item["feature_b"],
        "| Confidence:",
        item["confidence"]
    )

print("\nREVIEW CANDIDATES:")
for item in result["review_candidates"]:
    print(item)

print("\nUNSW-SPECIFIC FEATURES:")
print(result["specific_features_a"])

print("\nNSL-KDD-SPECIFIC FEATURES:")
print(result["specific_features_b"])

print("\n========== VERIFICATION ==========")

assert len(result["common_features"]) == 5
assert "ct_srv_src" in result["specific_features_a"]
assert "wrong_fragment" in result["specific_features_b"]

print("All feature separation checks passed!")
