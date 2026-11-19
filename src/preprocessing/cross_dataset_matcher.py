
# STEP 46I: REAL CROSS-DATASET FEATURE MATCHING

from src.preprocessing.preprocessing_pipeline import (
    preprocess_training_dataset
)

from src.preprocessing.semantic_matcher import (
    match_feature_profiles
)


def match_training_datasets(
    unsw_dataframe,
    nsl_dataframe,
    unsw_target,
    nsl_target
):
    print("\n========== STEP 46I: CROSS-DATASET MATCHING ==========")

    # STEP 46I-A: Profile actual UNSW training data
    unsw_result = preprocess_training_dataset(
        unsw_dataframe,
        "UNSW",
        unsw_target
    )

    # STEP 46I-B: Profile actual NSL-KDD training data
    nsl_result = preprocess_training_dataset(
        nsl_dataframe,
        "NSL-KDD",
        nsl_target
    )

    # STEP 46I-C: Match actual feature profiles
    matching = match_feature_profiles(
        unsw_result["profile"],
        nsl_result["profile"]
    )

    print("\n========== COMMON FEATURES ==========")
    for item in matching["common_features"]:
        print(
            f'{item["feature_a"]} <-> {item["feature_b"]} '
            f'| Confidence: {item["confidence"]}'
        )

    print("\n========== REVIEW CANDIDATES ==========")
    for item in matching["review_candidates"]:
        print(item)

    print("\n========== UNSW-SPECIFIC FEATURES ==========")
    print(matching["specific_features_a"])

    print("\n========== NSL-KDD-SPECIFIC FEATURES ==========")
    print(matching["specific_features_b"])

    return {
        "unsw": unsw_result,
        "nsl": nsl_result,
        "matching": matching
    }
