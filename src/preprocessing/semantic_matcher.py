

# STEP 29: CONFIDENCE-BASED FEATURE SEPARATION

from src.preprocessing.feature_harmonizer import (
    semantic_similarity
)


def match_feature_profiles(
    profile_a,
    profile_b,
    high_threshold=0.85,
    review_threshold=0.60
):
    candidates = []

    for name_a, details_a in profile_a.items():
        for name_b, details_b in profile_b.items():

            similarity = semantic_similarity(name_a, name_b)

            type_a = details_a["semantic_type"]
            type_b = details_b["semantic_type"]

            compatibility = (
                1.0 if type_a == type_b else 0.0
            )

            confidence = (
                0.80 * similarity
                + 0.20 * compatibility
            )

            if confidence >= review_threshold:
                candidates.append({
                    "feature_a": name_a,
                    "feature_b": name_b,
                    "confidence": round(confidence, 4)
                })

    candidates.sort(
        key=lambda x: x["confidence"],
        reverse=True
    )

    common = []
    review = []
    used_a = set()
    used_b = set()

    for item in candidates:

        a = item["feature_a"]
        b = item["feature_b"]

        if a in used_a or b in used_b:
            continue

        if item["confidence"] >= high_threshold:
            common.append(item)
            used_a.add(a)
            used_b.add(b)

        elif item["confidence"] >= review_threshold:
            review.append(item)

    return {
        "common_features": common,
        "review_candidates": review,
        "specific_features_a": [
            name for name in profile_a if name not in used_a
        ],
        "specific_features_b": [
            name for name in profile_b if name not in used_b
        ]
    }
