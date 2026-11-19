
# STEP 31: AUTOMATED FEATURE RELEVANCE AND REDUNDANCY ANALYSIS

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif


def analyze_feature_selection(
    dataframe,
    target_column,
    correlation_threshold=0.95,
    random_state=42
):
    print("\n========== STEP 31: FEATURE SELECTION ==========")

    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    X = dataframe.drop(columns=[target_column]).copy()
    y = dataframe[target_column].copy()

    # Separate numeric and categorical features
    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_columns = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    # Encode categorical features for relevance analysis
    X_encoded = X.copy()

    for column in categorical_columns:
        X_encoded[column] = (
            X_encoded[column]
            .astype("category")
            .cat.codes
        )

    # Handle missing values for analysis
    X_encoded = X_encoded.replace([np.inf, -np.inf], np.nan)
    X_encoded = X_encoded.fillna(X_encoded.median())

    # Encode target if categorical
    if not pd.api.types.is_numeric_dtype(y):
        y = y.astype("category").cat.codes

    # Calculate feature relevance
    scores = mutual_info_classif(
        X_encoded,
        y,
        random_state=random_state
    )

    relevance = pd.DataFrame({
        "feature": X_encoded.columns,
        "relevance_score": scores
    }).sort_values(
        "relevance_score",
        ascending=False
    )

    # Detect highly correlated numeric features
    correlation = X[numeric_columns].corr().abs()

    redundant_pairs = []

    for i in range(len(numeric_columns)):
        for j in range(i + 1, len(numeric_columns)):
            value = correlation.iloc[i, j]

            if pd.notna(value) and value >= correlation_threshold:
                redundant_pairs.append({
                    "feature_a": numeric_columns[i],
                    "feature_b": numeric_columns[j],
                    "correlation": round(float(value), 4)
                })

    print("\nTop 15 relevant features:")
    print(relevance.head(15).to_string(index=False))

    print("\nHighly correlated feature pairs:")
    print(pd.DataFrame(redundant_pairs).to_string(index=False)
          if redundant_pairs else "No highly correlated pairs found.")

    return {
        "relevance": relevance,
        "redundant_pairs": redundant_pairs,
        "all_features": X.columns.tolist()
    }
