
# STEP 37: TRAINING DATA PREPARATION

import pandas as pd


def prepare_training_data(
    dataframe,
    target_column,
    common_numeric,
    common_categorical,
    dataset_specific_numeric,
    dataset_specific_categorical,
    dataset_name="Dataset"
):
    print(f"\n========== STEP 37: {dataset_name} DATA PREPARATION ==========")

    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    # Collect all required input features
    required_features = (
        common_numeric
        + common_categorical
        + dataset_specific_numeric
        + dataset_specific_categorical
    )

    # Check missing feature columns
    missing = [
        column for column in required_features
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(f"Missing required features: {missing}")

    # Separate model inputs
    shared_numeric = dataframe[common_numeric].copy()
    shared_categorical = dataframe[common_categorical].copy()

    specific_numeric = dataframe[dataset_specific_numeric].copy()
    specific_categorical = dataframe[dataset_specific_categorical].copy()

    # Separate target labels
    labels = dataframe[target_column].copy()

    # Convert labels to integer format
    labels = labels.astype("int64")

    print("Dataset:", dataset_name)
    print("Total records:", len(dataframe))
    print("Shared numeric features:", len(common_numeric))
    print("Shared categorical features:", len(common_categorical))
    print("Dataset-specific numeric features:", len(dataset_specific_numeric))
    print("Dataset-specific categorical features:", len(dataset_specific_categorical))

    print("Label distribution:")
    print(labels.value_counts().sort_index().to_dict())

    return {
        "shared_numeric": shared_numeric,
        "shared_categorical": shared_categorical,
        "specific_numeric": specific_numeric,
        "specific_categorical": specific_categorical,
        "labels": labels
    }
