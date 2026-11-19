
# STEP 35: INTEGRATED PREPROCESSING PIPELINE

from src.preprocessing.data_profiler import profile_dataset
from src.preprocessing.feature_selection import analyze_feature_selection
from src.preprocessing.data_cleaner import clean_dataset


def preprocess_training_dataset(
    dataframe,
    dataset_name,
    target_column
):
    print(f"\n========== STEP 35: {dataset_name} PIPELINE ==========")

    # STEP 35A: Initial feature profiling
    print("\n--- Feature Profiling ---")

    profile = profile_dataset(
        dataframe,
        dataset_name,
        exclude_columns=[target_column]
    )

    # STEP 35B: Feature relevance and redundancy
    print("\n--- Feature Selection Analysis ---")

    selection = analyze_feature_selection(
        dataframe,
        target_column
    )

    # STEP 35C: Data cleaning
    print("\n--- Data Cleaning ---")

    cleaned_data = clean_dataset(
        dataframe,
        dataset_name
    )

    print("\nPreprocessing completed successfully.")

    return {
        "cleaned_data": cleaned_data,
        "profile": profile,
        "feature_selection": selection
    }
