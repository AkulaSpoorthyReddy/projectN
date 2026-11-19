
# STEP 33: AUTOMATED DATA CLEANING

import pandas as pd
import numpy as np


def clean_dataset(dataframe, dataset_name="Dataset"):
    print(f"\n========== STEP 33: CLEANING {dataset_name} ==========")

    cleaned = dataframe.copy()

    # Replace infinite values with missing values
    cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Remove duplicate records
    duplicates = int(cleaned.duplicated().sum())
    cleaned.drop_duplicates(inplace=True)

    # Separate numeric and categorical columns
    numeric_columns = cleaned.select_dtypes(
        include=np.number
    ).columns

    categorical_columns = cleaned.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    # Fill missing numerical values with median
    for column in numeric_columns:
        if cleaned[column].isna().any():
            median_value = cleaned[column].median()
            cleaned[column] = cleaned[column].fillna(median_value)

    # Fill missing categorical values with mode
    for column in categorical_columns:
        if cleaned[column].isna().any():
            mode_values = cleaned[column].mode(dropna=True)

            fill_value = (
                mode_values.iloc[0]
                if not mode_values.empty
                else "unknown"
            )

            cleaned[column] = cleaned[column].fillna(fill_value)

    print("Original rows:", len(dataframe))
    print("Removed duplicates:", duplicates)
    print("Remaining rows:", len(cleaned))
    print("Remaining missing values:", int(cleaned.isna().sum().sum()))

    return cleaned
