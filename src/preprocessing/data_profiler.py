
# STEP 24: AUTOMATED DATA PROFILING

import pandas as pd


def profile_dataset(dataframe, dataset_name="Dataset", exclude_columns=None):

    exclude_columns = set(exclude_columns or [])

    print(f"\n========== {dataset_name} PROFILE ==========")
    print("Total records:", len(dataframe))
    print("Total columns:", len(dataframe.columns))

    profile = {}

    for column in dataframe.columns:

        if column in exclude_columns:
            continue

        series = dataframe[column]

        dtype = str(series.dtype)
        missing = int(series.isna().sum())
        unique = int(series.nunique(dropna=True))

        details = {
            "name": column,
            "dtype": dtype,
            "missing_count": missing,
            "missing_percentage": round(
                missing / max(len(dataframe), 1) * 100, 2
            ),
            "unique_values": unique,
            "semantic_type": (
                "numeric" if pd.api.types.is_numeric_dtype(series)
                else "categorical"
            )
        }

        if pd.api.types.is_numeric_dtype(series):

            details["minimum"] = (
                float(series.min()) if series.notna().any() else None
            )
            details["maximum"] = (
                float(series.max()) if series.notna().any() else None
            )
            details["mean"] = (
                float(series.mean()) if series.notna().any() else None
            )
            details["std"] = (
                float(series.std()) if series.notna().any() else None
            )

        else:
            details["sample_values"] = (
                series.dropna().astype(str).unique()[:10].tolist()
            )

        profile[column] = details

    print("Profiled features:", len(profile))
    print("Duplicate rows:", int(dataframe.duplicated().sum()))

    return profile
