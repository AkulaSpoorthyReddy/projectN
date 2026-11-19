
# STEP 12: DATASET LOADER
# Supports CSV and Parquet files, including large CIC-DDoS2019 files.

from pathlib import Path
import pandas as pd


def load_dataset_file(file_path, chunksize=None):
    """
    Load a dataset from CSV or Parquet format.

    Args:
        file_path: Dataset file location.
        chunksize: Optional number of CSV rows per chunk.
                   Useful for large files.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    if file_path.stat().st_size == 0:
        raise ValueError("Dataset file is empty.")

    extension = file_path.suffix.lower()

    if extension == ".csv":
        if chunksize is not None:
            return pd.read_csv(
                file_path,
                chunksize=chunksize,
                low_memory=False
            )

        return pd.read_csv(
            file_path,
            low_memory=False
        )

    elif extension in [".parquet", ".pq"]:
        if chunksize is not None:
            raise ValueError(
                "Chunksize is currently supported only for CSV."
            )

        return pd.read_parquet(file_path)

    else:
        raise ValueError(
            f"Unsupported file format: {extension}"
        )


def inspect_dataset(dataframe, dataset_name="Dataset"):
    """Display basic information about a loaded dataset."""

    print(f"\n========== {dataset_name} INSPECTION ==========")

    print("Rows:", dataframe.shape[0])
    print("Columns:", dataframe.shape[1])

    print("\nColumn names:")
    print(dataframe.columns.tolist())

    print("\nMissing values:")
    print(dataframe.isnull().sum().sum())

    print("\nFirst five rows:")
    print(dataframe.head())

    print("\n===============================================")
