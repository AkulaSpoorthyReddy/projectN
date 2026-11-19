
# STEP 46L: DOWNLOAD UNSW-NB15 AND NSL-KDD TRAINING DATA

from pathlib import Path
from src.data.source_manager import (
    load_dataset_sources,
    download_csv
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_training_datasets():

    print("\n========== STEP 46L: DATASET DOWNLOAD ==========")

    sources = load_dataset_sources()

    datasets = {
        "UNSW_NB15_training-set.csv": (
            sources["UNSW-NB15"]["training"]
        ),
        "UNSW_NB15_testing-set.csv": (
            sources["UNSW-NB15"]["testing"]
        ),
        "KDDTrain+.csv": (
            sources["NSL-KDD"]["training"]
        ),
        "KDDTest+.csv": (
            sources["NSL-KDD"]["testing"]
        )
    }

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in datasets.items():

        destination = RAW_DATA_DIR / filename

        if destination.exists() and destination.stat().st_size > 0:
            print(f"\nAlready downloaded: {filename}")
            continue

        print(f"\nDownloading: {filename}")

        download_csv(url, destination)

        print(f"Saved: {destination}")
        print(f"Size: {destination.stat().st_size / (1024**2):.2f} MB")

    print("\nAll four dataset files are ready!")


if __name__ == "__main__":
    download_training_datasets()
