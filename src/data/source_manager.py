
# STEP 16: DATASET SOURCE MANAGER

import json
import hashlib
from pathlib import Path
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "dataset_sources.json"


def load_dataset_sources():
    """Load dataset download information from the project config."""

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def download_csv(url, destination):
    """Download a CSV file in chunks and verify that it is not empty."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    request = Request(
        url,
        headers={"User-Agent": "NOVEL-IDS-V2"}
    )

    with urlopen(request, timeout=60) as response:
        with open(destination, "wb") as file:
            while True:
                chunk = response.read(1024 * 1024)

                if not chunk:
                    break

                file.write(chunk)

    if not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError(f"Download failed: {destination.name}")

    return destination


def calculate_file_hash(file_path):
    """Calculate SHA-256 hash for file verification."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def get_cic_dataset():
    """Download/access CIC-DDoS2019 using KaggleHub."""

    import kagglehub

    sources = load_dataset_sources()
    dataset_id = sources["CIC-DDoS2019"]["kaggle_dataset"]

    return Path(kagglehub.dataset_download(dataset_id))


if __name__ == "__main__":
    sources = load_dataset_sources()

    print("========== STEP 16: DATASET SOURCES ==========")

    for dataset_name, details in sources.items():
        print(f"\n{dataset_name}")

        for file_type, source in details.items():
            print(f"{file_type}: {source}")
