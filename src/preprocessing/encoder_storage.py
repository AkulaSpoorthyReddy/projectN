
# STEP 41: SAVE AND LOAD FEATURE ENCODERS

import joblib
from pathlib import Path


def save_encoder(encoder, dataset_name):
    print(f"\n========== STEP 41: SAVE {dataset_name} ENCODER ==========")

    save_directory = Path("artifacts/preprocessing")
    save_directory.mkdir(parents=True, exist_ok=True)

    save_path = save_directory / f"{dataset_name.lower()}_encoder.pkl"

    joblib.dump(encoder, save_path)

    print("Encoder saved to:", save_path.resolve())

    return save_path


def load_encoder(dataset_name):
    print(f"\n========== STEP 41: LOAD {dataset_name} ENCODER ==========")

    save_path = (
        Path("artifacts/preprocessing")
        / f"{dataset_name.lower()}_encoder.pkl"
    )

    if not save_path.exists():
        raise FileNotFoundError(f"Encoder not found: {save_path}")

    encoder = joblib.load(save_path)

    print("Encoder loaded successfully.")

    return encoder
