
# STEP 1: NOVEL-IDS-V2 BACKEND
# Binary Normal/Attack detection + unified attack category prediction.

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent

ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "NOVEL-IDS-Results"
MULTICLASS_DIR = BASE_DIR / "multiclass_artifacts"

MODEL_FILE = MODEL_DIR / "best_model.pth"
PREPROCESS_FILE = MODEL_DIR / "preprocessing_artifacts.pkl"
MODEL_CODE = MODEL_DIR / "novel_ids_model.py"

ATTACK_MODEL_FILE = MULTICLASS_DIR / "unified_attack_classifier.pth"
ATTACK_CLASSES_FILE = MULTICLASS_DIR / "unified_attack_classes.json"

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from novel_ids_model import NOVELIDSModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512


# ---------------------------------------------------------
# STEP 1A: UNIFIED ATTACK CLASSIFIER ARCHITECTURE
# ---------------------------------------------------------

class UnifiedAttackClassifier(nn.Module):

    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# ---------------------------------------------------------
# STEP 1B: LOAD ALL TRAINED ASSETS
# ---------------------------------------------------------

def _load_assets():

    required_files = [
        MODEL_FILE,
        PREPROCESS_FILE,
        MODEL_CODE,
        ATTACK_MODEL_FILE,
        ATTACK_CLASSES_FILE
    ]

    missing = [
        str(path)
        for path in required_files
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Required model assets are missing:\n"
            + "\n".join(missing)
        )

    artifacts = joblib.load(PREPROCESS_FILE)

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=DEVICE,
        weights_only=False
    )

    config = checkpoint["model_config"]

    binary_model = NOVELIDSModel(**config).to(DEVICE)

    binary_model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    binary_model.eval()

    attack_checkpoint = torch.load(
        ATTACK_MODEL_FILE,
        map_location=DEVICE,
        weights_only=False
    )

    with open(ATTACK_CLASSES_FILE, "r") as f:
        attack_classes = json.load(f)

    attack_model = UnifiedAttackClassifier(
        input_dim=attack_checkpoint["input_dim"],
        num_classes=attack_checkpoint["num_classes"]
    ).to(DEVICE)

    attack_model.load_state_dict(
        attack_checkpoint["model_state_dict"]
    )

    attack_model.eval()

    if attack_classes != attack_checkpoint["class_names"]:
        raise ValueError(
            "Attack category names do not match the classifier checkpoint."
        )

    return (
        binary_model,
        attack_model,
        attack_classes,
        artifacts,
        checkpoint
    )


(
    _MODEL,
    _ATTACK_MODEL,
    _ATTACK_CLASSES,
    _A,
    _CHECKPOINT
) = _load_assets()


# ---------------------------------------------------------
# STEP 1C: NORMALIZE INPUT COLUMNS
# ---------------------------------------------------------

def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:

    out = df.copy()

    out.columns = [
        str(column).strip()
        for column in out.columns
    ]

    return out


# ---------------------------------------------------------
# STEP 1D: AUTOMATIC INPUT FORMAT DETECTION
# ---------------------------------------------------------

def _find_dataset(df: pd.DataFrame) -> str:

    cols = {column.lower() for column in df.columns}

    if {
        "protocol_type",
        "flag",
        "duration",
        "src_bytes",
        "dst_bytes"
    }.issubset(cols):

        return "NSL"

    if {
        "proto",
        "service",
        "state",
        "dur",
        "sbytes",
        "dbytes"
    }.issubset(cols):

        return "UNSW"

    if (
        "flow duration" in cols
        or "total length of fwd packets" in cols
    ):

        raise ValueError(
            "This appears to be CICFlowMeter/CIC-DDoS2019 data. "
            "The current model does not support this input format."
        )

    raise ValueError(
        "Unrecognized input format. Please upload a compatible "
        "UNSW-NB15 or NSL-KDD CSV file."
    )


# ---------------------------------------------------------
# STEP 1E: CATEGORICAL FEATURE ENCODING
# ---------------------------------------------------------

def _encode_category(
    series: pd.Series,
    vocab: dict,
    recovered: dict | None = None
):

    values = series.copy()

    if recovered:

        numeric = pd.to_numeric(
            values,
            errors="coerce"
        )

        mapped = numeric.map(recovered)

        values = mapped.where(
            mapped.notna(),
            values
        )

    values = (
        values.astype("string")
        .str.strip()
        .str.lower()
    )

    values = values.replace({
        "<na>": "",
        "nan": "",
        "none": ""
    })

    if vocab is _A["service_vocabulary"]:

        values = values.replace({
            "ftp_data": "ftp-data"
        })

    return (
        values.map(vocab)
        .fillna(0)
        .astype("int64")
    )


# ---------------------------------------------------------
# STEP 1F: UNSW FEATURE PREPROCESSING
# ---------------------------------------------------------

def _prepare_unsw(df: pd.DataFrame):

    a = _A

    required = (
        a["UNSW_SHARED_ALL"]
        + ["proto", "service", "state"]
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "UNSW file is missing required features: "
            + ", ".join(missing)
        )

    x = pd.DataFrame(index=df.index)

    x["protocol_id"] = _encode_category(
        df["proto"],
        a["protocol_vocabulary"],
        a["recovered_mappings"].get("proto")
    )

    x["service_id"] = _encode_category(
        df["service"],
        a["service_vocabulary"],
        a["recovered_mappings"].get("service")
    )

    x["state_id"] = _encode_category(
        df["state"],
        a["unsw_state_vocabulary"],
        a["recovered_mappings"].get("state")
    )

    # Handle different column names in the original UNSW file.
    aliases = {
        "Sload": "sload",
        "Dload": "dload",
        "Spkts": "spkts",
        "Dpkts": "dpkts",
        "smeansz": "smean",
        "dmeansz": "dmean",
        "res_bdy_len": "response_body_len",
        "Sjit": "sjit",
        "Djit": "djit",
        "Sintpkt": "sinpkt",
        "Dintpkt": "dinpkt"
    }

    column_lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    numeric = (
        a["UNSW_SHARED_ALL"]
        + a["UNSW_SPECIFIC_NUMERIC"]
    )

    for col in numeric:

        source_name = aliases.get(col, col)

        actual_column = column_lookup.get(
            source_name.lower()
        )

        if actual_column is not None:

            x[col] = pd.to_numeric(
                df[actual_column],
                errors="coerce"
            )

        else:

            # Missing sport/dsport are filled using
            # the corresponding training scaler center.
            x[col] = np.nan

    x[numeric] = x[numeric].replace(
        [np.inf, -np.inf],
        np.nan
    )

    for col in a["UNSW_LOG_COLUMNS"]:

        x[col] = np.log1p(
            x[col].clip(lower=0)
        )

    shared = a["UNSW_SHARED_ALL"]
    specific = a["UNSW_SPECIFIC_NUMERIC"]

    x[shared] = x[shared].fillna(
        pd.Series(
            a["shared_scaler"].center_,
            index=shared
        )
    )

    x[specific] = x[specific].fillna(
        pd.Series(
            a["unsw_specific_scaler"].center_,
            index=specific
        )
    )

    x[shared] = a["shared_scaler"].transform(
        x[shared]
    )

    x[specific] = a["unsw_specific_scaler"].transform(
        x[specific]
    )

    return x, "UNSW"


# ---------------------------------------------------------
# STEP 1G: NSL FEATURE PREPROCESSING
# ---------------------------------------------------------

def _prepare_nsl(df: pd.DataFrame):

    a = _A

    required = (
        a["NSL_SHARED_ALL"]
        + a["NSL_SPECIFIC_NUMERIC"]
        + ["protocol_type", "service", "flag"]
    )

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "NSL-KDD file is missing required features: "
            + ", ".join(missing)
        )

    x = pd.DataFrame(index=df.index)

    x["protocol_id"] = _encode_category(
        df["protocol_type"],
        a["protocol_vocabulary"]
    )

    x["service_id"] = _encode_category(
        df["service"],
        a["service_vocabulary"]
    )

    x["flag_id"] = _encode_category(
        df["flag"],
        a["nsl_flag_vocabulary"]
    )

    numeric = (
        a["NSL_SHARED_ALL"]
        + a["NSL_SPECIFIC_NUMERIC"]
    )

    for col in numeric:

        x[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    x[numeric] = x[numeric].replace(
        [np.inf, -np.inf],
        np.nan
    )

    for col in a["NSL_LOG_COLUMNS"]:

        x[col] = np.log1p(
            x[col].clip(lower=0)
        )

    shared = a["NSL_SHARED_ALL"]
    specific = a["NSL_SPECIFIC_NUMERIC"]

    x[shared] = x[shared].fillna(
        pd.Series(
            a["shared_scaler"].center_,
            index=shared
        )
    )

    x[specific] = x[specific].fillna(
        pd.Series(
            a["nsl_specific_scaler"].center_,
            index=specific
        )
    )

    x[shared] = a["shared_scaler"].transform(
        x[shared]
    )

    x[specific] = a["nsl_specific_scaler"].transform(
        x[specific]
    )

    return x, "NSL"


# ---------------------------------------------------------
# STEP 1H: BINARY NORMAL / ATTACK PREDICTION
# ---------------------------------------------------------

def _predict_binary(
    prepared: pd.DataFrame,
    dataset: str
):

    a = _A

    if dataset == "UNSW":

        shared_cols = a["UNSW_SHARED_ALL"]
        specific_cols = a["UNSW_SPECIFIC_NUMERIC"]
        category_col = "state_id"

    else:

        shared_cols = a["NSL_SHARED_ALL"]
        specific_cols = a["NSL_SPECIFIC_NUMERIC"]
        category_col = "flag_id"

    shared_num = torch.as_tensor(
        prepared[shared_cols].to_numpy(dtype=np.float32),
        device=DEVICE
    )

    shared_cat = torch.as_tensor(
        prepared[
            ["protocol_id", "service_id"]
        ].to_numpy(dtype=np.int64),
        device=DEVICE
    )

    specific_num = torch.as_tensor(
        prepared[specific_cols].to_numpy(dtype=np.float32),
        device=DEVICE
    )

    specific_cat = torch.as_tensor(
        prepared[[category_col]].to_numpy(dtype=np.int64),
        device=DEVICE
    )

    pred_parts = []
    prob_parts = []

    with torch.inference_mode():

        for start in range(0, len(prepared), BATCH_SIZE):

            end = start + BATCH_SIZE

            logits, _, _ = _MODEL(
                shared_num[start:end],
                shared_cat[start:end],
                specific_num[start:end],
                specific_cat[start:end],
                dataset=dataset
            )

            probs = torch.softmax(
                logits,
                dim=1
            )

            pred_parts.append(
                torch.argmax(
                    probs,
                    dim=1
                ).cpu().numpy()
            )

            prob_parts.append(
                probs.cpu().numpy()
            )

    return (
        np.concatenate(pred_parts),
        np.concatenate(prob_parts)
    )


# ---------------------------------------------------------
# STEP 1I: EXTRACT 64-D REPRESENTATIONS FOR ATTACK ROWS
# ---------------------------------------------------------

def _extract_attack_representations(
    prepared: pd.DataFrame,
    dataset: str,
    attack_indices: np.ndarray
):

    a = _A

    if dataset == "UNSW":

        shared_cols = a["UNSW_SHARED_ALL"]
        specific_cols = a["UNSW_SPECIFIC_NUMERIC"]
        category_col = "state_id"

    else:

        shared_cols = a["NSL_SHARED_ALL"]
        specific_cols = a["NSL_SPECIFIC_NUMERIC"]
        category_col = "flag_id"

    selected = prepared.iloc[attack_indices]

    shared_num = torch.as_tensor(
        selected[shared_cols].to_numpy(dtype=np.float32),
        device=DEVICE
    )

    shared_cat = torch.as_tensor(
        selected[
            ["protocol_id", "service_id"]
        ].to_numpy(dtype=np.int64),
        device=DEVICE
    )

    specific_num = torch.as_tensor(
        selected[specific_cols].to_numpy(dtype=np.float32),
        device=DEVICE
    )

    specific_cat = torch.as_tensor(
        selected[[category_col]].to_numpy(dtype=np.int64),
        device=DEVICE
    )

    representations = []

    with torch.inference_mode():

        for start in range(0, len(attack_indices), BATCH_SIZE):

            end = start + BATCH_SIZE

            # Same representation extraction pathway
            # used during unified classifier training.

            shared_tokens = _MODEL.shared_encoder(
                shared_num[start:end],
                shared_cat[start:end]
            )

            residual_tokens = _MODEL.adapters[dataset](
                specific_num[start:end],
                specific_cat[start:end]
            )

            fused_tokens, _ = _MODEL.fusions[dataset](
                shared_tokens,
                residual_tokens
            )

            combined_tokens = torch.cat(
                [shared_tokens, fused_tokens],
                dim=1
            )

            encoded = _MODEL.classifier.layer_norm(
                _MODEL.classifier.transformer(
                    combined_tokens
                )
            )

            pooled = encoded.mean(dim=1)

            representations.append(
                pooled.cpu().numpy()
            )

    return np.concatenate(representations, axis=0)


# ---------------------------------------------------------
# STEP 1J: UNIFIED ATTACK CATEGORY PREDICTION
# ---------------------------------------------------------

def _predict_attack_type(representations):

    attack_predictions = []

    with torch.inference_mode():

        for start in range(0, len(representations), BATCH_SIZE):

            end = start + BATCH_SIZE

            xb = torch.as_tensor(
                representations[start:end],
                dtype=torch.float32,
                device=DEVICE
            )

            logits = _ATTACK_MODEL(xb)

            predicted_indices = logits.argmax(
                dim=1
            ).cpu().numpy()

            attack_predictions.extend(
                [
                    _ATTACK_CLASSES[index]
                    for index in predicted_indices
                ]
            )

    return attack_predictions


# ---------------------------------------------------------
# STEP 1K: OPTIONAL GROUND-TRUTH READING
# ---------------------------------------------------------

def _ground_truth(df: pd.DataFrame, dataset: str):

    lower = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    if dataset == "UNSW":

        if "target" in lower:

            raw = pd.to_numeric(
                df[lower["target"]],
                errors="coerce"
            )

            if raw.notna().all() and set(
                raw.unique()
            ).issubset({0, 1}):

                return raw.astype(int).to_numpy()

            return (raw != 6).astype(int).to_numpy()

        if "label" in lower:

            raw = pd.to_numeric(
                df[lower["label"]],
                errors="coerce"
            )

            if raw.notna().all() and set(
                raw.unique()
            ).issubset({0, 1}):

                return raw.astype(int).to_numpy()

    else:

        if "label" in lower:

            return (
                df[lower["label"]]
                .astype(str)
                .str.strip()
                .str.lower()
                .ne("normal")
                .astype(int)
                .to_numpy()
            )

    return None


# ---------------------------------------------------------
# STEP 1L: MAIN FRONTEND FUNCTION
# ---------------------------------------------------------

def analyze_files(records: list[dict[str, Any]]):

    """
    Frontend contract:
    Accepts [{name, bytes, dataframe}]
    Returns (prediction_dataframe, metadata).
    """

    outputs = []

    all_true = []
    all_pred = []

    for record in records:

        name = str(
            record.get("name", "uploaded.csv")
        )

        df = _normalise_columns(
            record["dataframe"]
        )

        # Automatic format detection.
        dataset = _find_dataset(df)

        if dataset == "UNSW":
            prepared, dataset = _prepare_unsw(df)
        else:
            prepared, dataset = _prepare_nsl(df)

        # Stage 1: Binary prediction.
        predictions, probabilities = _predict_binary(
            prepared,
            dataset
        )

        # Initialize attack type as None.
        attack_types = np.full(
            len(df),
            "None",
            dtype=object
        )

        # Stage 2: Predict attack category only
        # for rows classified as Attack.
        attack_indices = np.where(
            predictions == 1
        )[0]

        if len(attack_indices) > 0:

            representations = _extract_attack_representations(
                prepared,
                dataset,
                attack_indices
            )

            predicted_attack_types = _predict_attack_type(
                representations
            )

            attack_types[attack_indices] = (
                predicted_attack_types
            )

        # Final prediction names.
        predicted_categories = np.where(
            predictions == 0,
            "Normal",
            "Attack"
        )

        result = pd.DataFrame({

            "source_file": name,

            "record_id": np.arange(len(df)),

            "predicted_category": predicted_categories,

            "prediction": predictions,

            "probability_normal": probabilities[:, 0],

            "probability_attack": probabilities[:, 1],

            "confidence_score": probabilities.max(axis=1),

            "attack_type": attack_types
        })

        outputs.append(result)

        # Ground truth is used only for optional evaluation.
        ground_truth = _ground_truth(
            df,
            dataset
        )

        if ground_truth is not None:

            all_true.extend(
                ground_truth.tolist()
            )

            all_pred.extend(
                predictions.tolist()
            )

    combined = (
        pd.concat(outputs, ignore_index=True)
        if outputs
        else pd.DataFrame()
    )

    metadata = {

        "device": str(DEVICE),

        "model_epoch": int(
            _CHECKPOINT.get("epoch", -1)
        ),

        "prediction_type": "Binary Normal / Attack",

        "attack_type_prediction": True,

        "attack_categories": len(_ATTACK_CLASSES),

        "ground_truth_evaluated": False,

        "evaluation_metrics": {}
    }

    if all_true and len(all_true) == len(all_pred):

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score
        )

        metadata["ground_truth_evaluated"] = True

        metadata["evaluation_metrics"] = {

            "accuracy": float(
                accuracy_score(all_true, all_pred)
            ),

            "precision": float(
                precision_score(
                    all_true,
                    all_pred,
                    zero_division=0
                )
            ),

            "recall": float(
                recall_score(
                    all_true,
                    all_pred,
                    zero_division=0
                )
            ),

            "f1_score": float(
                f1_score(
                    all_true,
                    all_pred,
                    zero_division=0
                )
            )
        }

    return combined, metadata
