from pathlib import Path
import json

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("Stage10 Training Resources")


def _safe_text(value):

    if value is None:
        return ""

    value = str(value)

    if value.lower() == "nan":
        return ""

    return value.strip()


def _validate_manifest(df, split_name):

    required_columns = [

        "pair_id",

        "audio_file",

        "feature_file",

        "training_text",

        "dataset_split"

    ]

    if df is None:

        return False, "Manifest dataframe is None"

    if len(df) == 0:

        return False, f"{split_name} manifest is empty"

    for col in required_columns:

        if col not in df.columns:

            return False, f"Missing required column in {split_name}: {col}"

    wrong_split = df[

        df["dataset_split"].astype(str).str.strip() != split_name

    ]

    if len(wrong_split) > 0:

        return False, f"Wrong dataset_split values found in {split_name} manifest"

    empty_text = df[

        df["training_text"].fillna("").astype(str).str.strip() == ""

    ]

    if len(empty_text) > 0:

        return False, f"Empty training_text records found in {split_name}"

    empty_feature = df[

        df["feature_file"].fillna("").astype(str).str.strip() == ""

    ]

    if len(empty_feature) > 0:

        return False, f"Empty feature_file records found in {split_name}"

    return True, "Manifest validation passed"


def _load_manifest(manifest_file: Path, split_name: str):

    logger.info(f"Loading {split_name} manifest: {manifest_file}")

    result = {

        "split": split_name,

        "file": str(manifest_file),

        "exists": False,

        "rows": 0,

        "valid": False,

        "message": ""

    }

    if not manifest_file.exists():

        result["message"] = f"{split_name} manifest not found"

        logger.warning(result["message"])

        return None, result

    result["exists"] = True

    try:

        df = pd.read_csv(manifest_file)

    except Exception as e:

        result["message"] = f"Cannot read {split_name} manifest: {e}"

        logger.warning(result["message"])

        return None, result

    result["rows"] = len(df)

    is_valid, message = _validate_manifest(
        df,
        split_name
    )

    result["valid"] = is_valid
    result["message"] = message

    if is_valid:

        logger.info(f"{split_name} manifest loaded successfully: {len(df)} rows")

    else:

        logger.warning(f"{split_name} manifest validation failed: {message}")

    return df, result


def _validate_vocabulary(vocab):

    required_keys = [

        "label_type",

        "blank_token",

        "unknown_token",

        "blank_id",

        "unknown_id",

        "vocab_size",

        "char_to_id",

        "id_to_char"

    ]

    if vocab is None:

        return False, "Vocabulary is None"

    for key in required_keys:

        if key not in vocab:

            return False, f"Missing vocabulary key: {key}"

    if not isinstance(vocab["char_to_id"], dict):

        return False, "char_to_id must be a dictionary"

    if not isinstance(vocab["id_to_char"], dict):

        return False, "id_to_char must be a dictionary"

    if int(vocab["vocab_size"]) <= 0:

        return False, "Invalid vocabulary size"

    if "<blank>" not in vocab["char_to_id"]:

        return False, "Missing <blank> token in vocabulary"

    if "<unk>" not in vocab["char_to_id"]:

        return False, "Missing <unk> token in vocabulary"

    return True, "Vocabulary validation passed"


def _load_vocabulary(vocab_file: Path):

    logger.info(f"Loading character vocabulary: {vocab_file}")

    result = {

        "file": str(vocab_file),

        "exists": False,

        "valid": False,

        "vocab_size": 0,

        "message": ""

    }

    if not vocab_file.exists():

        result["message"] = "Vocabulary file not found"

        logger.warning(result["message"])

        return None, result

    result["exists"] = True

    try:

        with open(vocab_file, "r", encoding="utf-8") as f:

            vocab = json.load(f)

    except Exception as e:

        result["message"] = f"Cannot read vocabulary file: {e}"

        logger.warning(result["message"])

        return None, result

    is_valid, message = _validate_vocabulary(vocab)

    result["valid"] = is_valid
    result["message"] = message

    if is_valid:

        result["vocab_size"] = vocab.get("vocab_size", 0)

        logger.info(f"Vocabulary loaded successfully. Size: {result['vocab_size']}")

    else:

        logger.warning(f"Vocabulary validation failed: {message}")

    return vocab, result


def load_training_resources(project_root: Path):

    logger.info("=" * 70)
    logger.info("Loading Training Resources")
    logger.info("=" * 70)

    splits_folder = project_root / "data" / "dataset_splits"

    training_interface_folder = project_root / "data" / "training_interface"

    train_manifest_file = splits_folder / "train_manifest.csv"

    val_manifest_file = splits_folder / "val_manifest.csv"

    test_manifest_file = splits_folder / "test_manifest.csv"

    vocab_file = training_interface_folder / "character_vocabulary.json"

    # --------------------------------------------------
    # Load manifests
    # --------------------------------------------------

    train_df, train_result = _load_manifest(
        train_manifest_file,
        "train"
    )

    val_df, val_result = _load_manifest(
        val_manifest_file,
        "val"
    )

    test_df, test_result = _load_manifest(
        test_manifest_file,
        "test"
    )

    # --------------------------------------------------
    # Load vocabulary
    # --------------------------------------------------

    vocab, vocab_result = _load_vocabulary(
        vocab_file
    )

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------

    all_valid = (

        train_result["valid"]

        and val_result["valid"]

        and test_result["valid"]

        and vocab_result["valid"]

    )

    resource_summary = {

        "status": "Loaded" if all_valid else "Failed",

        "message": (
            "All training resources loaded successfully"
            if all_valid
            else "Some training resources failed to load"
        ),

        "train_rows": train_result["rows"],

        "val_rows": val_result["rows"],

        "test_rows": test_result["rows"],

        "total_rows": (
            train_result["rows"]
            + val_result["rows"]
            + test_result["rows"]
        ),

        "vocab_size": vocab_result["vocab_size"],

        "train_manifest": train_result,

        "val_manifest": val_result,

        "test_manifest": test_result,

        "vocabulary": vocab_result

    }

    training_resources = {

        "train_df": train_df,

        "val_df": val_df,

        "test_df": test_df,

        "vocab": vocab,

        "char_to_id": vocab.get("char_to_id", {}) if vocab else {},

        "id_to_char": vocab.get("id_to_char", {}) if vocab else {},

        "blank_id": vocab.get("blank_id", 0) if vocab else 0,

        "unknown_id": vocab.get("unknown_id", 1) if vocab else 1

    }

    logger.info(f"Train rows      : {resource_summary['train_rows']}")
    logger.info(f"Validation rows : {resource_summary['val_rows']}")
    logger.info(f"Test rows       : {resource_summary['test_rows']}")
    logger.info(f"Total rows      : {resource_summary['total_rows']}")
    logger.info(f"Vocabulary size : {resource_summary['vocab_size']}")
    logger.info(f"Resource status : {resource_summary['status']}")

    logger.info("=" * 70)

    return training_resources, resource_summary