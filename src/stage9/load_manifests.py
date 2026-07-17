from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("Stage9 Load Manifests")


def _validate_manifest(df, split_name):

    required_columns = [

        "pair_id",

        "audio_file",

        "feature_file",

        "feature_type",

        "training_text",

        "dataset_split"

    ]

    for col in required_columns:

        if col not in df.columns:

            return False, f"Missing required column: {col}"

    if len(df) == 0:

        return False, f"{split_name} manifest is empty"

    wrong_split = df[

        df["dataset_split"].astype(str).str.strip() != split_name

    ]

    if len(wrong_split) > 0:

        return False, f"Wrong dataset_split values in {split_name} manifest"

    return True, "Manifest validation passed"


def load_dataset_manifests(project_root: Path):

    logger.info("=" * 70)
    logger.info("Loading Dataset Manifests")
    logger.info("=" * 70)

    splits_folder = project_root / "data" / "dataset_splits"

    train_manifest = splits_folder / "train_manifest.csv"
    val_manifest = splits_folder / "val_manifest.csv"
    test_manifest = splits_folder / "test_manifest.csv"

    manifest_files = {

        "train": train_manifest,

        "val": val_manifest,

        "test": test_manifest

    }

    loaded_manifests = {}

    summary = {

        "train_loaded": False,

        "val_loaded": False,

        "test_loaded": False,

        "train_rows": 0,

        "val_rows": 0,

        "test_rows": 0,

        "total_rows": 0,

        "status": "Failed",

        "message": ""

    }

    for split_name, manifest_file in manifest_files.items():

        logger.info(f"Loading {split_name} manifest: {manifest_file}")

        if not manifest_file.exists():

            message = f"{split_name} manifest not found: {manifest_file}"

            logger.warning(message)

            summary["message"] = message

            return None, None, None, summary

        try:

            df = pd.read_csv(manifest_file)

        except Exception as e:

            message = f"Cannot read {split_name} manifest: {e}"

            logger.warning(message)

            summary["message"] = message

            return None, None, None, summary

        is_valid, validation_message = _validate_manifest(

            df,

            split_name

        )

        if not is_valid:

            logger.warning(validation_message)

            summary["message"] = validation_message

            return None, None, None, summary

        loaded_manifests[split_name] = df

        summary[f"{split_name}_loaded"] = True
        summary[f"{split_name}_rows"] = len(df)

        logger.info(
            f"{split_name} manifest loaded successfully: {len(df)} records"
        )

    summary["total_rows"] = (

        summary["train_rows"]

        + summary["val_rows"]

        + summary["test_rows"]

    )

    summary["status"] = "Loaded"

    summary["message"] = "All dataset manifests loaded successfully"

    logger.info(f"Train rows      : {summary['train_rows']}")
    logger.info(f"Validation rows : {summary['val_rows']}")
    logger.info(f"Test rows       : {summary['test_rows']}")
    logger.info(f"Total rows      : {summary['total_rows']}")

    logger.info("=" * 70)

    return (

        loaded_manifests["train"],

        loaded_manifests["val"],

        loaded_manifests["test"],

        summary

    )