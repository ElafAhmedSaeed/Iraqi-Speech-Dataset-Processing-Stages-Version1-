from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("Stage8 Split Validator")


def _safe_text(value):

    if value is None:
        return ""

    value = str(value)

    if value.lower() == "nan":
        return ""

    return value.strip()


def _validate_manifest_file(project_root: Path, manifest_file: Path, split_name: str):

    result = {

        "split": split_name,

        "manifest_file": str(manifest_file),

        "exists": False,

        "rows": 0,

        "valid": False,

        "missing_feature_files": 0,

        "empty_training_text": 0,

        "message": ""

    }

    if not manifest_file.exists():

        result["message"] = "Manifest file not found"

        return result

    result["exists"] = True

    try:

        df = pd.read_csv(manifest_file)

    except Exception as e:

        result["message"] = f"Cannot read manifest file: {e}"

        return result

    result["rows"] = len(df)

    if len(df) == 0:

        result["message"] = "Manifest file is empty"

        return result

    required_columns = [

        "pair_id",

        "audio_file",

        "feature_file",

        "training_text",

        "dataset_split"

    ]

    for col in required_columns:

        if col not in df.columns:

            result["message"] = f"Missing required column: {col}"

            return result

    missing_feature_files = 0
    empty_training_text = 0
    wrong_split = 0

    for _, row in df.iterrows():

        feature_file = _safe_text(row.get("feature_file", ""))

        training_text = _safe_text(row.get("training_text", ""))

        dataset_split = _safe_text(row.get("dataset_split", ""))

        feature_path = Path(feature_file)

        if not feature_path.is_absolute():

            feature_path = project_root / feature_path

        if feature_file == "" or not feature_path.exists():

            missing_feature_files += 1

        if training_text == "":

            empty_training_text += 1

        if dataset_split != split_name:

            wrong_split += 1

    result["missing_feature_files"] = missing_feature_files
    result["empty_training_text"] = empty_training_text

    if missing_feature_files > 0:

        result["message"] = f"Missing feature files: {missing_feature_files}"

        return result

    if empty_training_text > 0:

        result["message"] = f"Empty training text records: {empty_training_text}"

        return result

    if wrong_split > 0:

        result["message"] = f"Wrong split labels: {wrong_split}"

        return result

    result["valid"] = True
    result["message"] = "Split manifest validation passed"

    return result


def validate_dataset_splits(project_root: Path, manifest_info):

    logger.info("=" * 70)
    logger.info("Validating Dataset Splits")
    logger.info("=" * 70)

    train_manifest = Path(manifest_info["train_manifest"])
    val_manifest = Path(manifest_info["val_manifest"])
    test_manifest = Path(manifest_info["test_manifest"])

    split_results = []

    train_result = _validate_manifest_file(
        project_root,
        train_manifest,
        "train"
    )

    val_result = _validate_manifest_file(
        project_root,
        val_manifest,
        "val"
    )

    test_result = _validate_manifest_file(
        project_root,
        test_manifest,
        "test"
    )

    split_results.extend([
        train_result,
        val_result,
        test_result
    ])

    # --------------------------------------------------
    # Check overlap between splits
    # --------------------------------------------------

    overlap_found = False
    overlap_message = ""

    try:

        train_df = pd.read_csv(train_manifest)
        val_df = pd.read_csv(val_manifest)
        test_df = pd.read_csv(test_manifest)

        train_ids = set(train_df["pair_id"].astype(str))
        val_ids = set(val_df["pair_id"].astype(str))
        test_ids = set(test_df["pair_id"].astype(str))

        train_val_overlap = train_ids.intersection(val_ids)
        train_test_overlap = train_ids.intersection(test_ids)
        val_test_overlap = val_ids.intersection(test_ids)

        total_overlap = (
            len(train_val_overlap)
            + len(train_test_overlap)
            + len(val_test_overlap)
        )

        if total_overlap > 0:

            overlap_found = True

            overlap_message = (
                f"Overlap found between splits. "
                f"train-val: {len(train_val_overlap)}, "
                f"train-test: {len(train_test_overlap)}, "
                f"val-test: {len(val_test_overlap)}"
            )

    except Exception as e:

        overlap_found = True

        overlap_message = f"Cannot check overlap: {e}"

    # --------------------------------------------------
    # Final validation status
    # --------------------------------------------------

    all_valid = all(item["valid"] for item in split_results)

    if overlap_found:

        all_valid = False

    validation_summary = {

        "splits_valid": all_valid,

        "overlap_found": overlap_found,

        "overlap_message": overlap_message,

        "train_rows": train_result["rows"],

        "val_rows": val_result["rows"],

        "test_rows": test_result["rows"],

        "total_rows": (
            train_result["rows"]
            + val_result["rows"]
            + test_result["rows"]
        ),

        "split_results": split_results,

        "validation_status": "Passed" if all_valid else "Failed",

        "validation_message": (
            "All dataset splits are valid"
            if all_valid
            else "Dataset split validation failed"
        )

    }

    for item in split_results:

        logger.info(
            f"{item['split']} | rows: {item['rows']} | "
            f"valid: {item['valid']} | message: {item['message']}"
        )

    if overlap_found:

        logger.warning(overlap_message)

    logger.info(f"Dataset split validation status: {validation_summary['validation_status']}")
    logger.info(f"Total rows: {validation_summary['total_rows']}")

    logger.info("=" * 70)

    return validation_summary