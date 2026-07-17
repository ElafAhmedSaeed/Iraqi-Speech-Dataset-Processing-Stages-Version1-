import numpy as np

from src.utils.logger import get_logger

logger = get_logger("Stage9 Data Loader Validator")


def _safe_text(value):

    if value is None:
        return ""

    value = str(value)

    if value.lower() == "nan":
        return ""

    return value.strip()


def _validate_single_dataset_object(item, split_name):

    audio_file = _safe_text(
        item.get("audio_file", "")
    )

    features = item.get("features", None)

    label_ids = item.get("label_ids", [])

    label_text = _safe_text(
        item.get("label_text", "")
    )

    dataset_split = _safe_text(
        item.get("dataset_split", "")
    )

    # --------------------------------------------------
    # Check dataset split
    # --------------------------------------------------

    if dataset_split != split_name:

        return {

            "valid": False,

            "message": f"Wrong dataset split: expected {split_name}, got {dataset_split}"

        }

    # --------------------------------------------------
    # Check features
    # --------------------------------------------------

    if features is None:

        return {

            "valid": False,

            "message": "Features are None"

        }

    try:

        features = np.asarray(features)

    except Exception as e:

        return {

            "valid": False,

            "message": f"Cannot convert features to numpy array: {e}"

        }

    if features.size == 0:

        return {

            "valid": False,

            "message": "Feature array is empty"

        }

    if features.ndim != 2:

        return {

            "valid": False,

            "message": f"Invalid feature dimensions: {features.ndim}"

        }

    if np.isnan(features).any():

        return {

            "valid": False,

            "message": "Feature contains NaN values"

        }

    if np.isinf(features).any():

        return {

            "valid": False,

            "message": "Feature contains infinite values"

        }

    feature_dim = features.shape[0]

    feature_length = features.shape[1]

    if feature_dim <= 0:

        return {

            "valid": False,

            "message": "Invalid feature dimension"

        }

    if feature_length <= 0:

        return {

            "valid": False,

            "message": "Invalid feature length"

        }

    # --------------------------------------------------
    # Check labels
    # --------------------------------------------------

    if label_ids is None:

        return {

            "valid": False,

            "message": "Label IDs are None"

        }

    if len(label_ids) == 0:

        return {

            "valid": False,

            "message": "Label IDs are empty"

        }

    if label_text == "":

        return {

            "valid": False,

            "message": "Label text is empty"

        }

    label_length = len(label_ids)

    if label_length <= 0:

        return {

            "valid": False,

            "message": "Invalid label length"

        }

    # --------------------------------------------------
    # Optional CTC compatibility check
    # --------------------------------------------------
    # In CTC-based ASR, the feature time length should usually
    # be greater than or equal to the label length.
    # --------------------------------------------------

    if feature_length < label_length:

        return {

            "valid": False,

            "message": (
                f"Feature length is shorter than label length "
                f"({feature_length} < {label_length})"
            )

        }

    return {

        "valid": True,

        "message": "Dataset object validation passed"

    }


def _validate_single_split(dataset, split_name):

    logger.info("-" * 70)
    logger.info(f"Validating dataset split: {split_name}")
    logger.info("-" * 70)

    total = len(dataset)

    valid = 0
    invalid = 0

    validated_dataset = []

    for index, item in enumerate(dataset, start=1):

        audio_file = _safe_text(
            item.get("audio_file", "")
        )

        logger.info(
            f"[{index}/{total}] Validating dataset object: {audio_file}"
        )

        result = _validate_single_dataset_object(
            item,
            split_name
        )

        item["data_loader_valid"] = result["valid"]

        item["data_loader_validation_status"] = (
            "Passed" if result["valid"] else "Failed"
        )

        item["data_loader_validation_message"] = result["message"]

        validated_dataset.append(item)

        if result["valid"]:

            valid += 1

        else:

            invalid += 1

            logger.warning(
                f"Validation failed for {audio_file}: {result['message']}"
            )

    split_summary = {

        "split": split_name,

        "total_objects": total,

        "valid_objects": valid,

        "invalid_objects": invalid,

        "status": "Passed" if invalid == 0 and valid > 0 else "Failed",

        "message": (
            "All dataset objects are valid"
            if invalid == 0 and valid > 0
            else "Some dataset objects are invalid"
        )

    }

    logger.info(f"{split_name} total objects   : {total}")
    logger.info(f"{split_name} valid objects   : {valid}")
    logger.info(f"{split_name} invalid objects : {invalid}")
    logger.info(f"{split_name} status          : {split_summary['status']}")

    return validated_dataset, split_summary


def validate_data_loader(
    train_dataset,
    val_dataset,
    test_dataset
):

    logger.info("=" * 70)
    logger.info("Validating Data Loader Datasets")
    logger.info("=" * 70)

    train_validated, train_summary = _validate_single_split(
        train_dataset,
        "train"
    )

    val_validated, val_summary = _validate_single_split(
        val_dataset,
        "val"
    )

    test_validated, test_summary = _validate_single_split(
        test_dataset,
        "test"
    )

    total_objects = (
        train_summary["total_objects"]
        + val_summary["total_objects"]
        + test_summary["total_objects"]
    )

    total_valid = (
        train_summary["valid_objects"]
        + val_summary["valid_objects"]
        + test_summary["valid_objects"]
    )

    total_invalid = (
        train_summary["invalid_objects"]
        + val_summary["invalid_objects"]
        + test_summary["invalid_objects"]
    )

    all_passed = (

        train_summary["status"] == "Passed"

        and val_summary["status"] == "Passed"

        and test_summary["status"] == "Passed"

    )

    validation_summary = {

        "status": "Passed" if all_passed else "Failed",

        "message": (
            "Data loader validation passed"
            if all_passed
            else "Data loader validation failed"
        ),

        "total_objects": total_objects,

        "valid_objects": total_valid,

        "invalid_objects": total_invalid,

        "train_summary": train_summary,

        "val_summary": val_summary,

        "test_summary": test_summary

    }

    logger.info("=" * 70)
    logger.info(f"Total dataset objects : {total_objects}")
    logger.info(f"Valid objects         : {total_valid}")
    logger.info(f"Invalid objects       : {total_invalid}")
    logger.info(f"Validation status     : {validation_summary['status']}")
    logger.info("=" * 70)

    return train_validated, val_validated, test_validated, validation_summary