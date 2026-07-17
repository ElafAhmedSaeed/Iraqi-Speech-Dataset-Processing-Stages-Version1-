import numpy as np

from src.utils.logger import get_logger

logger = get_logger("Stage10 Batch Validator")


def _to_numpy(value):

    if value is None:
        return None

    # If value is already numpy
    if isinstance(value, np.ndarray):
        return value

    # If value is torch tensor
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def _validate_single_batch(batch, split_name, batch_index):

    if batch is None:

        return {

            "valid": False,

            "message": "Batch is None"

        }

    required_keys = [

        "features",

        "labels",

        "flat_labels",

        "input_lengths",

        "label_lengths",

        "batch_size",

        "max_input_length",

        "feature_dim",

        "max_label_length"

    ]

    for key in required_keys:

        if key not in batch:

            return {

                "valid": False,

                "message": f"Missing batch key: {key}"

            }

    features = _to_numpy(
        batch["features"]
    )

    labels = _to_numpy(
        batch["labels"]
    )

    flat_labels = _to_numpy(
        batch["flat_labels"]
    )

    input_lengths = _to_numpy(
        batch["input_lengths"]
    )

    label_lengths = _to_numpy(
        batch["label_lengths"]
    )

    batch_size = int(
        batch["batch_size"]
    )

    # --------------------------------------------------
    # Validate features
    # --------------------------------------------------

    if features is None:

        return {

            "valid": False,

            "message": "Features are None"

        }

    if features.ndim != 3:

        return {

            "valid": False,

            "message": f"Features must be 3D, got {features.ndim}D"

        }

    if features.shape[0] != batch_size:

        return {

            "valid": False,

            "message": "Feature batch size mismatch"

        }

    if features.shape[1] <= 0:

        return {

            "valid": False,

            "message": "Invalid max input length"

        }

    if features.shape[2] <= 0:

        return {

            "valid": False,

            "message": "Invalid feature dimension"

        }

    if np.isnan(features).any():

        return {

            "valid": False,

            "message": "Features contain NaN values"

        }

    if np.isinf(features).any():

        return {

            "valid": False,

            "message": "Features contain infinite values"

        }

    # --------------------------------------------------
    # Validate labels
    # --------------------------------------------------

    if labels is None:

        return {

            "valid": False,

            "message": "Labels are None"

        }

    if labels.ndim != 2:

        return {

            "valid": False,

            "message": f"Labels must be 2D, got {labels.ndim}D"

        }

    if labels.shape[0] != batch_size:

        return {

            "valid": False,

            "message": "Label batch size mismatch"

        }

    if labels.shape[1] <= 0:

        return {

            "valid": False,

            "message": "Invalid max label length"

        }

    # --------------------------------------------------
    # Validate flat labels
    # --------------------------------------------------

    if flat_labels is None:

        return {

            "valid": False,

            "message": "Flat labels are None"

        }

    if flat_labels.ndim != 1:

        return {

            "valid": False,

            "message": "Flat labels must be 1D"

        }

    if len(flat_labels) == 0:

        return {

            "valid": False,

            "message": "Flat labels are empty"

        }

    # --------------------------------------------------
    # Validate lengths
    # --------------------------------------------------

    if input_lengths is None or label_lengths is None:

        return {

            "valid": False,

            "message": "Input lengths or label lengths are None"

        }

    if input_lengths.ndim != 1:

        return {

            "valid": False,

            "message": "Input lengths must be 1D"

        }

    if label_lengths.ndim != 1:

        return {

            "valid": False,

            "message": "Label lengths must be 1D"

        }

    if len(input_lengths) != batch_size:

        return {

            "valid": False,

            "message": "Input lengths size mismatch"

        }

    if len(label_lengths) != batch_size:

        return {

            "valid": False,

            "message": "Label lengths size mismatch"

        }

    if np.any(input_lengths <= 0):

        return {

            "valid": False,

            "message": "Input lengths contain zero or negative values"

        }

    if np.any(label_lengths <= 0):

        return {

            "valid": False,

            "message": "Label lengths contain zero or negative values"

        }

    if np.any(input_lengths > features.shape[1]):

        return {

            "valid": False,

            "message": "Some input lengths exceed padded feature length"

        }

    if np.any(label_lengths > labels.shape[1]):

        return {

            "valid": False,

            "message": "Some label lengths exceed padded label length"

        }

    # --------------------------------------------------
    # CTC compatibility check
    # --------------------------------------------------

    if np.any(input_lengths < label_lengths):

        return {

            "valid": False,

            "message": "Some input lengths are shorter than label lengths"

        }

    return {

        "valid": True,

        "message": "Training batch validation passed"

    }


def _validate_loader_batches(loader, split_name, max_batches=3):

    logger.info("-" * 70)
    logger.info(f"Validating training batches for split: {split_name}")
    logger.info("-" * 70)

    if loader is None:

        return {

            "split": split_name,

            "status": "Failed",

            "message": "Loader is None",

            "checked_batches": 0,

            "valid_batches": 0,

            "invalid_batches": 0

        }

    checked_batches = 0
    valid_batches = 0
    invalid_batches = 0

    last_feature_shape = ""
    last_label_shape = ""

    try:

        for batch_index, batch in enumerate(loader, start=1):

            if checked_batches >= max_batches:

                break

            logger.info(
                f"Validating {split_name} batch {batch_index}"
            )

            result = _validate_single_batch(
                batch,
                split_name,
                batch_index
            )

            checked_batches += 1

            if result["valid"]:

                valid_batches += 1

                last_feature_shape = str(
                    _to_numpy(batch["features"]).shape
                )

                last_label_shape = str(
                    _to_numpy(batch["labels"]).shape
                )

                logger.info(
                    f"{split_name} batch {batch_index} passed | "
                    f"features: {last_feature_shape} | "
                    f"labels: {last_label_shape}"
                )

            else:

                invalid_batches += 1

                logger.warning(
                    f"{split_name} batch {batch_index} failed: "
                    f"{result['message']}"
                )

    except Exception as e:

        return {

            "split": split_name,

            "status": "Failed",

            "message": f"Batch iteration failed: {e}",

            "checked_batches": checked_batches,

            "valid_batches": valid_batches,

            "invalid_batches": invalid_batches,

            "last_feature_shape": last_feature_shape,

            "last_label_shape": last_label_shape

        }

    status = "Passed" if checked_batches > 0 and invalid_batches == 0 else "Failed"

    message = (
        "Training batch validation passed"
        if status == "Passed"
        else "Some training batches failed validation"
    )

    summary = {

        "split": split_name,

        "status": status,

        "message": message,

        "checked_batches": checked_batches,

        "valid_batches": valid_batches,

        "invalid_batches": invalid_batches,

        "last_feature_shape": last_feature_shape,

        "last_label_shape": last_label_shape

    }

    logger.info(f"{split_name} checked batches : {checked_batches}")
    logger.info(f"{split_name} valid batches   : {valid_batches}")
    logger.info(f"{split_name} invalid batches : {invalid_batches}")
    logger.info(f"{split_name} status          : {status}")

    return summary


def validate_training_batches(
    data_loaders,
    max_batches=3
):

    logger.info("=" * 70)
    logger.info("Validating Training Batches")
    logger.info("=" * 70)

    train_loader = data_loaders.get("train_loader")
    val_loader = data_loaders.get("val_loader")
    test_loader = data_loaders.get("test_loader")

    train_summary = _validate_loader_batches(
        train_loader,
        "train",
        max_batches=max_batches
    )

    val_summary = _validate_loader_batches(
        val_loader,
        "val",
        max_batches=max_batches
    )

    test_summary = _validate_loader_batches(
        test_loader,
        "test",
        max_batches=max_batches
    )

    all_passed = (

        train_summary["status"] == "Passed"

        and val_summary["status"] == "Passed"

        and test_summary["status"] == "Passed"

    )

    batch_validation_summary = {

        "status": "Passed" if all_passed else "Failed",

        "message": (
            "All training batches are valid"
            if all_passed
            else "Training batch validation failed"
        ),

        "max_batches_checked_per_split": max_batches,

        "train_summary": train_summary,

        "val_summary": val_summary,

        "test_summary": test_summary,

        "total_checked_batches": (
            train_summary["checked_batches"]
            + val_summary["checked_batches"]
            + test_summary["checked_batches"]
        ),

        "total_valid_batches": (
            train_summary["valid_batches"]
            + val_summary["valid_batches"]
            + test_summary["valid_batches"]
        ),

        "total_invalid_batches": (
            train_summary["invalid_batches"]
            + val_summary["invalid_batches"]
            + test_summary["invalid_batches"]
        )

    }

    logger.info("=" * 70)
    logger.info(f"Batch validation status : {batch_validation_summary['status']}")
    logger.info(f"Checked batches         : {batch_validation_summary['total_checked_batches']}")
    logger.info(f"Valid batches           : {batch_validation_summary['total_valid_batches']}")
    logger.info(f"Invalid batches         : {batch_validation_summary['total_invalid_batches']}")
    logger.info("=" * 70)

    return batch_validation_summary