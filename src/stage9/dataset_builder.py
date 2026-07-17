import numpy as np

from src.utils.logger import get_logger

logger = get_logger("Stage9 Dataset Builder")


def _safe_text(value):

    if value is None:
        return ""

    value = str(value)

    if value.lower() == "nan":
        return ""

    return value.strip()


def _build_single_split_dataset(records, split_name: str):

    logger.info("-" * 70)
    logger.info(f"Building dataset objects for split: {split_name}")
    logger.info("-" * 70)

    dataset = []

    total = len(records)
    built = 0
    skipped = 0

    for index, record in enumerate(records, start=1):

        audio_file = _safe_text(
            record.get("audio_file", "")
        )

        logger.info(
            f"[{index}/{total}] Building dataset object: {audio_file}"
        )

        features = record.get("features", None)

        label_ids = record.get("label_ids", [])

        label_text = _safe_text(
            record.get("label_text", "")
        )

        feature_file = _safe_text(
            record.get("feature_file", "")
        )

        feature_type = _safe_text(
            record.get("feature_type", "")
        )

        dataset_split = _safe_text(
            record.get("dataset_split", "")
        )

        # --------------------------------------------------
        # Validate feature
        # --------------------------------------------------

        if features is None:

            logger.warning(
                f"Skipped {audio_file}: features are None"
            )

            skipped += 1

            continue

        try:

            features = np.asarray(features)

        except Exception as e:

            logger.warning(
                f"Skipped {audio_file}: cannot convert features to array: {e}"
            )

            skipped += 1

            continue

        if features.size == 0:

            logger.warning(
                f"Skipped {audio_file}: empty feature array"
            )

            skipped += 1

            continue

        if features.ndim != 2:

            logger.warning(
                f"Skipped {audio_file}: invalid feature dimension {features.ndim}"
            )

            skipped += 1

            continue

        # --------------------------------------------------
        # Validate label
        # --------------------------------------------------

        if label_ids is None or len(label_ids) == 0:

            logger.warning(
                f"Skipped {audio_file}: empty label ids"
            )

            skipped += 1

            continue

        if label_text == "":

            logger.warning(
                f"Skipped {audio_file}: empty label text"
            )

            skipped += 1

            continue

        # --------------------------------------------------
        # Feature shape information
        # --------------------------------------------------

        feature_dim = int(features.shape[0])

        feature_length = int(features.shape[1])

        label_length = int(len(label_ids))

        # --------------------------------------------------
        # Build dataset object
        # --------------------------------------------------

        dataset_object = {

            "pair_id": record.get("pair_id", ""),

            "audio_file": audio_file,

            "feature_file": feature_file,

            "feature_type": feature_type,

            "features": features,

            "feature_shape": features.shape,

            "feature_dim": feature_dim,

            "feature_length": feature_length,

            "label_text": label_text,

            "label_ids": label_ids,

            "label_length": label_length,

            "label_type": record.get("label_type", "character"),

            "duration": record.get("duration", ""),

            "sample_rate": record.get("sample_rate", ""),

            "dataset_split": dataset_split,

            "dataset_object_ready": True,

            "dataset_object_status": "Built",

            "dataset_object_message": "Dataset object built successfully"

        }

        dataset.append(dataset_object)

        built += 1

    split_summary = {

        "split": split_name,

        "total_records": total,

        "built_objects": built,

        "skipped_objects": skipped,

        "status": "Built" if built > 0 and skipped == 0 else "Built with warnings"

    }

    logger.info(f"{split_name} total records    : {total}")
    logger.info(f"{split_name} built objects    : {built}")
    logger.info(f"{split_name} skipped objects  : {skipped}")

    return dataset, split_summary


def build_dataset_objects(
    train_labeled,
    val_labeled,
    test_labeled
):

    logger.info("=" * 70)
    logger.info("Building Dataset Objects")
    logger.info("=" * 70)

    train_dataset, train_summary = _build_single_split_dataset(
        train_labeled,
        "train"
    )

    val_dataset, val_summary = _build_single_split_dataset(
        val_labeled,
        "val"
    )

    test_dataset, test_summary = _build_single_split_dataset(
        test_labeled,
        "test"
    )

    total_built = (
        train_summary["built_objects"]
        + val_summary["built_objects"]
        + test_summary["built_objects"]
    )

    total_skipped = (
        train_summary["skipped_objects"]
        + val_summary["skipped_objects"]
        + test_summary["skipped_objects"]
    )

    dataset_summary = {

        "status": "Built" if total_built > 0 and total_skipped == 0 else "Built with warnings",

        "message": (
            "All dataset objects built successfully"
            if total_skipped == 0
            else "Some dataset objects were skipped"
        ),

        "train_summary": train_summary,

        "val_summary": val_summary,

        "test_summary": test_summary,

        "train_objects": len(train_dataset),

        "val_objects": len(val_dataset),

        "test_objects": len(test_dataset),

        "total_objects": total_built,

        "skipped_objects": total_skipped

    }

    logger.info("=" * 70)
    logger.info(f"Train dataset objects      : {len(train_dataset)}")
    logger.info(f"Validation dataset objects : {len(val_dataset)}")
    logger.info(f"Test dataset objects       : {len(test_dataset)}")
    logger.info(f"Total dataset objects      : {total_built}")
    logger.info(f"Skipped objects            : {total_skipped}")
    logger.info(f"Dataset building status    : {dataset_summary['status']}")
    logger.info("=" * 70)

    return train_dataset, val_dataset, test_dataset, dataset_summary