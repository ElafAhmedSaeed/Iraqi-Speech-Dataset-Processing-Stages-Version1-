from pathlib import Path

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("Stage9 Feature Loader")


def _safe_text(value):

    if value is None:
        return ""

    value = str(value)

    if value.lower() == "nan":
        return ""

    return value.strip()


def _resolve_feature_path(project_root: Path, feature_file):

    feature_file = _safe_text(feature_file)

    if feature_file == "":
        return None

    feature_path = Path(feature_file)

    if not feature_path.is_absolute():
        feature_path = project_root / feature_path

    return feature_path


def _load_single_split_features(project_root: Path, df, split_name: str):

    logger.info("-" * 70)
    logger.info(f"Loading feature files for split: {split_name}")
    logger.info("-" * 70)

    loaded_records = []

    total = len(df)
    loaded = 0
    failed = 0
    missing_file = 0
    empty_feature = 0

    for index, row in df.iterrows():

        audio_file = _safe_text(row.get("audio_file", ""))

        feature_file = _safe_text(row.get("feature_file", ""))

        logger.info(
            f"[{index + 1}/{total}] Loading feature: {audio_file}"
        )

        feature_path = _resolve_feature_path(
            project_root,
            feature_file
        )

        if feature_path is None:

            logger.warning(
                f"Missing feature path for audio file: {audio_file}"
            )

            failed += 1

            continue

        if not feature_path.exists():

            logger.warning(
                f"Feature file not found: {feature_path}"
            )

            missing_file += 1
            failed += 1

            continue

        try:

            features = np.load(
                feature_path,
                allow_pickle=False
            )

            if features is None:

                logger.warning(
                    f"Feature array is None: {feature_path}"
                )

                empty_feature += 1
                failed += 1

                continue

            if features.size == 0:

                logger.warning(
                    f"Feature array is empty: {feature_path}"
                )

                empty_feature += 1
                failed += 1

                continue

            record = row.to_dict()

            record["feature_path"] = str(feature_path)
            record["features"] = features
            record["loaded_feature_shape"] = features.shape
            record["feature_loaded"] = True
            record["load_status"] = "Loaded"
            record["load_message"] = "Feature file loaded successfully"

            loaded_records.append(record)

            loaded += 1

        except Exception as e:

            logger.warning(
                f"Failed to load feature file {feature_path}: {e}"
            )

            failed += 1

            continue

    split_summary = {

        "split": split_name,

        "total_records": total,

        "loaded_records": loaded,

        "failed_records": failed,

        "missing_feature_files": missing_file,

        "empty_feature_files": empty_feature,

        "status": "Loaded" if loaded > 0 and failed == 0 else "Loaded with warnings"

    }

    logger.info(f"{split_name} total records        : {total}")
    logger.info(f"{split_name} loaded records       : {loaded}")
    logger.info(f"{split_name} failed records       : {failed}")
    logger.info(f"{split_name} missing files        : {missing_file}")
    logger.info(f"{split_name} empty feature files  : {empty_feature}")

    return loaded_records, split_summary


def load_feature_files(project_root: Path, train_df, val_df, test_df):

    logger.info("=" * 70)
    logger.info("Loading Feature Files")
    logger.info("=" * 70)

    train_features, train_summary = _load_single_split_features(
        project_root,
        train_df,
        "train"
    )

    val_features, val_summary = _load_single_split_features(
        project_root,
        val_df,
        "val"
    )

    test_features, test_summary = _load_single_split_features(
        project_root,
        test_df,
        "test"
    )

    total_loaded = (
        train_summary["loaded_records"]
        + val_summary["loaded_records"]
        + test_summary["loaded_records"]
    )

    total_failed = (
        train_summary["failed_records"]
        + val_summary["failed_records"]
        + test_summary["failed_records"]
    )

    load_summary = {

        "train_summary": train_summary,

        "val_summary": val_summary,

        "test_summary": test_summary,

        "total_loaded": total_loaded,

        "total_failed": total_failed,

        "status": "Loaded" if total_failed == 0 else "Loaded with warnings",

        "message": (
            "All feature files loaded successfully"
            if total_failed == 0
            else "Some feature files failed to load"
        )

    }

    logger.info("=" * 70)
    logger.info(f"Total loaded feature files : {total_loaded}")
    logger.info(f"Total failed feature files : {total_failed}")
    logger.info(f"Feature loading status     : {load_summary['status']}")
    logger.info("=" * 70)

    return train_features, val_features, test_features, load_summary