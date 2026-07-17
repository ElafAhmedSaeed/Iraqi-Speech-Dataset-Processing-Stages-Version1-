from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("Stage8 Manifest Saver")


def save_single_manifest(records, output_file: Path):

    if len(records) == 0:

        df = pd.DataFrame()

    else:

        df = pd.DataFrame(records)

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    return df


def save_dataset_manifests(project_root: Path, train_records, val_records, test_records):

    logger.info("=" * 70)
    logger.info("Saving Dataset Manifests")
    logger.info("=" * 70)

    output_folder = project_root / "data" / "dataset_splits"

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    train_file = output_folder / "train_manifest.csv"
    val_file = output_folder / "val_manifest.csv"
    test_file = output_folder / "test_manifest.csv"

    # --------------------------------------------------
    # Save manifest files
    # --------------------------------------------------

    train_df = save_single_manifest(
        train_records,
        train_file
    )

    val_df = save_single_manifest(
        val_records,
        val_file
    )

    test_df = save_single_manifest(
        test_records,
        test_file
    )

    # --------------------------------------------------
    # Manifest summary
    # --------------------------------------------------

    manifest_info = {

        "output_folder": str(output_folder),

        "train_manifest": str(train_file),

        "val_manifest": str(val_file),

        "test_manifest": str(test_file),

        "train_count": len(train_df),

        "val_count": len(val_df),

        "test_count": len(test_df),

        "total_count": len(train_df) + len(val_df) + len(test_df),

        "save_status": "Saved",

        "save_message": "Dataset manifests saved successfully"

    }

    logger.info(f"Train manifest saved      : {train_file}")
    logger.info(f"Validation manifest saved : {val_file}")
    logger.info(f"Test manifest saved       : {test_file}")

    logger.info(f"Train records             : {manifest_info['train_count']}")
    logger.info(f"Validation records        : {manifest_info['val_count']}")
    logger.info(f"Test records              : {manifest_info['test_count']}")
    logger.info(f"Total records             : {manifest_info['total_count']}")

    logger.info("=" * 70)

    return manifest_info