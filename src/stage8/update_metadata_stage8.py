from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("Stage8 Metadata")


def update_metadata_stage8(
    project_root: Path,
    train_records,
    val_records,
    test_records,
    validation_summary
):

    logger.info("=" * 70)
    logger.info("Updating Metadata for Stage 8")
    logger.info("=" * 70)

    metadata_file = project_root / "metadata" / "metadata.csv"

    if not metadata_file.exists():

        logger.warning("metadata.csv not found.")

        return None

    df = pd.read_csv(metadata_file)

    # --------------------------------------------------
    # Required Stage 8 columns
    # --------------------------------------------------

    required_columns = {

        "dataset_split": "",

        "included_in_training": False,

        "dataset_status": "",

        "dataset_message": ""

    }

    for col, default in required_columns.items():

        if col not in df.columns:

            df[col] = default

    # --------------------------------------------------
    # Fix column data types
    # --------------------------------------------------

    text_columns = [

        "dataset_split",

        "dataset_status",

        "dataset_message"

    ]

    for col in text_columns:

        df[col] = df[col].fillna("").astype(str)

    df["included_in_training"] = (

        df["included_in_training"]

        .fillna(False)

        .astype(bool)

    )

    # --------------------------------------------------
    # Combine all records
    # --------------------------------------------------

    all_records = []

    all_records.extend(train_records)
    all_records.extend(val_records)
    all_records.extend(test_records)

    updated = 0
    skipped = 0

    # --------------------------------------------------
    # Update rows
    # --------------------------------------------------

    for record in all_records:

        audio_file = str(record.get("audio_file", "")).strip()

        dataset_split = str(record.get("dataset_split", "")).strip()

        if audio_file == "":

            skipped += 1

            continue

        mask = (

            df["audio_file"]

            .fillna("")

            .astype(str)

            .str.strip()

            == audio_file

        )

        if not mask.any():

            logger.warning(f"Audio file not found in metadata: {audio_file}")

            skipped += 1

            continue

        df.loc[mask, "dataset_split"] = dataset_split
        df.loc[mask, "included_in_training"] = True
        df.loc[mask, "dataset_status"] = "Prepared"
        df.loc[mask, "dataset_message"] = "Sample included in dataset split"

        updated += int(mask.sum())

    # --------------------------------------------------
    # If validation failed, mark dataset message
    # --------------------------------------------------

    if validation_summary.get("validation_status", "") != "Passed":

        df["dataset_status"] = "Validation Failed"

        df["dataset_message"] = validation_summary.get(

            "validation_message",

            "Dataset split validation failed"

        )

    # --------------------------------------------------
    # Save metadata
    # --------------------------------------------------

    df.to_csv(

        metadata_file,

        index=False,

        encoding="utf-8-sig"

    )

    logger.info(f"Metadata updated records : {updated}")
    logger.info(f"Metadata skipped records : {skipped}")
    logger.info(f"Dataset validation       : {validation_summary.get('validation_status', '')}")

    logger.info("=" * 70)

    return df