from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("Stage7 Metadata")


def update_metadata_stage7(project_root: Path, saved_features):

    logger.info("=" * 70)
    logger.info("Updating Metadata for Stage 7")
    logger.info("=" * 70)

    metadata_file = project_root / "metadata" / "metadata.csv"

    if not metadata_file.exists():

        logger.warning("metadata.csv not found.")

        return None

    df = pd.read_csv(metadata_file)

    # --------------------------------------------------
    # Required Stage 7 columns
    # --------------------------------------------------

    required_columns = {

        "feature_extracted": False,

        "feature_type": "",

        "feature_file": "",

        "feature_shape": "",

        "feature_status": "",

        "feature_message": ""

    }

    for col, default in required_columns.items():

        if col not in df.columns:

            df[col] = default

    # --------------------------------------------------
    # Fix column data types
    # --------------------------------------------------

    text_columns = [

        "feature_type",

        "feature_file",

        "feature_shape",

        "feature_status",

        "feature_message"

    ]

    for col in text_columns:

        df[col] = df[col].fillna("").astype(str)

    df["feature_extracted"] = (

        df["feature_extracted"]

        .fillna(False)

        .astype(bool)

    )

    # --------------------------------------------------
    # Update metadata rows
    # --------------------------------------------------

    updated = 0
    skipped = 0

    for item in saved_features:

        audio_file = str(item.get("audio_file", ""))

        if audio_file == "":

            skipped += 1

            continue

        mask = (

            df["audio_file"]

            .fillna("")

            .astype(str)

            == audio_file

        )

        if not mask.any():

            logger.warning(f"Audio file not found in metadata: {audio_file}")

            skipped += 1

            continue

        feature_saved = bool(item.get("feature_saved", False))

        feature_type = str(item.get("feature_type", ""))

        feature_file = str(item.get("feature_file", ""))

        feature_shape = str(item.get("feature_shape", ""))

        save_status = str(item.get("save_status", ""))

        save_message = str(item.get("save_message", ""))

        df.loc[mask, "feature_extracted"] = feature_saved
        df.loc[mask, "feature_type"] = feature_type
        df.loc[mask, "feature_file"] = feature_file
        df.loc[mask, "feature_shape"] = feature_shape
        df.loc[mask, "feature_status"] = save_status
        df.loc[mask, "feature_message"] = save_message

        updated += int(mask.sum())

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

    logger.info("=" * 70)

    return df