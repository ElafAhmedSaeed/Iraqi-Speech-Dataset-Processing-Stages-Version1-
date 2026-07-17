from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("Stage6 Metadata")


def update_metadata_stage6(project_root: Path, dataset):

    logger.info("=" * 70)
    logger.info("Updating Metadata")
    logger.info("=" * 70)

    metadata_file = project_root / "metadata" / "metadata.csv"

    if not metadata_file.exists():

        logger.warning("metadata.csv not found.")

        return

    df = pd.read_csv(metadata_file)

    # --------------------------------------------------
    # Create columns if they don't exist
    # --------------------------------------------------

    columns = {

        "alignment_ready": False,

        "preparation_status": "",

        "alignment_status": "",

        "alignment_score": 0.0,

        "alignment_validation": ""

    }

    for column, default_value in columns.items():

        if column not in df.columns:

            df[column] = default_value

    # --------------------------------------------------
    # Update rows
    # --------------------------------------------------

    for sample in dataset:

        mask = df["audio_file"] == sample["audio_file"]

        df.loc[mask, "alignment_ready"] = bool(
            sample["alignment_ready"]
        )

        df.loc[mask, "preparation_status"] = str(
            sample["preparation_status"]
        )

        df.loc[mask, "alignment_status"] = str(
            sample["alignment_status"]
        )

        df.loc[mask, "alignment_score"] = float(
            sample["alignment_score"]
        )

        df.loc[mask, "alignment_validation"] = str(
            sample["alignment_validation"]
        )

    df.to_csv(metadata_file, index=False)

    logger.info("Metadata updated successfully.")

    logger.info("=" * 70)