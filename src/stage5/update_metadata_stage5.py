from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger
METADATA_AUDIO_COLUMN = "audio_file"

logger = get_logger("Stage5 Metadata")


def update_metadata_stage5(project_root: Path, dataset):

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

        "dataset_type": "",

        "segmentation_required": False,

        "segmentation_decision": "",

        "segments": 0,

        "segmentation_status": "",

        "segment_validation": ""

    }

    for column, default_value in columns.items():

        if column not in df.columns:

            df[column] = default_value

    # --------------------------------------------------
    # Update rows
    # --------------------------------------------------

    for audio in dataset:

        #mask = df["file_name"] == audio["file_name"]
        #mask = df["audio_file"] == audio["file_name"]
        # AUDIO_COLUMN = "audio_file"
        #
        # mask = df[AUDIO_COLUMN] == audio["file_name"]

        mask = df[METADATA_AUDIO_COLUMN] == audio["file_name"]

        df.loc[mask, "dataset_type"] = str(audio["dataset_type"])

        df.loc[mask, "segmentation_required"] = bool(audio["segmentation_required"])

        df.loc[mask, "segmentation_decision"] = str(audio["decision"])

        df.loc[mask, "segments"] = int(audio["segments"])

        df.loc[mask, "segmentation_status"] = str(audio["segmentation_status"])

        df.loc[mask, "segment_validation"] = str(audio["segment_validation"])

    df.to_csv(metadata_file, index=False)

    logger.info("Metadata updated successfully.")

    logger.info("=" * 70)