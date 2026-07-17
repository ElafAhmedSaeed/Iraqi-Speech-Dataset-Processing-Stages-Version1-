from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("Metadata Stage4")


def update_metadata_stage4(project_root: Path, validated_audio):

    metadata_file = project_root / "metadata" / "metadata.csv"

    if not metadata_file.exists():

        logger.error("metadata.csv not found.")

        return

    df = pd.read_csv(metadata_file)

    # -------------------------------------------------
    # Create Columns if Missing
    # -------------------------------------------------

    columns = {

        "audio_processed": False,
        "processed_sample_rate": 0,
        "processed_duration": 0.0,
        "processing_status": "",
        "issues": "",
        "warnings": ""

    }

    for col, default in columns.items():

        if col not in df.columns:
            df[col] = default

    # -------------------------------------------------
    # Fix Data Types
    # -------------------------------------------------

    df["audio_processed"] = (
        df["audio_processed"]
        .replace({"TRUE": True, "FALSE": False})
        .fillna(False)
        .astype(bool)
    )

    df["processed_sample_rate"] = (
        pd.to_numeric(
            df["processed_sample_rate"],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

    df["processed_duration"] = (
        pd.to_numeric(
            df["processed_duration"],
            errors="coerce"
        )
        .fillna(0.0)
    )


    # -------------------------------------------------
    # Update Metadata
    # -------------------------------------------------

    for audio in validated_audio:

        file_name = audio["file_name"]

        #mask = df["file_name"] == file_name
        mask = df["audio_file"] == file_name

        df.loc[mask, "audio_processed"] = True

        df.loc[mask, "processed_sample_rate"] = audio["sample_rate"]

        df.loc[mask, "processed_duration"] = round(

            audio["duration"],

            3

        )

        df.loc[mask, "validation_status"] = audio["validation_status"]

        df.loc[mask, "issues"] = audio["issues"]

        df.loc[mask, "warnings"] = audio["warnings"]

    df.to_csv(

        metadata_file,

        index=False,

        encoding="utf-8-sig"

    )

    logger.info("Metadata updated successfully.")

    return df