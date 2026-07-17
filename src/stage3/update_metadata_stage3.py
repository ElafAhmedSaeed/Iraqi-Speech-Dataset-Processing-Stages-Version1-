from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("Metadata Stage3")


def update_metadata_stage3(project_root, validated_transcripts):

    metadata_file = project_root / "metadata" / "metadata.csv"

    if not metadata_file.exists():
        logger.warning("metadata.csv not found.")
        return

    df = pd.read_csv(metadata_file)

    # --------------------------------------------------
    # Ensure required columns exist
    # --------------------------------------------------

    required_columns = {
        "transcript": "",
        "normalized_text": "",
        "text_processed": False,
        "text_status": ""
    }

    for col, default in required_columns.items():

        if col not in df.columns:
            df[col] = default

    # --------------------------------------------------
    # Force text columns to string
    # --------------------------------------------------

    for col in ["transcript", "normalized_text", "text_status"]:

        df[col] = (
            df[col]
            .fillna("")
            .astype("string")
        )

    # Boolean column
    df["text_processed"] = (
        df["text_processed"]
        .fillna(False)
        .astype(bool)
    )

    # --------------------------------------------------
    # Update rows
    # --------------------------------------------------

    updated = 0

    for transcript in validated_transcripts:

        txt_name = Path(transcript["file_name"]).stem

        mask = (
            df["transcript_file"]
            .fillna("")
            .astype(str)
            .apply(lambda x: Path(x).stem)
            == txt_name
        )

        if mask.any():

            df.loc[mask, "transcript"] = str(
                transcript["processed_text"]
            )

            df.loc[mask, "normalized_text"] = str(
                transcript["normalized_text"]
            )

            df.loc[mask, "text_processed"] = True

            df.loc[mask, "text_status"] = "Processed"

            updated += 1

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    df.to_csv(
        metadata_file,
        index=False,
        encoding="utf-8-sig"
    )

    logger.info(f"Metadata updated successfully ({updated} records).")

    return df