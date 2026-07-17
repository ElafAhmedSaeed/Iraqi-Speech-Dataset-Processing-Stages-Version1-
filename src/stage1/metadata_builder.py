"""
Stage 1
Metadata Builder

Create the master metadata file for the Iraqi Speech Dataset.
"""

from pathlib import Path
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("Metadata Builder")


def build_metadata():

    logger.info("=" * 70)
    logger.info("Building Metadata...")
    logger.info("=" * 70)

    project_root = Path(__file__).resolve().parents[2]

    pair_file = project_root / "metadata" / "pair_index.csv"

    if not pair_file.exists():
        logger.error("pair_index.csv not found.")
        return

    pair_df = pd.read_csv(pair_file)

    # --------------------------------------------------
    # Keep only matched pairs
    # --------------------------------------------------

    pair_df = pair_df[pair_df["status"] == "Matched"].copy()

    metadata = pd.DataFrame({

        "pair_id": pair_df["pair_id"],

        "audio_file": pair_df["audio_file"],

        "transcript_file": pair_df["transcript_file"],

        "status": pair_df["status"],

        # ---------- Speaker Information ----------
        "speaker_id": "",

        "speaker_name": "",

        "gender": "",

        "age": "",

        # ---------- Iraqi Dialect ----------
        "dialect": "",

        "governorate": "",

        # ---------- Audio Information ----------
        "duration_sec": "",

        "sample_rate": "",

        "channels": "",

        "bit_depth": "",

        # ---------- Transcript ----------
        "transcript": "",

        "normalized_text": "",

        # ---------- Processing ----------
        "audio_processed": False,

        "segmented": False,

        "aligned": False,

        "alignment_score": "",

        # ---------- Dataset ----------
        "source": "",

        "language": "Arabic",

        "country": "Iraq",

        "remarks": ""

    })

    output = project_root / "metadata" / "metadata.csv"

    metadata.to_csv(

        output,

        index=False,

        encoding="utf-8-sig"

    )

    logger.info(f"Metadata created successfully.")

    logger.info(f"Valid pairs : {len(metadata)}")

    logger.info(f"Saved to : {output}")

    logger.info("=" * 70)

    return metadata


if __name__ == "__main__":

    build_metadata()