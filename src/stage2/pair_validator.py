from pathlib import Path
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("Pair Validator")


def validate_pairs(project_root: Path):

    logger.info("=" * 70)
    logger.info("Validating Audio-Transcript Pairs...")
    logger.info("=" * 70)

    metadata = pd.read_csv(
        project_root / "metadata" / "metadata.csv"
    )

    audio = pd.read_csv(
        project_root / "metadata" / "audio_validation.csv"
    )

    transcript = pd.read_csv(
        project_root / "metadata" / "transcript_validation.csv"
    )

    # --------------------------------------------
    # Merge Audio + Transcript
    # --------------------------------------------

    df = metadata.merge(
        audio[
            [
                "pair_id",
                "audio_valid"
            ]
        ],
        on="pair_id",
        how="left"
    )

    df = df.merge(
        transcript[
            [
                "pair_id",
                "status"
            ]
        ].rename(
            columns={
                "status": "transcript_status"
            }
        ),
        on="pair_id",
        how="left"
    )

    pair_results = []

    for _, row in df.iterrows():

        audio_valid = bool(row["audio_valid"])

        #transcript_valid = row["status"] == "Valid"
        transcript_valid = row["transcript_status"] == "Valid"

        if audio_valid and transcript_valid:

            pair_valid = True

            pair_status = "Valid"

        elif not audio_valid and transcript_valid:

            pair_valid = False

            pair_status = "Invalid Audio"

        elif audio_valid and not transcript_valid:

            pair_valid = False

            pair_status = "Invalid Transcript"

        else:

            pair_valid = False

            pair_status = "Invalid Audio & Transcript"

        pair_results.append({

            "pair_id": row["pair_id"],

            "audio_file": row["audio_file"],

            "transcript_file": row["transcript_file"],

            "audio_valid": audio_valid,

            "transcript_valid": transcript_valid,

            "pair_valid": pair_valid,

            "status": pair_status

        })

    pair_df = pd.DataFrame(pair_results)

    logger.info(f"Validated {len(pair_df)} pairs.")

    logger.info("=" * 70)

    return pair_df