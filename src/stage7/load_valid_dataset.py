from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("Stage7 Load Dataset")


def load_valid_dataset(project_root: Path):

    logger.info("=" * 70)
    logger.info("Loading Valid Dataset for Feature Extraction")
    logger.info("=" * 70)

    metadata_file = project_root / "metadata" / "metadata.csv"
    audio_folder = project_root / "data" / "processed_audio"
    alignment_folder = project_root / "data" / "alignment"

    if not metadata_file.exists():

        logger.warning("metadata.csv not found.")

        return []

    df = pd.read_csv(metadata_file)

    # --------------------------------------------------
    # Ensure required columns exist
    # --------------------------------------------------

    required_columns = [
        "pair_id",
        "audio_file",
        "audio_processed",
        "alignment_validation"
    ]

    for col in required_columns:

        if col not in df.columns:

            logger.warning(f"Required column missing: {col}")

            return []

    # --------------------------------------------------
    # Select valid samples
    # --------------------------------------------------

    valid_df = df[
        (df["audio_processed"].astype(str).str.lower() == "true")
        &
        (df["alignment_validation"].astype(str) == "Passed")
    ].copy()

    dataset = []

    loaded = 0
    missing_audio = 0
    missing_alignment = 0

    for _, row in valid_df.iterrows():

        audio_file = str(row["audio_file"])

        audio_path = audio_folder / audio_file

        alignment_file = alignment_folder / (
            Path(audio_file).stem + ".json"
        )

        if not audio_path.exists():

            logger.warning(f"Missing audio file: {audio_file}")

            missing_audio += 1

            continue

        if not alignment_file.exists():

            logger.warning(f"Missing alignment file: {alignment_file.name}")

            missing_alignment += 1

            continue

        dataset.append({

            "pair_id": row["pair_id"],

            "audio_file": audio_file,

            "audio_path": str(audio_path),

            "alignment_file": str(alignment_file),

            "duration": row.get("processed_duration", ""),

            "sample_rate": row.get("processed_sample_rate", "")

        })

        loaded += 1

    logger.info(f"Valid metadata rows : {len(valid_df)}")
    logger.info(f"Loaded samples      : {loaded}")
    logger.info(f"Missing audio       : {missing_audio}")
    logger.info(f"Missing alignment   : {missing_alignment}")

    logger.info("=" * 70)

    return dataset