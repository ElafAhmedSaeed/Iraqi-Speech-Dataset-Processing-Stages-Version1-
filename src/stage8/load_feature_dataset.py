from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("Stage8 Load Feature Dataset")


def _is_true(value):

    return str(value).strip().lower() in ["true", "1", "yes"]


def _safe_text(value):

    if value is None:
        return ""

    value = str(value)

    if value.lower() == "nan":
        return ""

    return value.strip()


def load_feature_dataset(project_root: Path):

    logger.info("=" * 70)
    logger.info("Loading Feature Dataset for Training Preparation")
    logger.info("=" * 70)

    metadata_file = project_root / "metadata" / "metadata.csv"

    if not metadata_file.exists():

        logger.warning("metadata.csv not found.")

        return []

    df = pd.read_csv(metadata_file)

    # --------------------------------------------------
    # Required columns from previous stages
    # --------------------------------------------------

    required_columns = [

        "pair_id",

        "audio_file",

        "feature_extracted",

        "feature_type",

        "feature_file",

        "feature_shape",

        "feature_status",

        "transcript",

        "normalized_text"

    ]

    for col in required_columns:

        if col not in df.columns:

            logger.warning(f"Required column missing: {col}")

            return []

    # --------------------------------------------------
    # Load valid feature samples
    # --------------------------------------------------

    feature_dataset = []

    total_rows = len(df)

    selected = 0
    skipped_not_extracted = 0
    skipped_not_saved = 0
    skipped_missing_feature_file = 0
    skipped_missing_text = 0
    skipped_file_not_found = 0

    for _, row in df.iterrows():

        pair_id = row.get("pair_id", "")

        audio_file = _safe_text(row.get("audio_file", ""))

        feature_extracted = _is_true(row.get("feature_extracted", False))

        feature_status = _safe_text(row.get("feature_status", ""))

        feature_file = _safe_text(row.get("feature_file", ""))

        feature_type = _safe_text(row.get("feature_type", ""))

        feature_shape = _safe_text(row.get("feature_shape", ""))

        transcript = _safe_text(row.get("transcript", ""))

        normalized_text = _safe_text(row.get("normalized_text", ""))

        duration = row.get("processed_duration", row.get("duration_sec", ""))

        sample_rate = row.get("processed_sample_rate", row.get("sample_rate", ""))

        # --------------------------------------------------
        # Condition 1: Feature extracted
        # --------------------------------------------------

        if not feature_extracted:

            skipped_not_extracted += 1

            continue

        # --------------------------------------------------
        # Condition 2: Feature status saved
        # --------------------------------------------------

        if feature_status != "Saved":

            skipped_not_saved += 1

            continue

        # --------------------------------------------------
        # Condition 3: Feature file path exists in metadata
        # --------------------------------------------------

        if feature_file == "":

            skipped_missing_feature_file += 1

            continue

        feature_path = Path(feature_file)

        # If relative path is stored, convert it to full path
        if not feature_path.is_absolute():

            feature_path = project_root / feature_path

        # --------------------------------------------------
        # Condition 4: Feature file exists on disk
        # --------------------------------------------------

        if not feature_path.exists():

            logger.warning(f"Feature file not found: {feature_path}")

            skipped_file_not_found += 1

            continue

        # --------------------------------------------------
        # Condition 5: Text exists
        # --------------------------------------------------

        if normalized_text == "" and transcript == "":

            skipped_missing_text += 1

            continue

        # Prefer normalized text for training
        training_text = normalized_text if normalized_text != "" else transcript

        feature_dataset.append({

            "pair_id": pair_id,

            "audio_file": audio_file,

            "feature_file": str(feature_path),

            "feature_type": feature_type,

            "feature_shape": feature_shape,

            "transcript": transcript,

            "normalized_text": normalized_text,

            "training_text": training_text,

            "duration": duration,

            "sample_rate": sample_rate,

            "dataset_ready": True,

            "dataset_status": "Loaded"

        })

        selected += 1

    # --------------------------------------------------
    # Logging summary
    # --------------------------------------------------

    logger.info(f"Total metadata rows            : {total_rows}")
    logger.info(f"Selected feature samples       : {selected}")
    logger.info(f"Skipped not extracted          : {skipped_not_extracted}")
    logger.info(f"Skipped not saved              : {skipped_not_saved}")
    logger.info(f"Skipped missing feature file   : {skipped_missing_feature_file}")
    logger.info(f"Skipped feature file not found : {skipped_file_not_found}")
    logger.info(f"Skipped missing text           : {skipped_missing_text}")

    logger.info("=" * 70)

    return feature_dataset