"""
Stage 1 - File Discovery Engine

This module scans the dataset folders and creates an inventory
of all audio and transcript files.
"""

from pathlib import Path
import pandas as pd

from config.settings import config
from src.utils.logger import get_logger

logger = get_logger("File Discovery")

# ------------------------------------------------------------------
# Supported Extensions
# ------------------------------------------------------------------

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}
TEXT_EXTENSIONS = {".txt", ".json", ".csv"}


def discover_files():
    """
    Scan dataset folders and generate file_inventory.csv
    """

    logger.info("=" * 70)
    logger.info("Starting File Discovery")
    logger.info("=" * 70)

    project_root = Path(__file__).resolve().parents[2]

    audio_folder = project_root / config["paths"]["raw_audio"]
    transcript_folder = project_root / config["paths"]["raw_transcripts"]

    # ---------------------------------------------------------
    # Check folders
    # ---------------------------------------------------------

    if not audio_folder.exists():
        logger.error(f"Audio folder not found: {audio_folder}")
        return None

    if not transcript_folder.exists():
        logger.error(f"Transcript folder not found: {transcript_folder}")
        return None

    files = []

    temp_id = 1
    audio_count = 0
    transcript_count = 0

    # ---------------------------------------------------------
    # Scan Audio Files
    # ---------------------------------------------------------

    logger.info("Scanning audio files...")

    for file in sorted(audio_folder.rglob("*")):

        if file.is_file() and file.suffix.lower() in AUDIO_EXTENSIONS:

            audio_count += 1

            files.append({

                "temp_id": f"TEMP_{temp_id:06d}",

                "file_name": file.name,

                "file_type": "Audio",

                "extension": file.suffix.lower(),

                "relative_path": str(file.relative_to(project_root)),

                "size_bytes": file.stat().st_size

            })

            temp_id += 1

    # ---------------------------------------------------------
    # Scan Transcript Files
    # ---------------------------------------------------------

    logger.info("Scanning transcript files...")

    for file in sorted(transcript_folder.rglob("*")):

        if file.is_file() and file.suffix.lower() in TEXT_EXTENSIONS:

            transcript_count += 1

            files.append({

                "temp_id": f"TEMP_{temp_id:06d}",

                "file_name": file.name,

                "file_type": "Transcript",

                "extension": file.suffix.lower(),

                "relative_path": str(file.relative_to(project_root)),

                "size_bytes": file.stat().st_size

            })

            temp_id += 1

    # ---------------------------------------------------------
    # Create DataFrame
    # ---------------------------------------------------------

    df = pd.DataFrame(files)

    output = project_root / "metadata" / "file_inventory.csv"

    df.to_csv(output, index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------
    # Logging Summary
    # ---------------------------------------------------------

    logger.info("-" * 70)
    logger.info(f"Audio Files      : {audio_count}")
    logger.info(f"Transcript Files : {transcript_count}")
    logger.info(f"Total Files      : {len(df)}")
    logger.info(f"Inventory Saved  : {output}")
    logger.info("=" * 70)

    return df