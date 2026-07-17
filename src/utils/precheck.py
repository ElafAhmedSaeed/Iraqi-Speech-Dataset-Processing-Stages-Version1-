from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger("PreCheck")


def check_audio_folder(audio_folder: Path):

    if not audio_folder.exists():

        logger.warning(f"Folder not found: {audio_folder}")

        return False

    audio_files = list(audio_folder.glob("*"))

    if len(audio_files) == 0:

        logger.warning("No audio files found.")

        return False

    return True


def check_transcript_folder(transcript_folder: Path):

    if not transcript_folder.exists():

        logger.warning(f"Folder not found: {transcript_folder}")

        return False

    transcript_files = list(transcript_folder.glob("*.txt"))

    if len(transcript_files) == 0:

        logger.warning("No transcript files found.")

        return False

    return True


def check_metadata(metadata_file: Path):

    if not metadata_file.exists():

        logger.warning("metadata.csv not found.")

        return False

    return True


def stage_skip(stage_name: str, reason: str):

    logger.info("=" * 70)
    logger.info(f"{stage_name} Skipped")
    logger.info(f"Reason : {reason}")
    logger.info("=" * 70)