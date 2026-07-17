from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger("Load Transcripts")


def load_transcripts(project_root: Path):

    logger.info("=" * 70)
    logger.info("Loading Transcript Files...")
    logger.info("=" * 70)

    transcript_folder = project_root / "data" / "raw" / "transcripts"

    transcripts = []

    txt_files = sorted(transcript_folder.glob("*.txt"))

    if len(txt_files) == 0:

        logger.warning("No transcript files found.")

        return transcripts

    for txt_file in txt_files:

        try:

            text = txt_file.read_text(
                encoding="utf-8"
            )

            transcripts.append({

                "file_name": txt_file.name,

                "file_path": str(txt_file),

                "text": text

            })

        except Exception as e:

            logger.error(f"Cannot read {txt_file.name}")

            logger.error(str(e))

    logger.info(f"Loaded {len(transcripts)} transcript(s).")

    logger.info("=" * 70)

    return transcripts