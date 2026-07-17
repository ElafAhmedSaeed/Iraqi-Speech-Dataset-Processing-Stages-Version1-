from pathlib import Path

from src.utils.logger import get_logger

from src.stage3.load_transcripts import load_transcripts
from src.stage3.text_cleaner import clean_text
from src.stage3.text_normalizer import normalize_text
from src.stage3.iraqi_processor import process_iraqi_text
from src.stage3.transcript_quality import validate_transcript
from src.stage3.save_processed_transcripts import save_processed_transcripts
from src.stage3.update_metadata_stage3 import update_metadata_stage3

logger = get_logger("Stage 3")


def run_stage3():

    logger.info("=" * 70)
    logger.info("Stage 3 Started")
    logger.info("=" * 70)

    project_root = Path(__file__).resolve().parents[2]

    # --------------------------------------------------
    # Step 1
    # --------------------------------------------------

    logger.info("Step 1 : Load Transcripts")

    transcripts = load_transcripts(project_root)

    # --------------------------------------------------
    # Step 2
    # --------------------------------------------------

    logger.info("Step 2 : Text Cleaning")

    cleaned_transcripts = []

    for transcript in transcripts:

        result = clean_text(
            transcript["text"]
        )

        cleaned_transcripts.append({

            "file_name": transcript["file_name"],
            "file_path": transcript["file_path"],
            "original_text": result["original_text"],
            "clean_text": result["clean_text"],
            "changed": result["changed"]

        })

    logger.info(
        f"Cleaned {len(cleaned_transcripts)} transcript(s)."
    )

    # --------------------------------------------------
    # Step 3
    # --------------------------------------------------

    logger.info("Step 3 : Text Normalization")

    normalized_transcripts = []

    for transcript in cleaned_transcripts:

        result = normalize_text(
            transcript["clean_text"]
        )

        normalized_transcripts.append({

            "file_name": transcript["file_name"],
            "file_path": transcript["file_path"],
            "original_text": transcript["original_text"],
            "clean_text": transcript["clean_text"],
            "normalized_text": result["normalized_text"],
            "changed": result["changed"]

        })

    logger.info(
        f"Normalized {len(normalized_transcripts)} transcript(s)."
    )

    # --------------------------------------------------
    # Step 4
    # --------------------------------------------------

    logger.info("Step 4 : Iraqi Dialect Processing")

    processed_transcripts = []

    for transcript in normalized_transcripts:

        result = process_iraqi_text(
            transcript["normalized_text"]
        )

        processed_transcripts.append({

            "file_name": transcript["file_name"],
            "file_path": transcript["file_path"],
            "original_text": transcript["original_text"],
            "clean_text": transcript["clean_text"],
            "normalized_text": transcript["normalized_text"],
            "processed_text": result["processed_text"],
            "changed": result["changed"]

        })

    logger.info(
        f"Iraqi processing completed for {len(processed_transcripts)} transcript(s)."
    )

    # --------------------------------------------------
    # Step 5
    # --------------------------------------------------

    logger.info("Step 5 : Transcript Quality Validation")

    validated_transcripts = []

    for transcript in processed_transcripts:

        result = validate_transcript(
            transcript["processed_text"]
        )

        validated_transcripts.append({

            **transcript,
            **result

        })

    logger.info(
        f"Validated {len(validated_transcripts)} transcript(s)."
    )

    # --------------------------------------------------
    # Step 6
    # --------------------------------------------------

    logger.info("Step 6 : Save Processed Transcripts")

    save_processed_transcripts(
        project_root,
        validated_transcripts
    )

    # --------------------------------------------------
    # Step 7
    # --------------------------------------------------

    logger.info("Step 7 : Update Metadata")

    update_metadata_stage3(
        project_root,
        validated_transcripts
    )

    logger.info(f"Loaded {len(transcripts)} transcript(s).")

    logger.info("=" * 70)
    logger.info("Stage 3 Finished")
    logger.info("=" * 70)