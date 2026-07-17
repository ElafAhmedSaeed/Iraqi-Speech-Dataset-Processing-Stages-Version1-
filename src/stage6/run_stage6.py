from pathlib import Path

from src.utils.logger import get_logger

from src.stage6.load_audio_transcript import load_audio_transcript
from src.stage6.alignment_preparation import prepare_alignment
from src.stage6.forced_alignment import run_forced_alignment
from src.stage6.alignment_validator import validate_alignment
from src.stage6.export_alignment import export_alignment
from src.stage6.update_metadata_stage6 import update_metadata_stage6
from src.stage6.alignment_report import generate_alignment_report

logger = get_logger("Stage6")


def run_stage6():

    # Project Root
    project_root = Path(__file__).resolve().parents[2]

    logger.info("=" * 70)
    logger.info("Stage 6 - Forced Alignment")
    logger.info("=" * 70)

    # --------------------------------------------------
    # Step 1
    # --------------------------------------------------

    logger.info("Step 1 : Load Audio & Transcript")

    dataset = load_audio_transcript(project_root)

    logger.info(f"Loaded dataset size: {len(dataset)}")

    # --------------------------------------------------
    # Step 2
    # --------------------------------------------------

    logger.info("Step 2 : Alignment Preparation")

    prepared_dataset = prepare_alignment(dataset)

    # --------------------------------------------------
    # Step 3
    # --------------------------------------------------

    logger.info("Step 3 : Forced Alignment")

    aligned_dataset = run_forced_alignment(prepared_dataset)

    # --------------------------------------------------
    # Step 4
    # --------------------------------------------------

    logger.info("Step 4 : Alignment Validation")

    validated_dataset = validate_alignment(aligned_dataset)

    # --------------------------------------------------
    # Step 5
    # --------------------------------------------------

    logger.info("Step 5 : Export Alignment Files")

    export_alignment(
        project_root,
        validated_dataset
    )

    # --------------------------------------------------
    # Step 6
    # --------------------------------------------------

    logger.info("Step 6 : Update Metadata")

    update_metadata_stage6(
        project_root,
        validated_dataset
    )

    # --------------------------------------------------
    # Step 7
    # --------------------------------------------------

    logger.info("Step 7 : Generate Alignment Report")

    generate_alignment_report(
        project_root,
        validated_dataset
    )

    logger.info("=" * 70)
    logger.info("Stage 6 Finished")
    logger.info("=" * 70)