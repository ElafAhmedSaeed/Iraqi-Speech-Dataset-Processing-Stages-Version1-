from pathlib import Path

from src.utils.logger import get_logger
# ==============================
# Stage 2 Modules
# ==============================

from src.stage2.audio_validator import validate_audio_dataset
from src.stage2.transcript_validator import validate_transcripts
from src.stage2.pair_validator import validate_pairs
from src.stage2.metadata_updater import update_metadata
from src.stage2.validation_report import build_validation_report

logger = get_logger("Stage 2")


def run_stage2():

    logger.info("=" * 70)
    logger.info("Stage 2 Started : Data Validation")
    logger.info("=" * 70)

    # --------------------------------------------------
    # Project Paths
    # --------------------------------------------------

    project_root = Path(__file__).resolve().parents[2]

    metadata_path = project_root / "metadata" / "metadata.csv"

    audio_folder = project_root / "data" / "raw" / "audio"

    transcript_folder = project_root / "data" / "raw" / "transcripts"

    # ==================================================
    # Step 1 : Audio Validation
    # ==================================================

    logger.info("Step 1 : Audio Validation")

    audio_results = validate_audio_dataset(
        metadata_path=metadata_path,
        audio_folder=audio_folder
    )

    audio_output = project_root / "metadata" / "audio_validation.csv"

    audio_results.to_csv(
        audio_output,
        index=False,
        encoding="utf-8-sig"
    )

    logger.info(f"Audio validation saved to: {audio_output}")

    # ==================================================
    # Step 2 : Transcript Validation
    # ==================================================

    logger.info("Step 2 : Transcript Validation")

    transcript_results = validate_transcripts(
        metadata_path=metadata_path,
        transcript_folder=transcript_folder
    )

    transcript_output = project_root / "metadata" / "transcript_validation.csv"

    transcript_results.to_csv(
        transcript_output,
        index=False,
        encoding="utf-8-sig"
    )

    logger.info(f"Transcript validation saved to: {transcript_output}")

    # ==================================================
    # Step 3 : Pair Validation
    # ==================================================

    logger.info("Step 3 : Pair Validation (Coming Soon)")

    # ==================================================
    # Step 3 : Pair Validation
    # ==================================================

    #logger.info("Step 3 : Pair Validation")

    pair_results = validate_pairs(project_root)

    pair_output = project_root / "metadata" / "pair_validation.csv"

    pair_results.to_csv(
        pair_output,
        index=False,
        encoding="utf-8-sig"
    )

    logger.info(f"Pair validation saved to: {pair_output}")

    # ==================================================
    # Step 4 : Metadata Update
    # ==================================================

    #logger.info("Step 4 : Metadata Update (Coming Soon)")

    logger.info("Step 4 : Metadata Update")

    update_metadata(project_root)

    # ==================================================
    # Step 5 : Validation Report
    # ==================================================

    #logger.info("Step 5 : Validation Report (Coming Soon)")

    # --------------------------------------------------
    # Step 5 : Validation Report
    # --------------------------------------------------

    logger.info("Step 5 : Validation Report")

    build_validation_report(project_root)

    logger.info("=" * 70)
    logger.info("Stage 2 Completed Successfully")
    logger.info("=" * 70)