from pathlib import Path

from src.utils.logger import get_logger

from src.stage5.audio_loader import load_processed_audio
from src.stage5.dataset_analyzer import analyze_dataset
from src.stage5.segmentation_decision import segmentation_decision
from src.stage5.dataset_detector import detect_dataset_type
from src.stage5.audio_segmentation import run_audio_segmentation
from src.stage5.segment_validator import validate_segments
from src.stage5.update_metadata_stage5 import update_metadata_stage5
from src.stage5.segmentation_report import generate_segmentation_report

logger = get_logger("Stage5")


def run_stage5():

    logger.info("=" * 70)
    logger.info("Stage 5 Started")
    logger.info("=" * 70)

    project_root = Path(__file__).resolve().parents[2]

    # --------------------------------------------------
    # Step 1
    # --------------------------------------------------

    logger.info("Step 1 : Load Processed Audio")

    processed_audio = load_processed_audio(project_root)

    logger.info(f"Loaded {len(processed_audio)} audio file(s).")

    # --------------------------------------------------
    # Step 2
    # --------------------------------------------------

    logger.info("Step 2 : Analyze Audio Dataset")

    analyzed_dataset = analyze_dataset(

        processed_audio

    )

    # --------------------------------------------------
    # Step 3
    # --------------------------------------------------

    logger.info("Step 3 : Detect Dataset Type")

    detected_dataset = detect_dataset_type(

        analyzed_dataset

    )

    # --------------------------------------------------
    # Step 4
    # --------------------------------------------------

    logger.info("Step 4 : Segmentation Decision")

    decision_dataset = segmentation_decision(

        detected_dataset

    )

    # --------------------------------------------------
    # Step 5
    # --------------------------------------------------

    logger.info("Step 5 : Audio Segmentation")

    segmented_dataset = run_audio_segmentation(

        decision_dataset

    )

    # --------------------------------------------------
    # Step 6
    # --------------------------------------------------

    logger.info("Step 6 : Segment Validation")

    validated_dataset = validate_segments(

        segmented_dataset

    )

    # --------------------------------------------------
    # Step 7
    # --------------------------------------------------

    logger.info("Step 7 : Update Metadata")

    update_metadata_stage5(

        project_root,

        validated_dataset

    )

    # --------------------------------------------------
    # Step 8
    # --------------------------------------------------

    logger.info("Step 8 : Generate Segmentation Report")

    generate_segmentation_report(

        project_root,

        validated_dataset

    )

    logger.info("=" * 70)
    logger.info("Stage 5 Finished")
    logger.info("=" * 70)