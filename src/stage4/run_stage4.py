from pathlib import Path

from src.utils.logger import get_logger

from src.stage4.load_audio import load_audio_files
from src.stage4.audio_standardizer import standardize_audio
from src.stage4.audio_enhancer import enhance_audio
from src.stage4.silence_processor import process_silence
from src.stage4.audio_validator import validate_audio
from src.stage4.save_processed_audio import save_processed_audio
from src.stage4.update_metadata_stage4 import update_metadata_stage4
from src.stage4.processing_report import generate_processing_report

logger = get_logger("Stage 4")


def run_stage4():

    logger.info("=" * 70)
    logger.info("Stage 4 Started")
    logger.info("=" * 70)

    project_root = Path(__file__).resolve().parents[2]

    # --------------------------------------------------
    # Step 1
    # --------------------------------------------------

    logger.info("Step 1 : Load Audio Files")

    audio_files = load_audio_files(project_root)

    logger.info(f"Loaded {len(audio_files)} audio file(s).")

    # --------------------------------------------------
    # Step 2
    # --------------------------------------------------

    logger.info("Step 2 : Audio Standardization")

    standardized_audio = []

    for audio_file in audio_files:
        result = standardize_audio(audio_file)

        standardized_audio.append(result)

    logger.info(

        f"Standardized {len(standardized_audio)} audio file(s)."

    )

    # --------------------------------------------------
    # Step 3
    # --------------------------------------------------

    logger.info("Step 3 : Audio Quality Enhancement")

    enhanced_audio = []

    for audio_file in standardized_audio:
        result = enhance_audio(audio_file)

        enhanced_audio.append(result)

    logger.info(

        f"Enhanced {len(enhanced_audio)} audio file(s)."

    )

    # --------------------------------------------------
    # Step 4
    # --------------------------------------------------

    logger.info("Step 4 : Silence Processing")

    processed_audio = []

    for audio_file in enhanced_audio:
        result = process_silence(audio_file)

        processed_audio.append(result)

    logger.info(

        f"Processed silence for {len(processed_audio)} audio file(s)."

    )

    # --------------------------------------------------
    # Step 5
    # --------------------------------------------------

    logger.info("Step 5 : Audio Quality Validation")

    validated_audio = []

    for audio_file in processed_audio:
        result = validate_audio(audio_file)

        validated_audio.append(result)

    logger.info(

        f"Validated {len(validated_audio)} audio file(s)."

    )

    # --------------------------------------------------
    # Step 6
    # --------------------------------------------------

    logger.info("Step 6 : Save Processed Audio")

    report = save_processed_audio(

        project_root,

        validated_audio

    )

    logger.info(

        f"Saved {len(report)} processed audio file(s)."

    )

    # --------------------------------------------------
    # Step 7
    # --------------------------------------------------

    logger.info("Step 7 : Update Metadata")

    update_metadata_stage4(

        project_root,

        validated_audio

    )

    # --------------------------------------------------
    # Step 8
    # --------------------------------------------------

    logger.info("Step 8 : Generate Processing Report")

    generate_processing_report(

        project_root,

        validated_audio

    )


    logger.info("=" * 70)
    logger.info("Stage 4 Finished")
    logger.info("=" * 70)