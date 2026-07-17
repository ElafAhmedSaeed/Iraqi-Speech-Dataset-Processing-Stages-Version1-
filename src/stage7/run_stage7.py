from pathlib import Path

from src.utils.logger import get_logger

from src.stage7.load_valid_dataset import load_valid_dataset
from src.stage7.feature_preparation import prepare_feature_extraction
from src.stage7.feature_extractor import extract_audio_features
from src.stage7.feature_validator import validate_features
from src.stage7.save_features import save_features
from src.stage7.update_metadata_stage7 import update_metadata_stage7
from src.stage7.feature_report import generate_feature_report

logger = get_logger("Stage7")


# --------------------------------------------------
# Feature Configuration
# --------------------------------------------------

FEATURE_TYPE = "mfcc"

# Available options:
# "mfcc"
# "mel"
# "combined"


def run_stage7():

    project_root = Path(__file__).resolve().parents[2]

    logger.info("=" * 70)
    logger.info("Stage 7 - Feature Extraction")
    logger.info("=" * 70)

    logger.info(f"Selected Feature Type : {FEATURE_TYPE}")

    # --------------------------------------------------
    # Step 1
    # --------------------------------------------------

    logger.info("Step 1 : Load Valid Dataset")

    valid_dataset = load_valid_dataset(project_root)

    logger.info(f"Loaded valid samples: {len(valid_dataset)}")

    if len(valid_dataset) == 0:

        logger.warning("No valid samples found. Stage 7 stopped.")

        return

    # --------------------------------------------------
    # Step 2
    # --------------------------------------------------

    logger.info("Step 2 : Feature Preparation")

    feature_config = prepare_feature_extraction(
        project_root,
        FEATURE_TYPE
    )

    if feature_config is None:

        logger.warning("Feature configuration failed. Stage 7 stopped.")

        return

    logger.info("Feature configuration prepared successfully.")

    # --------------------------------------------------
    # Step 3
    # --------------------------------------------------

    logger.info("Step 3 : Audio Feature Extraction")

    extracted_features = extract_audio_features(
        valid_dataset,
        feature_config
    )

    logger.info(f"Feature extraction results: {len(extracted_features)}")

    # --------------------------------------------------
    # Step 4
    # --------------------------------------------------

    logger.info("Step 4 : Feature Validation")

    validated_features = validate_features(
        extracted_features
    )

    valid_count = sum(
        1 for item in validated_features
        if item.get("feature_valid", False)
    )

    logger.info(f"Valid features: {valid_count}")

    # --------------------------------------------------
    # Step 5
    # --------------------------------------------------

    logger.info("Step 5 : Save Features")

    saved_features = save_features(
        project_root,
        validated_features,
        feature_config
    )

    saved_count = sum(
        1 for item in saved_features
        if item.get("feature_saved", False)
    )

    logger.info(f"Saved feature files: {saved_count}")

    # --------------------------------------------------
    # Step 6
    # --------------------------------------------------

    logger.info("Step 6 : Update Metadata")

    update_metadata_stage7(
        project_root,
        saved_features
    )

    # --------------------------------------------------
    # Step 7
    # --------------------------------------------------

    logger.info("Step 7 : Generate Feature Report")

    generate_feature_report(
        project_root,
        saved_features
    )

    logger.info("=" * 70)
    logger.info("Stage 7 Finished")
    logger.info("=" * 70)