from pathlib import Path

from src.utils.logger import get_logger

from src.stage8.load_feature_dataset import load_feature_dataset
from src.stage8.training_record_preparer import prepare_training_records
from src.stage8.dataset_splitter import split_dataset
from src.stage8.manifest_saver import save_dataset_manifests
from src.stage8.split_validator import validate_dataset_splits
from src.stage8.update_metadata_stage8 import update_metadata_stage8
from src.stage8.dataset_report import generate_dataset_report

logger = get_logger("Stage8")


def run_stage8():

    project_root = Path(__file__).resolve().parents[2]

    logger.info("=" * 70)
    logger.info("Stage 8 - Dataset Preparation for Model Training")
    logger.info("=" * 70)

    # --------------------------------------------------
    # Step 1
    # --------------------------------------------------

    logger.info("Step 1 : Load Feature Dataset")

    feature_dataset = load_feature_dataset(project_root)

    logger.info(f"Loaded feature dataset samples: {len(feature_dataset)}")

    if len(feature_dataset) == 0:

        logger.warning("No valid feature samples found. Stage 8 stopped.")

        return

    # --------------------------------------------------
    # Step 2
    # --------------------------------------------------

    logger.info("Step 2 : Prepare Training Records")

    training_records = prepare_training_records(
        feature_dataset
    )

    logger.info(f"Prepared training records: {len(training_records)}")

    if len(training_records) == 0:

        logger.warning("No training records prepared. Stage 8 stopped.")

        return

    # --------------------------------------------------
    # Step 3
    # --------------------------------------------------

    logger.info("Step 3 : Split Dataset")

    train_records, val_records, test_records = split_dataset(
        training_records,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )

    logger.info(f"Train records      : {len(train_records)}")
    logger.info(f"Validation records : {len(val_records)}")
    logger.info(f"Test records       : {len(test_records)}")

    if len(train_records) == 0:

        logger.warning("Train split is empty. Stage 8 stopped.")

        return

    # --------------------------------------------------
    # Step 4
    # --------------------------------------------------

    logger.info("Step 4 : Save Dataset Manifests")

    manifest_info = save_dataset_manifests(
        project_root,
        train_records,
        val_records,
        test_records
    )

    logger.info(
        f"Dataset manifests status: {manifest_info['save_status']}"
    )

    # --------------------------------------------------
    # Step 5
    # --------------------------------------------------

    logger.info("Step 5 : Validate Dataset Splits")

    validation_summary = validate_dataset_splits(
        project_root,
        manifest_info
    )

    logger.info(
        f"Dataset split validation: {validation_summary['validation_status']}"
    )

    if validation_summary["validation_status"] != "Passed":

        logger.warning(
            "Dataset split validation failed. Metadata and report will still be generated."
        )

    # --------------------------------------------------
    # Step 6
    # --------------------------------------------------

    logger.info("Step 6 : Update Metadata")

    update_metadata_stage8(
        project_root,
        train_records,
        val_records,
        test_records,
        validation_summary
    )

    # --------------------------------------------------
    # Step 7
    # --------------------------------------------------

    logger.info("Step 7 : Generate Dataset Report")

    generate_dataset_report(
        project_root,
        train_records,
        val_records,
        test_records,
        manifest_info,
        validation_summary
    )

    logger.info("=" * 70)
    logger.info("Stage 8 Finished")
    logger.info("=" * 70)