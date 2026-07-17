from pathlib import Path

from src.utils.logger import get_logger

from src.stage10.training_resources_loader import load_training_resources
from src.stage10.asr_dataset import build_asr_datasets
from src.stage10.batch_collate import test_collate_function
from src.stage10.data_loader_builder import build_data_loaders
from src.stage10.training_batch_validator import validate_training_batches
from src.stage10.training_setup_report import generate_training_setup_report

logger = get_logger("Stage10")


def run_stage10():

    project_root = Path(__file__).resolve().parents[2]

    logger.info("=" * 70)
    logger.info("Stage 10 - Training Setup & Batch Preparation")
    logger.info("=" * 70)

    # --------------------------------------------------
    # Step 1
    # --------------------------------------------------

    logger.info("Step 1 : Load Training Resources")

    training_resources, resource_summary = load_training_resources(
        project_root
    )

    if resource_summary["status"] != "Loaded":

        logger.warning("Training resources loading failed. Stage 10 stopped.")
        logger.warning(resource_summary["message"])

        return

    logger.info("Training resources loaded successfully.")

    logger.info(f"Train samples      : {len(training_resources['train_df'])}")
    logger.info(f"Validation samples : {len(training_resources['val_df'])}")
    logger.info(f"Test samples       : {len(training_resources['test_df'])}")
    logger.info(f"Vocabulary size    : {resource_summary['vocab_size']}")

    # --------------------------------------------------
    # Step 2
    # --------------------------------------------------

    logger.info("Step 2 : Build ASR Dataset")

    train_dataset, val_dataset, test_dataset, dataset_summary = build_asr_datasets(
        project_root,
        training_resources
    )

    logger.info(f"Train dataset      : {len(train_dataset)}")
    logger.info(f"Validation dataset : {len(val_dataset)}")
    logger.info(f"Test dataset       : {len(test_dataset)}")

    if dataset_summary["status"] != "Built":

        logger.warning("ASR dataset building failed. Stage 10 stopped.")

        return

    sample = train_dataset[0]

    logger.info(f"Sample feature shape : {sample['features'].shape}")
    logger.info(f"Sample label length  : {sample['label_length']}")
    logger.info(f"Sample input length  : {sample['input_length']}")

    # --------------------------------------------------
    # Step 3
    # --------------------------------------------------

    logger.info("Step 3 : Create Batch Collate Function")

    collate_results, collate_summary = test_collate_function(
        train_dataset,
        val_dataset,
        test_dataset,
        blank_id=training_resources["blank_id"],
        batch_size=4
    )

    logger.info(
        f"Collate function status: {collate_summary['status']}"
    )

    if collate_summary["status"] != "Passed":

        logger.warning("Batch collate function test failed. Stage 10 stopped.")

        return

    # --------------------------------------------------
    # Step 4
    # --------------------------------------------------

    logger.info("Step 4 : Build DataLoaders")

    data_loaders, loader_summary = build_data_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        blank_id=training_resources["blank_id"],
        batch_size=4,
        num_workers=0
    )

    logger.info(f"DataLoader status : {loader_summary['status']}")
    logger.info(f"Total batches     : {loader_summary['total_batches']}")

    if loader_summary["status"] != "Built":

        logger.warning("DataLoaders building failed. Stage 10 stopped.")

        return

    # --------------------------------------------------
    # Step 5
    # --------------------------------------------------

    logger.info("Step 5 : Validate Training Batches")

    batch_validation_summary = validate_training_batches(
        data_loaders,
        max_batches=3
    )

    logger.info(
        f"Training batch validation status: "
        f"{batch_validation_summary['status']}"
    )

    if batch_validation_summary["status"] != "Passed":

        logger.warning("Training batch validation failed. Report will still be generated.")

    # --------------------------------------------------
    # Step 6
    # --------------------------------------------------

    logger.info("Step 6 : Generate Training Setup Report")

    generate_training_setup_report(
        project_root,
        resource_summary,
        dataset_summary,
        collate_summary,
        loader_summary,
        batch_validation_summary
    )

    logger.info("=" * 70)
    logger.info("Stage 10 Finished")
    logger.info("=" * 70)


if __name__ == "__main__":

    run_stage10()