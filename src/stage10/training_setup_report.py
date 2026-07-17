from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("Stage10 Training Setup Report")


def generate_training_setup_report(
    project_root: Path,
    resource_summary,
    dataset_summary,
    collate_summary,
    loader_summary,
    batch_validation_summary
):

    logger.info("=" * 70)
    logger.info("Generating Training Setup Report")
    logger.info("=" * 70)

    reports_folder = project_root / "reports"

    reports_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    report_file = reports_folder / "training_setup_report.csv"

    # --------------------------------------------------
    # Determine final readiness
    # --------------------------------------------------

    resources_ready = resource_summary.get("status", "") == "Loaded"

    datasets_ready = dataset_summary.get("status", "") == "Built"

    collate_ready = collate_summary.get("status", "") == "Passed"

    loaders_ready = loader_summary.get("status", "") == "Built"

    batches_ready = batch_validation_summary.get("status", "") == "Passed"

    final_ready = (
        resources_ready
        and datasets_ready
        and collate_ready
        and loaders_ready
        and batches_ready
    )

    final_status = "Ready for Training" if final_ready else "Not Ready"

    # --------------------------------------------------
    # Main report data
    # --------------------------------------------------

    report_data = [{

        "stage": "Stage 10",

        "stage_name": "Training Setup & Batch Preparation",

        # -------------------------------
        # Training resources
        # -------------------------------

        "resources_status": resource_summary.get("status", ""),

        "resources_message": resource_summary.get("message", ""),

        "train_rows": resource_summary.get("train_rows", 0),

        "val_rows": resource_summary.get("val_rows", 0),

        "test_rows": resource_summary.get("test_rows", 0),

        "total_rows": resource_summary.get("total_rows", 0),

        "vocab_size": resource_summary.get("vocab_size", 0),

        # -------------------------------
        # Dataset objects
        # -------------------------------

        "dataset_status": dataset_summary.get("status", ""),

        "dataset_message": dataset_summary.get("message", ""),

        "train_dataset_records": dataset_summary.get("train_records", 0),

        "val_dataset_records": dataset_summary.get("val_records", 0),

        "test_dataset_records": dataset_summary.get("test_records", 0),

        "total_dataset_records": dataset_summary.get("total_records", 0),

        # -------------------------------
        # Collate function
        # -------------------------------

        "collate_status": collate_summary.get("status", ""),

        "collate_message": collate_summary.get("message", ""),

        "collate_tested_splits": collate_summary.get("tested_splits", 0),

        "collate_failed_splits": collate_summary.get("failed_splits", 0),

        "collate_batch_size": collate_summary.get("batch_size", 0),

        # -------------------------------
        # DataLoaders
        # -------------------------------

        "loader_status": loader_summary.get("status", ""),

        "loader_message": loader_summary.get("message", ""),

        "batch_size": loader_summary.get("batch_size", 0),

        "num_workers": loader_summary.get("num_workers", 0),

        "torch_available": loader_summary.get("torch_available", ""),

        "total_batches": loader_summary.get("total_batches", 0),

        "train_num_batches": loader_summary.get(
            "train_summary",
            {}
        ).get("num_batches", 0),

        "val_num_batches": loader_summary.get(
            "val_summary",
            {}
        ).get("num_batches", 0),

        "test_num_batches": loader_summary.get(
            "test_summary",
            {}
        ).get("num_batches", 0),

        # -------------------------------
        # Batch validation
        # -------------------------------

        "batch_validation_status": batch_validation_summary.get("status", ""),

        "batch_validation_message": batch_validation_summary.get("message", ""),

        "checked_batches": batch_validation_summary.get("total_checked_batches", 0),

        "valid_batches": batch_validation_summary.get("total_valid_batches", 0),

        "invalid_batches": batch_validation_summary.get("total_invalid_batches", 0),

        # -------------------------------
        # Final status
        # -------------------------------

        "final_status": final_status,

        "ready_for_training": final_ready

    }]

    report_df = pd.DataFrame(report_data)

    report_df.to_csv(
        report_file,
        index=False,
        encoding="utf-8-sig"
    )

    logger.info(f"Training setup report saved to : {report_file}")
    logger.info(f"Resources ready               : {resources_ready}")
    logger.info(f"Datasets ready                : {datasets_ready}")
    logger.info(f"Collate ready                 : {collate_ready}")
    logger.info(f"DataLoaders ready             : {loaders_ready}")
    logger.info(f"Batches ready                 : {batches_ready}")
    logger.info(f"Final status                  : {final_status}")

    logger.info("=" * 70)

    return report_df