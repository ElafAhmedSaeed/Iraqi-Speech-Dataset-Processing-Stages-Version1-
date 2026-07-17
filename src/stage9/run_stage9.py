from pathlib import Path

from src.utils.logger import get_logger

from src.stage9.load_manifests import load_dataset_manifests
from src.stage9.feature_loader import load_feature_files
from src.stage9.text_label_preparer import prepare_text_labels
from src.stage9.dataset_builder import build_dataset_objects
from src.stage9.data_loader_validator import validate_data_loader
from src.stage9.data_loader_report import generate_data_loader_report

logger = get_logger("Stage9")


def run_stage9():

    project_root = Path(__file__).resolve().parents[2]

    logger.info("=" * 70)
    logger.info("Stage 9 - Data Loader & Training Interface Preparation")
    logger.info("=" * 70)

    # --------------------------------------------------
    # Step 1
    # --------------------------------------------------

    logger.info("Step 1 : Load Dataset Manifests")

    train_df, val_df, test_df, manifest_summary = load_dataset_manifests(
        project_root
    )

    if manifest_summary["status"] != "Loaded":

        logger.warning("Dataset manifests loading failed. Stage 9 stopped.")
        logger.warning(manifest_summary["message"])

        return

    logger.info("Dataset manifests loaded successfully.")

    logger.info(f"Train samples      : {len(train_df)}")
    logger.info(f"Validation samples : {len(val_df)}")
    logger.info(f"Test samples       : {len(test_df)}")

    # --------------------------------------------------
    # Step 2
    # --------------------------------------------------

    logger.info("Step 2 : Load Feature Files")

    train_features, val_features, test_features, feature_load_summary = load_feature_files(
        project_root,
        train_df,
        val_df,
        test_df
    )

    logger.info(f"Loaded train features      : {len(train_features)}")
    logger.info(f"Loaded validation features : {len(val_features)}")
    logger.info(f"Loaded test features       : {len(test_features)}")

    if feature_load_summary["total_loaded"] == 0:

        logger.warning("No feature files loaded. Stage 9 stopped.")

        return

    # --------------------------------------------------
    # Step 3
    # --------------------------------------------------

    logger.info("Step 3 : Prepare Text Labels")

    train_labeled, val_labeled, test_labeled, vocab, label_summary = prepare_text_labels(
        project_root,
        train_features,
        val_features,
        test_features
    )

    logger.info(f"Prepared labels : {label_summary['prepared_labels']}")
    logger.info(f"Vocabulary size : {label_summary['vocab_size']}")

    if label_summary["prepared_labels"] == 0:

        logger.warning("No text labels prepared. Stage 9 stopped.")

        return

    # --------------------------------------------------
    # Step 4
    # --------------------------------------------------

    logger.info("Step 4 : Build Dataset Objects")

    train_dataset, val_dataset, test_dataset, dataset_summary = build_dataset_objects(
        train_labeled,
        val_labeled,
        test_labeled
    )

    logger.info(f"Train dataset objects      : {len(train_dataset)}")
    logger.info(f"Validation dataset objects : {len(val_dataset)}")
    logger.info(f"Test dataset objects       : {len(test_dataset)}")

    if dataset_summary["total_objects"] == 0:

        logger.warning("No dataset objects built. Stage 9 stopped.")

        return

    # --------------------------------------------------
    # Step 5
    # --------------------------------------------------

    logger.info("Step 5 : Validate Data Loader")

    train_validated, val_validated, test_validated, data_loader_summary = validate_data_loader(
        train_dataset,
        val_dataset,
        test_dataset
    )

    logger.info(
        f"Data loader validation status: {data_loader_summary['status']}"
    )

    if data_loader_summary["status"] != "Passed":

        logger.warning(
            "Data loader validation failed. Report will still be generated."
        )

    # --------------------------------------------------
    # Step 6
    # --------------------------------------------------

    logger.info("Step 6 : Generate Data Loader Report")

    generate_data_loader_report(
        project_root,
        manifest_summary,
        feature_load_summary,
        label_summary,
        dataset_summary,
        data_loader_summary,
        train_validated,
        val_validated,
        test_validated
    )

    logger.info("=" * 70)
    logger.info("Stage 9 Finished")
    logger.info("=" * 70)


if __name__ == "__main__":

    run_stage9()