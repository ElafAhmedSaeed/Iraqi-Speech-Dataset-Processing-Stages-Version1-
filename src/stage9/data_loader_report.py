from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("Stage9 Data Loader Report")


def _safe_text(value):

    if value is None:
        return ""

    value = str(value)

    if value.lower() == "nan":
        return ""

    return value.strip()


def _prepare_record_rows(records, split_name):

    rows = []

    for item in records:

        rows.append({

            "pair_id": item.get("pair_id", ""),

            "audio_file": item.get("audio_file", ""),

            "feature_file": item.get("feature_file", ""),

            "feature_type": item.get("feature_type", ""),

            "feature_shape": str(item.get("feature_shape", "")),

            "feature_dim": item.get("feature_dim", ""),

            "feature_length": item.get("feature_length", ""),

            "label_text": item.get("label_text", ""),

            "label_length": item.get("label_length", ""),

            "label_type": item.get("label_type", ""),

            "duration": item.get("duration", ""),

            "sample_rate": item.get("sample_rate", ""),

            "dataset_split": split_name,

            "dataset_object_status": item.get("dataset_object_status", ""),

            "data_loader_valid": item.get("data_loader_valid", ""),

            "data_loader_validation_status": item.get(
                "data_loader_validation_status",
                ""
            ),

            "data_loader_validation_message": item.get(
                "data_loader_validation_message",
                ""
            )

        })

    return rows


def generate_data_loader_report(
    project_root: Path,
    manifest_summary,
    feature_load_summary,
    label_summary,
    dataset_summary,
    data_loader_summary,
    train_validated,
    val_validated,
    test_validated
):

    logger.info("=" * 70)
    logger.info("Generating Data Loader Report")
    logger.info("=" * 70)

    reports_folder = project_root / "reports"

    reports_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_report_file = reports_folder / "data_loader_report.csv"

    records_report_file = reports_folder / "data_loader_records_report.csv"

    # --------------------------------------------------
    # Summary report
    # --------------------------------------------------

    summary_data = [{

        "stage": "Stage 9",

        "stage_name": "Data Loader & Training Interface Preparation",

        "manifest_status": manifest_summary.get("status", ""),

        "manifest_message": manifest_summary.get("message", ""),

        "train_manifest_rows": manifest_summary.get("train_rows", 0),

        "val_manifest_rows": manifest_summary.get("val_rows", 0),

        "test_manifest_rows": manifest_summary.get("test_rows", 0),

        "total_manifest_rows": manifest_summary.get("total_rows", 0),

        "feature_loading_status": feature_load_summary.get("status", ""),

        "feature_loading_message": feature_load_summary.get("message", ""),

        "total_loaded_features": feature_load_summary.get("total_loaded", 0),

        "total_failed_features": feature_load_summary.get("total_failed", 0),

        "label_status": label_summary.get("status", ""),

        "label_message": label_summary.get("message", ""),

        "label_type": label_summary.get("label_type", ""),

        "vocab_size": label_summary.get("vocab_size", 0),

        "vocab_file": label_summary.get("vocab_file", ""),

        "prepared_labels": label_summary.get("prepared_labels", 0),

        "skipped_labels": label_summary.get("skipped_labels", 0),

        "dataset_building_status": dataset_summary.get("status", ""),

        "dataset_building_message": dataset_summary.get("message", ""),

        "train_dataset_objects": dataset_summary.get("train_objects", 0),

        "val_dataset_objects": dataset_summary.get("val_objects", 0),

        "test_dataset_objects": dataset_summary.get("test_objects", 0),

        "total_dataset_objects": dataset_summary.get("total_objects", 0),

        "skipped_dataset_objects": dataset_summary.get("skipped_objects", 0),

        "data_loader_validation_status": data_loader_summary.get("status", ""),

        "data_loader_validation_message": data_loader_summary.get("message", ""),

        "valid_objects": data_loader_summary.get("valid_objects", 0),

        "invalid_objects": data_loader_summary.get("invalid_objects", 0),

        "final_status": (
            "Ready"
            if data_loader_summary.get("status", "") == "Passed"
            else "Not Ready"
        )

    }]

    summary_df = pd.DataFrame(summary_data)

    summary_df.to_csv(
        summary_report_file,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------
    # Detailed records report
    # --------------------------------------------------

    record_rows = []

    record_rows.extend(
        _prepare_record_rows(train_validated, "train")
    )

    record_rows.extend(
        _prepare_record_rows(val_validated, "val")
    )

    record_rows.extend(
        _prepare_record_rows(test_validated, "test")
    )

    records_df = pd.DataFrame(record_rows)

    records_df.to_csv(
        records_report_file,
        index=False,
        encoding="utf-8-sig"
    )

    logger.info(f"Summary report saved to : {summary_report_file}")
    logger.info(f"Records report saved to : {records_report_file}")

    logger.info(f"Total dataset objects   : {data_loader_summary.get('total_objects', 0)}")
    logger.info(f"Valid objects           : {data_loader_summary.get('valid_objects', 0)}")
    logger.info(f"Invalid objects         : {data_loader_summary.get('invalid_objects', 0)}")
    logger.info(f"Final status            : {summary_data[0]['final_status']}")

    logger.info("=" * 70)

    return summary_df, records_df