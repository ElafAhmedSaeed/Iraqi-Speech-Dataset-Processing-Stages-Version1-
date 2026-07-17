from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("Stage8 Dataset Report")


def generate_dataset_report(
    project_root: Path,
    train_records,
    val_records,
    test_records,
    manifest_info,
    validation_summary
):

    logger.info("=" * 70)
    logger.info("Generating Dataset Preparation Report")
    logger.info("=" * 70)

    reports_folder = project_root / "reports"

    reports_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    report_file = reports_folder / "dataset_preparation_report.csv"

    train_count = len(train_records)
    val_count = len(val_records)
    test_count = len(test_records)

    total_count = train_count + val_count + test_count

    if total_count > 0:

        train_ratio = round(train_count / total_count, 4)
        val_ratio = round(val_count / total_count, 4)
        test_ratio = round(test_count / total_count, 4)

    else:

        train_ratio = 0
        val_ratio = 0
        test_ratio = 0

    # --------------------------------------------------
    # Text statistics
    # --------------------------------------------------

    all_records = []

    all_records.extend(train_records)
    all_records.extend(val_records)
    all_records.extend(test_records)

    word_counts = []
    text_lengths = []
    durations = []

    for record in all_records:

        word_counts.append(
            record.get("word_count", 0)
        )

        text_lengths.append(
            record.get("text_length", 0)
        )

        try:

            durations.append(
                float(record.get("duration", 0))
            )

        except Exception:

            durations.append(0)

    avg_words = round(sum(word_counts) / len(word_counts), 2) if word_counts else 0
    avg_text_length = round(sum(text_lengths) / len(text_lengths), 2) if text_lengths else 0
    avg_duration = round(sum(durations) / len(durations), 2) if durations else 0

    # --------------------------------------------------
    # Main report row
    # --------------------------------------------------

    report_data = [{

        "stage": "Stage 8",

        "stage_name": "Dataset Preparation for Model Training",

        "total_records": total_count,

        "train_count": train_count,

        "val_count": val_count,

        "test_count": test_count,

        "train_ratio": train_ratio,

        "val_ratio": val_ratio,

        "test_ratio": test_ratio,

        "average_word_count": avg_words,

        "average_text_length": avg_text_length,

        "average_duration_sec": avg_duration,

        "train_manifest": manifest_info.get("train_manifest", ""),

        "val_manifest": manifest_info.get("val_manifest", ""),

        "test_manifest": manifest_info.get("test_manifest", ""),

        "manifest_save_status": manifest_info.get("save_status", ""),

        "split_validation_status": validation_summary.get("validation_status", ""),

        "split_validation_message": validation_summary.get("validation_message", ""),

        "overlap_found": validation_summary.get("overlap_found", ""),

        "overlap_message": validation_summary.get("overlap_message", ""),

        "dataset_status": (
            "Prepared"
            if validation_summary.get("validation_status", "") == "Passed"
            else "Prepared with warnings"
        )

    }]

    report_df = pd.DataFrame(report_data)

    report_df.to_csv(
        report_file,
        index=False,
        encoding="utf-8-sig"
    )

    logger.info(f"Dataset report saved to : {report_file}")
    logger.info(f"Total records           : {total_count}")
    logger.info(f"Train records           : {train_count}")
    logger.info(f"Validation records      : {val_count}")
    logger.info(f"Test records            : {test_count}")
    logger.info(f"Validation status       : {validation_summary.get('validation_status', '')}")

    logger.info("=" * 70)

    return report_df