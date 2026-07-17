from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("Stage7 Feature Report")


def generate_feature_report(project_root: Path, saved_features):

    logger.info("=" * 70)
    logger.info("Generating Feature Extraction Report")
    logger.info("=" * 70)

    reports_folder = project_root / "reports"

    reports_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    report_file = reports_folder / "feature_report.csv"

    report_rows = []

    total = len(saved_features)

    saved = 0
    skipped = 0
    failed = 0
    validation_passed = 0
    validation_failed = 0

    for item in saved_features:

        feature_saved = bool(
            item.get("feature_saved", False)
        )

        feature_valid = bool(
            item.get("feature_valid", False)
        )

        save_status = str(
            item.get("save_status", "")
        )

        validation_status = str(
            item.get("validation_status", "")
        )

        if feature_saved:
            saved += 1

        elif save_status == "Skipped":
            skipped += 1

        else:
            failed += 1

        if feature_valid:
            validation_passed += 1

        else:
            validation_failed += 1

        report_rows.append({

            "pair_id": item.get("pair_id", ""),

            "audio_file": item.get("audio_file", ""),

            "feature_type": item.get("feature_type", ""),

            "feature_file": item.get("feature_file", ""),

            "feature_file_name": item.get("feature_file_name", ""),

            "feature_shape": str(item.get("feature_shape", "")),

            "feature_saved": feature_saved,

            "feature_status": save_status,

            "feature_message": item.get("save_message", ""),

            "feature_valid": feature_valid,

            "validation_status": validation_status,

            "validation_message": item.get("validation_message", ""),

            "duration": item.get("duration", ""),

            "sample_rate": item.get("sample_rate", "")

        })

    report_df = pd.DataFrame(report_rows)

    report_df.to_csv(
        report_file,
        index=False,
        encoding="utf-8-sig"
    )

    logger.info(f"Total features       : {total}")
    logger.info(f"Saved features       : {saved}")
    logger.info(f"Skipped features     : {skipped}")
    logger.info(f"Failed features      : {failed}")
    logger.info(f"Validation passed    : {validation_passed}")
    logger.info(f"Validation failed    : {validation_failed}")
    logger.info(f"Report saved to      : {report_file}")

    logger.info("=" * 70)

    return report_df