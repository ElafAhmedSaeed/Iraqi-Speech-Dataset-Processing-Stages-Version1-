from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("Stage5 Report")


def generate_segmentation_report(project_root: Path, dataset):

    logger.info("=" * 70)
    logger.info("Generating Segmentation Report")
    logger.info("=" * 70)

    reports_folder = project_root / "reports"

    reports_folder.mkdir(exist_ok=True)

    report_file = reports_folder / "segmentation_report.csv"

    rows = []

    for audio in dataset:

        rows.append({

            "audio_file": audio["file_name"],

            "dataset_type": audio["dataset_type"],

            "segmentation_required": audio["segmentation_required"],

            "segmentation_decision": audio["decision"],

            "segments": audio["segments"],

            "segmentation_status": audio["segmentation_status"],

            "segment_validation": audio["segment_validation"]

        })

    df = pd.DataFrame(rows)

    df.to_csv(report_file, index=False)

    logger.info(f"Report saved to: {report_file}")

    logger.info(f"Total files: {len(df)}")

    logger.info("=" * 70)