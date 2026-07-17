from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("Stage6 Report")


def generate_alignment_report(project_root: Path, dataset):

    logger.info("=" * 70)
    logger.info("Generating Alignment Report")
    logger.info("=" * 70)

    reports_folder = project_root / "reports"

    reports_folder.mkdir(parents=True, exist_ok=True)

    report_file = reports_folder / "alignment_report.csv"

    rows = []

    for sample in dataset:

        rows.append({

            "pair_id": sample["pair_id"],

            "audio_file": sample["audio_file"],

            "alignment_ready": sample["alignment_ready"],

            "preparation_status": sample["preparation_status"],

            "alignment_status": sample["alignment_status"],

            "alignment_score": sample["alignment_score"],

            "alignment_validation": sample["alignment_validation"]

        })

    df = pd.DataFrame(rows)

    df.to_csv(report_file, index=False)

    logger.info(f"Report saved to: {report_file}")

    logger.info(f"Total files: {len(df)}")

    logger.info("=" * 70)