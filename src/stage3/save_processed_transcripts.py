from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("Save Processed")


def save_processed_transcripts(project_root, transcripts):

    logger.info("=" * 70)
    logger.info("Saving Processed Transcripts")
    logger.info("=" * 70)

    output_folder = project_root / "data" / "processed" / "transcripts"

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    report = []

    for item in transcripts:

        output_file = output_folder / item["file_name"]

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(item["processed_text"])

        report.append({

            "file_name": item["file_name"],

            "characters": item["characters"],

            "words": item["words"],

            "status": item["status"],

            "message": item["message"]

        })

    report_df = pd.DataFrame(report)

    report_path = project_root / "metadata" / "transcript_processing_report.csv"

    report_df.to_csv(
        report_path,
        index=False,
        encoding="utf-8-sig"
    )

    logger.info(f"Processed transcripts : {len(report)}")

    logger.info(f"Saved to : {output_folder}")

    logger.info(f"Report : {report_path}")

    return report_df