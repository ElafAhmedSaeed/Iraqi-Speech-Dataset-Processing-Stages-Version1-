from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("Audio Report")


def generate_processing_report(project_root: Path, validated_audio):

    logger.info("=" * 70)
    logger.info("Generating Audio Processing Report")
    logger.info("=" * 70)

    if len(validated_audio) == 0:

        logger.warning("No processed audio found.")

        return

    df = pd.DataFrame(validated_audio)

    report = {

        "Total Files": len(df),

        "Valid Files": len(df[df["validation_status"] == "Valid"]),

        "Warning Files": len(df[df["validation_status"] == "Warning"]),

        "Invalid Files": len(df[df["validation_status"] == "Invalid"]),

        "Average Duration (sec)": round(df["duration"].mean(), 3),

        "Average Sample Rate": round(df["sample_rate"].mean(), 2),

        "Average RMS": round(df["rms"].mean(), 5),

        "Average Silence Ratio": round(df["silence_ratio"].mean(), 5),

        "Average Clipping Ratio": round(df["clipping_ratio"].mean(), 5)

    }

    report_df = pd.DataFrame(

        report.items(),

        columns=[

            "Metric",

            "Value"

        ]

    )

    report_folder = project_root / "reports"

    report_folder.mkdir(

        parents=True,

        exist_ok=True

    )

    report_path = report_folder / "audio_processing_summary.csv"

    report_df.to_csv(

        report_path,

        index=False,

        encoding="utf-8-sig"

    )

    logger.info(f"Report saved to {report_path}")

    logger.info("=" * 70)

    return report_df