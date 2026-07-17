from pathlib import Path
from datetime import datetime

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("Validation Report")


def build_validation_report(project_root: Path):

    logger.info("=" * 70)
    logger.info("Building Validation Report...")
    logger.info("=" * 70)

    metadata_folder = project_root / "metadata"
    reports_folder = project_root / "reports"

    reports_folder.mkdir(exist_ok=True)

    audio_df = pd.read_csv(metadata_folder / "audio_validation.csv")
    transcript_df = pd.read_csv(metadata_folder / "transcript_validation.csv")
    pair_df = pd.read_csv(metadata_folder / "pair_validation.csv")

    # ---------------------------------------------------------
    # Audio Statistics
    # ---------------------------------------------------------

    total_audio = len(audio_df)

    valid_audio = audio_df["audio_valid"].sum()

    invalid_audio = total_audio - valid_audio

    # ---------------------------------------------------------
    # Transcript Statistics
    # ---------------------------------------------------------

    total_transcripts = len(transcript_df)

    valid_transcripts = (transcript_df["status"] == "Valid").sum()

    invalid_transcripts = total_transcripts - valid_transcripts

    # ---------------------------------------------------------
    # Pair Statistics
    # ---------------------------------------------------------

    total_pairs = len(pair_df)

    valid_pairs = pair_df["pair_valid"].sum()

    invalid_pairs = total_pairs - valid_pairs

    # ---------------------------------------------------------
    # Dataset Status
    # ---------------------------------------------------------

    if invalid_pairs == 0:
        dataset_status = "READY FOR STAGE 3"
    else:
        dataset_status = "DATASET CONTAINS ERRORS"

    # ---------------------------------------------------------
    # Summary CSV
    # ---------------------------------------------------------

    summary = pd.DataFrame({

        "Metric": [

            "Total Audio",

            "Valid Audio",

            "Invalid Audio",

            "Total Transcripts",

            "Valid Transcripts",

            "Invalid Transcripts",

            "Total Pairs",

            "Valid Pairs",

            "Invalid Pairs",

            "Dataset Status"

        ],

        "Value": [

            total_audio,

            valid_audio,

            invalid_audio,

            total_transcripts,

            valid_transcripts,

            invalid_transcripts,

            total_pairs,

            valid_pairs,

            invalid_pairs,

            dataset_status

        ]

    })

    summary_csv = metadata_folder / "validation_summary.csv"

    summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    # ---------------------------------------------------------
    # TXT Report
    # ---------------------------------------------------------

    report_txt = reports_folder / "validation_report.txt"

    with open(report_txt, "w", encoding="utf-8") as f:

        f.write("=" * 60 + "\n")
        f.write("IRAQI SPEECH DATASET\n")
        f.write("VALIDATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Generated : {datetime.now()}\n\n")

        f.write(f"Total Audio Files      : {total_audio}\n")
        f.write(f"Valid Audio Files      : {valid_audio}\n")
        f.write(f"Invalid Audio Files    : {invalid_audio}\n\n")

        f.write(f"Total Transcripts      : {total_transcripts}\n")
        f.write(f"Valid Transcripts      : {valid_transcripts}\n")
        f.write(f"Invalid Transcripts    : {invalid_transcripts}\n\n")

        f.write(f"Total Pairs            : {total_pairs}\n")
        f.write(f"Valid Pairs            : {valid_pairs}\n")
        f.write(f"Invalid Pairs          : {invalid_pairs}\n\n")

        f.write(f"Dataset Status         : {dataset_status}\n")

    logger.info("Validation Report Created Successfully.")

    logger.info(f"Summary CSV : {summary_csv}")

    logger.info(f"TXT Report  : {report_txt}")

    logger.info("=" * 70)