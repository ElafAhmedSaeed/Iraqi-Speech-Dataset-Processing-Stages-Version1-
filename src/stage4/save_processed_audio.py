from pathlib import Path

import pandas as pd
import soundfile as sf

from src.utils.logger import get_logger

logger = get_logger("Save Processed Audio")


def save_processed_audio(project_root: Path, audio_files):

    logger.info("=" * 70)
    logger.info("Saving Processed Audio Files")
    logger.info("=" * 70)

    output_folder = project_root / "data" / "processed_audio"

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    report = []

    for item in audio_files:

        output_file = output_folder / item["file_name"]

        try:

            sf.write(

                file=str(output_file),

                data=item["audio"],

                samplerate=item["sample_rate"],

                subtype="PCM_16"

            )

            report.append({

                "file_name": item["file_name"],

                "sample_rate": item["sample_rate"],

                "channels": item["channels"],

                "duration": round(item["duration"], 3),

                "samples": item["samples"],

                "validation_status": item["validation_status"],

                "issues": item["issues"],

                "warnings": item["warnings"]

            })

        except Exception as e:

            logger.error(f"Cannot save {item['file_name']}")

            logger.error(str(e))

    report_df = pd.DataFrame(report)

    report_path = project_root / "metadata" / "audio_processing_report.csv"

    report_df.to_csv(

        report_path,

        index=False,

        encoding="utf-8-sig"

    )

    logger.info(f"Saved {len(report_df)} audio file(s).")

    logger.info(f"Output Folder : {output_folder}")

    logger.info(f"Report : {report_path}")

    logger.info("=" * 70)

    return report_df