from pathlib import Path
import json

from src.utils.logger import get_logger

logger = get_logger("Stage6 Export Alignment")


def export_alignment(project_root: Path, dataset):

    logger.info("=" * 70)
    logger.info("Exporting Alignment Files")
    logger.info("=" * 70)

    output_folder = project_root / "data" / "alignment"

    output_folder.mkdir(parents=True, exist_ok=True)

    exported = 0
    skipped = 0

    for sample in dataset:

        if sample["alignment_validation"] != "Passed":

            skipped += 1
            continue

        output_file = output_folder / (
            Path(sample["audio_file"]).stem + ".json"
        )

        alignment_data = {

            "pair_id": sample["pair_id"],

            "audio_file": sample["audio_file"],

            "alignment_score": sample["alignment_score"],

            "words": sample["alignment_result"]

        }

        with open(output_file, "w", encoding="utf-8") as f:

            json.dump(
                alignment_data,
                f,
                ensure_ascii=False,
                indent=4
            )

        exported += 1

    logger.info(f"Exported : {exported}")

    logger.info(f"Skipped  : {skipped}")

    logger.info("=" * 70)