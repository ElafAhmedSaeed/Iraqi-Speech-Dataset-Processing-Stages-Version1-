"""
Stage 1
Rename Planner

Generate rename mapping without modifying files.
"""

from pathlib import Path
import pandas as pd
import re

from src.utils.logger import get_logger

logger = get_logger("Rename Planner")


def generate_rename_plan():
    """
    Generate rename_mapping.csv without renaming files.
    """

    project_root = Path(__file__).resolve().parents[2]

    inventory = project_root / "metadata" / "file_inventory.csv"

    if not inventory.exists():
        logger.error("file_inventory.csv not found.")
        return

    logger.info("=" * 70)
    logger.info("Generating Rename Plan...")
    logger.info("=" * 70)

    df = pd.read_csv(inventory)

    mapping = []

    for _, row in df.iterrows():

        old_name = row["file_name"]
        extension = row["extension"]
        file_type = row["file_type"]
        relative_path = row["relative_path"]

        # -----------------------------------------------------
        # Extract numeric part from filename
        # Example:
        # IQA_000123.wav -> 123
        # IQT_000123.txt -> 123
        # -----------------------------------------------------

        match = re.search(r"(\d+)", old_name)

        if not match:
            logger.warning(f"Cannot extract number from: {old_name}")
            continue

        number = int(match.group(1))

        # -----------------------------------------------------
        # Unified filename
        # IQ_000001.wav
        # IQ_000001.txt
        # -----------------------------------------------------

        new_name = f"IQ_{number:06d}{extension}"

        mapping.append({

            "old_name": old_name,
            "new_name": new_name,
            "file_type": file_type,
            "relative_path": relative_path

        })

    mapping_df = pd.DataFrame(mapping)

    output = project_root / "metadata" / "rename_mapping.csv"

    mapping_df.to_csv(
        output,
        index=False,
        encoding="utf-8-sig"
    )

    logger.info(f"Rename plan generated successfully.")
    logger.info(f"Total files : {len(mapping_df)}")
    logger.info(f"Saved to    : {output}")

    logger.info("=" * 70)

    return mapping_df


if __name__ == "__main__":
    generate_rename_plan()