"""
Rename Executor

Apply rename_mapping.csv
"""

from pathlib import Path
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("Rename Executor")


def apply_rename():

    project_root = Path(__file__).resolve().parents[2]

    mapping_file = project_root / "metadata" / "rename_mapping.csv"

    if not mapping_file.exists():

        logger.error("rename_mapping.csv not found.")

        return

    mapping = pd.read_csv(mapping_file)

    renamed = 0

    for _, row in mapping.iterrows():

        relative_path = Path(row["relative_path"])

        old_file = project_root / relative_path

        new_file = old_file.with_name(row["new_name"])

        if old_file.exists():

            old_file.rename(new_file)

            renamed += 1

            logger.info(f"{old_file.name} -> {new_file.name}")

        else:

            logger.warning(f"Missing file: {old_file}")

    logger.info(f"Total renamed files : {renamed}")