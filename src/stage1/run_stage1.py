"""
Stage 1 Runner
Data Organization
"""

from src.utils.logger import get_logger

# Step 1
from src.stage1.create_structure import create_project_structure

# Step 2
from src.stage1.discover_files import discover_files

# Step 3
from src.stage1.rename_planner import generate_rename_plan

# Step 4
from src.stage1.rename_executor import apply_rename

# Step 5
from src.stage1.pair_generator import generate_pair_index

# Step 6
from src.stage1.metadata_builder import build_metadata


logger = get_logger("Stage 1")


def run_stage1():

    logger.info("=" * 70)
    logger.info("Stage 1 Started : Data Organization")
    logger.info("=" * 70)

    try:

        # -------------------------------------------------
        # Step 1
        # -------------------------------------------------
        logger.info("Step 1 : Create Project Structure")
        create_project_structure()

        # -------------------------------------------------
        # Step 2
        # -------------------------------------------------
        logger.info("Step 2 : Discover Files")
        discover_files()

        # -------------------------------------------------
        # Step 3
        # -------------------------------------------------
        logger.info("Step 3 : Generate Rename Plan")
        generate_rename_plan()

        # -------------------------------------------------
        # Step 4
        # -------------------------------------------------
        logger.info("Step 4 : Apply Rename")
        apply_rename()

        # -------------------------------------------------
        # Step 5
        # -------------------------------------------------
        logger.info("Step 5 : Generate Pair Index")
        generate_pair_index()

        # -------------------------------------------------
        # Step 6
        # -------------------------------------------------
        logger.info("Step 6 : Build Metadata")
        build_metadata()

        logger.info("=" * 70)
        logger.info("Stage 1 Completed Successfully")
        logger.info("=" * 70)

    except Exception as e:

        logger.exception("Stage 1 Failed")
        raise e


if __name__ == "__main__":
    run_stage1()