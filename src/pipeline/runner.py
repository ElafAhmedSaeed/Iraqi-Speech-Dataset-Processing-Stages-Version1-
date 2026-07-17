from src.utils.logger import get_logger

from src.stage1.run_stage1 import run_stage1
from src.stage2.run_stage2 import run_stage2
from src.stage3.run_stage3 import run_stage3
from src.stage4.run_stage4 import run_stage4
from src.stage5.run_stage5 import run_stage5
from src.stage6.run_stage6 import run_stage6
from src.stage7.run_stage7 import run_stage7
from src.stage8.run_stage8 import run_stage8
from src.stage9.run_stage9 import run_stage9
from src.stage10.run_stage10 import run_stage10

logger = get_logger("Pipeline")


def run_pipeline():

    logger.info("=" * 70)
    logger.info("Pipeline Started")
    logger.info("=" * 70)

    # --------------------------------------------------
    # Stage 1
    # --------------------------------------------------

    run_stage1()

    # --------------------------------------------------
    # Stage 2
    # --------------------------------------------------

    run_stage2()

    # --------------------------------------------------
    # Stage 3
    # --------------------------------------------------
    run_stage3()

    # --------------------------------------------------
    # Stage 4
    # --------------------------------------------------
    run_stage4()

    # --------------------------------------------------
    # Stage 5
    # --------------------------------------------------
    run_stage5()

    # --------------------------------------------------
    # Stage 6
    # --------------------------------------------------
    run_stage6()
    #run_stage6(project_root)

    # --------------------------------------------------
    # Stage 7
    # --------------------------------------------------
    run_stage7()

    # --------------------------------------------------
    # Stage 8
    # --------------------------------------------------
    run_stage8()

    # --------------------------------------------------
    # Stage 9
    # --------------------------------------------------
    run_stage9()

    # --------------------------------------------------
    # Stage 10
    # --------------------------------------------------
    run_stage10()

    logger.info("=" * 70)
    logger.info("Pipeline Finished")
    logger.info("=" * 70)