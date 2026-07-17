# from config.settings import config
#
# print("=" * 40)
#
# print(config["project"]["name"])
#
# print(config["audio"]["sample_rate"])
#
# print(config["paths"]["raw_audio"])
#
# print("=" * 40)
#
# from src.utils.logger import get_logger
#
# logger = get_logger("Main")
#
# logger.info("Project Started")
#
# logger.info("Loading Configuration")
#
# logger.warning("This is a warning message")
#
# logger.error("This is a test error")
#
# logger.info("Project Finished")

from src.pipeline.runner import run_pipeline

if __name__ == "__main__":

    run_pipeline()