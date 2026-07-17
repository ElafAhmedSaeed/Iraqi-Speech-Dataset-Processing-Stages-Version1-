from src.utils.logger import get_logger

logger = get_logger("Stage5 Dataset Detector")


def detect_dataset_type(analyzed_dataset):

    logger.info("=" * 70)
    logger.info("Detecting Dataset Type")
    logger.info("=" * 70)

    results = []

    segmented_count = 0
    long_count = 0

    for audio in analyzed_dataset:

        if audio["is_long_audio"]:

            dataset_type = "long_recording"
            long_count += 1

        else:

            dataset_type = "already_segmented"
            segmented_count += 1

        audio["dataset_type"] = dataset_type

        results.append(audio)

    logger.info(f"Already Segmented : {segmented_count}")
    logger.info(f"Long Recordings   : {long_count}")

    logger.info("=" * 70)

    return results