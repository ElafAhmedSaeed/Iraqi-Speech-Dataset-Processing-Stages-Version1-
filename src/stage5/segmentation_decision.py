from src.utils.logger import get_logger

logger = get_logger("Stage5 Segmentation Decision")


def segmentation_decision(dataset):

    logger.info("=" * 70)
    logger.info("Segmentation Decision")
    logger.info("=" * 70)

    skip_count = 0
    segmentation_count = 0

    results = []

    for audio in dataset:

        if audio["dataset_type"] == "already_segmented":

            audio["segmentation_required"] = False
            audio["decision"] = "skip"

            skip_count += 1

        elif audio["dataset_type"] == "long_recording":

            audio["segmentation_required"] = True
            audio["decision"] = "run_segmentation"

            segmentation_count += 1

        else:

            audio["segmentation_required"] = False
            audio["decision"] = "unknown"

        results.append(audio)

    logger.info(f"Skip Segmentation : {skip_count}")
    logger.info(f"Run Segmentation  : {segmentation_count}")

    logger.info("=" * 70)

    return results