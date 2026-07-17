from src.utils.logger import get_logger

logger = get_logger("Stage5 Audio Segmentation")


def run_audio_segmentation(dataset):

    logger.info("=" * 70)
    logger.info("Running Audio Segmentation")
    logger.info("=" * 70)

    results = []

    skipped = 0
    completed = 0

    for audio in dataset:

        # --------------------------------------------
        # Already segmented
        # --------------------------------------------

        if audio["decision"] == "skip":

            audio["segments"] = 1
            audio["segmentation_status"] = "Skipped"

            skipped += 1

        # --------------------------------------------
        # Long recording
        # --------------------------------------------

        elif audio["decision"] == "run_segmentation":

            # سيتم استبدال هذا لاحقاً بمحرك VAD الحقيقي

            estimated_segments = max(
                1,
                int(audio["duration"] / 10)
            )

            audio["segments"] = estimated_segments
            audio["segmentation_status"] = "Completed"

            completed += 1

        else:

            audio["segments"] = 0
            audio["segmentation_status"] = "Unknown"

        results.append(audio)

    logger.info(f"Skipped     : {skipped}")
    logger.info(f"Segmented   : {completed}")

    logger.info("=" * 70)

    return results