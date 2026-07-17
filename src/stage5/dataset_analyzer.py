from src.utils.logger import get_logger

logger = get_logger("Stage5 Dataset Analyzer")


LONG_AUDIO_THRESHOLD = 20.0      # seconds


def analyze_dataset(audio_dataset):

    logger.info("=" * 70)
    logger.info("Analyzing Audio Dataset")
    logger.info("=" * 70)

    analyzed_dataset = []

    short_count = 0
    long_count = 0

    for audio in audio_dataset:

        duration = audio["duration"]

        is_long = duration > LONG_AUDIO_THRESHOLD

        if is_long:

            long_count += 1

            action = "Segmentation Required"

        else:

            short_count += 1

            action = "Skip Segmentation"

        analyzed_audio = {

            **audio,

            "is_short_audio": not is_long,

            "is_long_audio": is_long,

            "recommended_action": action

        }

        analyzed_dataset.append(analyzed_audio)

    logger.info(f"Total Files : {len(analyzed_dataset)}")
    logger.info(f"Short Files : {short_count}")
    logger.info(f"Long Files  : {long_count}")

    logger.info("=" * 70)

    return analyzed_dataset