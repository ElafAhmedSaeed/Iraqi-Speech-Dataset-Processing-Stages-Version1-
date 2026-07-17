from src.utils.logger import get_logger

logger = get_logger("Stage5 Segment Validator")


def validate_segments(dataset):

    logger.info("=" * 70)
    logger.info("Validating Segmentation Results")
    logger.info("=" * 70)

    passed = 0
    failed = 0

    results = []

    for audio in dataset:

        if audio["segments"] >= 1:

            audio["segment_validation"] = "Passed"

            passed += 1

        else:

            audio["segment_validation"] = "Failed"

            failed += 1

        results.append(audio)

    logger.info(f"Passed : {passed}")
    logger.info(f"Failed : {failed}")

    logger.info("=" * 70)

    return results