from src.utils.logger import get_logger

logger = get_logger("Stage6 Alignment Validation")


def validate_alignment(dataset):

    logger.info("=" * 70)
    logger.info("Validating Alignment Results")
    logger.info("=" * 70)

    passed = 0
    failed = 0

    results = []

    for sample in dataset:

        valid = True

        # --------------------------------------------------
        # Alignment executed
        # --------------------------------------------------

        if sample["alignment_status"] != "Completed":

            valid = False

        # --------------------------------------------------
        # Alignment list
        # --------------------------------------------------

        elif len(sample["alignment_result"]) == 0:

            valid = False

        # --------------------------------------------------
        # Timing validation
        # --------------------------------------------------

        else:

            for word in sample["alignment_result"]:

                if word["start"] >= word["end"]:

                    valid = False
                    break

        # --------------------------------------------------
        # Save validation
        # --------------------------------------------------

        if valid:

            sample["alignment_validation"] = "Passed"

            passed += 1

        else:

            sample["alignment_validation"] = "Failed"

            failed += 1

        results.append(sample)

    logger.info(f"Passed : {passed}")
    logger.info(f"Failed : {failed}")

    logger.info("=" * 70)

    return results