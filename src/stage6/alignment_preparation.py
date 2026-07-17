from src.utils.logger import get_logger

logger = get_logger("Stage6 Alignment Preparation")


def prepare_alignment(dataset):

    logger.info("=" * 70)
    logger.info("Preparing Alignment Dataset")
    logger.info("=" * 70)

    prepared = 0
    skipped = 0

    results = []

    for sample in dataset:

        ready = True
        reason = ""

        # ------------------------------------------
        # Check transcript
        # ------------------------------------------

        #transcript = str(sample["transcript"]).strip()
        transcript = sample.get("transcript", "")

        if transcript is None:
            transcript = ""

        transcript = str(transcript).strip()

        if transcript.lower() == "nan":
            transcript = ""

        if transcript == "":

            ready = False
            reason = "Empty Transcript"

        # ------------------------------------------
        # Check sample rate
        # ------------------------------------------

        elif sample["sample_rate"] <= 0:

            ready = False
            reason = "Invalid Sample Rate"

        # ------------------------------------------
        # Check duration
        # ------------------------------------------

        elif sample["duration"] <= 0:

            ready = False
            reason = "Invalid Duration"

        # ------------------------------------------
        # Save result
        # ------------------------------------------

        sample["alignment_ready"] = ready

        if ready:

            sample["preparation_status"] = "Ready"
            prepared += 1

        else:

            sample["preparation_status"] = reason
            skipped += 1

        results.append(sample)

    logger.info(f"Ready   : {prepared}")
    logger.info(f"Skipped : {skipped}")

    logger.info("=" * 70)

    return results