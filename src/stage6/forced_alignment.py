from src.utils.logger import get_logger

logger = get_logger("Stage6 Forced Alignment")


def run_forced_alignment(dataset):

    logger.info("=" * 70)
    logger.info("Running Forced Alignment")
    logger.info("=" * 70)

    aligned = 0
    skipped = 0

    results = []

    for sample in dataset:

        # ------------------------------------------
        # Skip samples not ready
        # ------------------------------------------

        if not sample["alignment_ready"]:

            sample["alignment_status"] = "Skipped"

            sample["alignment_score"] = 0.0

            sample["alignment_result"] = []

            skipped += 1

            results.append(sample)

            continue

        # ------------------------------------------
        # Placeholder Alignment
        # ------------------------------------------

        #words = sample["transcript"].split()
        # ------------------------------------------
        # Safe transcript handling
        # ------------------------------------------

        transcript = str(sample.get("transcript", "")).strip()

        if transcript == "" or transcript.lower() == "nan":
            sample["alignment_status"] = "Skipped"
            sample["alignment_score"] = 0.0
            sample["alignment_result"] = []

            skipped += 1
            results.append(sample)
            continue

        words = transcript.split()

        duration = sample["duration"]

        word_duration = duration / max(len(words), 1)

        alignment = []

        current_time = 0.0

        for word in words:

            start = round(current_time, 3)

            end = round(current_time + word_duration, 3)

            alignment.append({

                "word": word,

                "start": start,

                "end": end

            })

            current_time += word_duration

        sample["alignment_result"] = alignment

        sample["alignment_score"] = 1.0

        sample["alignment_status"] = "Completed"

        aligned += 1

        results.append(sample)

    logger.info(f"Aligned : {aligned}")

    logger.info(f"Skipped : {skipped}")

    logger.info("=" * 70)

    return results