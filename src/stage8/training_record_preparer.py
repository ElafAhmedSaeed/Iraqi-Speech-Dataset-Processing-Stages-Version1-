from src.utils.logger import get_logger

logger = get_logger("Stage8 Training Records")


def _safe_text(value):

    if value is None:
        return ""

    value = str(value)

    if value.lower() == "nan":
        return ""

    return value.strip()


def _count_words(text):

    text = _safe_text(text)

    if text == "":
        return 0

    return len(text.split())


def prepare_training_records(feature_dataset):

    logger.info("=" * 70)
    logger.info("Preparing Training Records")
    logger.info("=" * 70)

    training_records = []

    total = len(feature_dataset)

    prepared = 0
    skipped = 0

    for index, sample in enumerate(feature_dataset, start=1):

        pair_id = sample.get("pair_id", "")

        audio_file = _safe_text(sample.get("audio_file", ""))

        feature_file = _safe_text(sample.get("feature_file", ""))

        feature_type = _safe_text(sample.get("feature_type", ""))

        feature_shape = _safe_text(sample.get("feature_shape", ""))

        transcript = _safe_text(sample.get("transcript", ""))

        normalized_text = _safe_text(sample.get("normalized_text", ""))

        training_text = _safe_text(sample.get("training_text", ""))

        duration = sample.get("duration", "")

        sample_rate = sample.get("sample_rate", "")

        logger.info(f"[{index}/{total}] Preparing record: {audio_file}")

        # --------------------------------------------------
        # Basic validation
        # --------------------------------------------------

        if audio_file == "":

            logger.warning("Skipped record: missing audio_file")

            skipped += 1

            continue

        if feature_file == "":

            logger.warning(f"Skipped record {audio_file}: missing feature_file")

            skipped += 1

            continue

        if training_text == "":

            logger.warning(f"Skipped record {audio_file}: missing training_text")

            skipped += 1

            continue

        # --------------------------------------------------
        # Text statistics
        # --------------------------------------------------

        text_length = len(training_text)

        word_count = _count_words(training_text)

        if word_count == 0:

            logger.warning(f"Skipped record {audio_file}: zero words")

            skipped += 1

            continue

        # --------------------------------------------------
        # Create training record
        # --------------------------------------------------

        record = {

            "pair_id": pair_id,

            "audio_file": audio_file,

            "feature_file": feature_file,

            "feature_type": feature_type,

            "feature_shape": feature_shape,

            "transcript": transcript,

            "normalized_text": normalized_text,

            "training_text": training_text,

            "text_length": text_length,

            "word_count": word_count,

            "duration": duration,

            "sample_rate": sample_rate,

            "record_ready": True,

            "record_status": "Prepared",

            "record_message": "Training record prepared successfully"

        }

        training_records.append(record)

        prepared += 1

    logger.info(f"Total feature samples : {total}")
    logger.info(f"Prepared records      : {prepared}")
    logger.info(f"Skipped records       : {skipped}")

    logger.info("=" * 70)

    return training_records