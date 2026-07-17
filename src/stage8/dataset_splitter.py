import random

from src.utils.logger import get_logger

logger = get_logger("Stage8 Dataset Splitter")


def split_dataset(
    training_records,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    random_state=42
):

    logger.info("=" * 70)
    logger.info("Splitting Dataset")
    logger.info("=" * 70)

    total_ratio = train_ratio + val_ratio + test_ratio

    if round(total_ratio, 2) != 1.00:

        logger.warning(
            f"Invalid split ratios. Sum must be 1.0, got {total_ratio}"
        )

        return [], [], []

    records = list(training_records)

    random.seed(random_state)

    random.shuffle(records)

    total = len(records)

    train_count = int(total * train_ratio)

    val_count = int(total * val_ratio)

    test_count = total - train_count - val_count

    train_records = records[:train_count]

    val_records = records[train_count:train_count + val_count]

    test_records = records[train_count + val_count:]

    for record in train_records:

        record["dataset_split"] = "train"

        record["included_in_training"] = True

        record["split_status"] = "Assigned"

    for record in val_records:

        record["dataset_split"] = "val"

        record["included_in_training"] = True

        record["split_status"] = "Assigned"

    for record in test_records:

        record["dataset_split"] = "test"

        record["included_in_training"] = True

        record["split_status"] = "Assigned"

    logger.info(f"Total records      : {total}")
    logger.info(f"Train records      : {len(train_records)}")
    logger.info(f"Validation records : {len(val_records)}")
    logger.info(f"Test records       : {len(test_records)}")
    logger.info(f"Random state       : {random_state}")

    logger.info("=" * 70)

    return train_records, val_records, test_records