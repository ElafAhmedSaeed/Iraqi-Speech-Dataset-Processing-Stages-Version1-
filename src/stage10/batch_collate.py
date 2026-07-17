import numpy as np

from src.utils.logger import get_logger

logger = get_logger("Stage10 Batch Collate")


def _pad_features(feature_list, pad_value=0.0):

    if len(feature_list) == 0:

        raise ValueError("Feature list is empty")

    feature_dims = [features.shape[1] for features in feature_list]

    if len(set(feature_dims)) != 1:

        raise ValueError(
            f"Feature dimensions are not consistent: {feature_dims}"
        )

    batch_size = len(feature_list)

    max_time = max(features.shape[0] for features in feature_list)

    feature_dim = feature_dims[0]

    padded_features = np.full(
        shape=(batch_size, max_time, feature_dim),
        fill_value=pad_value,
        dtype=np.float32
    )

    input_lengths = np.zeros(
        shape=(batch_size,),
        dtype=np.int64
    )

    for index, features in enumerate(feature_list):

        time_steps = features.shape[0]

        padded_features[index, :time_steps, :] = features

        input_lengths[index] = time_steps

    return padded_features, input_lengths


def _pad_labels(label_list, pad_value=0):

    if len(label_list) == 0:

        raise ValueError("Label list is empty")

    batch_size = len(label_list)

    max_label_length = max(len(labels) for labels in label_list)

    padded_labels = np.full(
        shape=(batch_size, max_label_length),
        fill_value=pad_value,
        dtype=np.int64
    )

    label_lengths = np.zeros(
        shape=(batch_size,),
        dtype=np.int64
    )

    flat_labels = []

    for index, labels in enumerate(label_list):

        labels = np.asarray(
            labels,
            dtype=np.int64
        )

        label_length = len(labels)

        padded_labels[index, :label_length] = labels

        label_lengths[index] = label_length

        flat_labels.extend(labels.tolist())

    flat_labels = np.asarray(
        flat_labels,
        dtype=np.int64
    )

    return padded_labels, label_lengths, flat_labels


def asr_collate_fn(
    batch,
    feature_pad_value=0.0,
    label_pad_value=0
):

    if batch is None or len(batch) == 0:

        raise ValueError("Batch is empty")

    valid_batch = []

    for sample in batch:

        if sample is None:

            continue

        if "features" not in sample or "labels" not in sample:

            continue

        valid_batch.append(sample)

    if len(valid_batch) == 0:

        raise ValueError("No valid samples found in batch")

    feature_list = []

    label_list = []

    audio_files = []

    feature_files = []

    label_texts = []

    pair_ids = []

    dataset_splits = []

    for sample in valid_batch:

        features = np.asarray(
            sample["features"],
            dtype=np.float32
        )

        labels = np.asarray(
            sample["labels"],
            dtype=np.int64
        )

        if features.ndim != 2:

            raise ValueError(
                f"Invalid feature shape for {sample.get('audio_file', '')}: "
                f"{features.shape}"
            )

        if labels.ndim != 1:

            raise ValueError(
                f"Invalid label shape for {sample.get('audio_file', '')}: "
                f"{labels.shape}"
            )

        feature_list.append(features)

        label_list.append(labels)

        audio_files.append(
            sample.get("audio_file", "")
        )

        feature_files.append(
            sample.get("feature_file", "")
        )

        label_texts.append(
            sample.get("label_text", "")
        )

        pair_ids.append(
            sample.get("pair_id", "")
        )

        dataset_splits.append(
            sample.get("dataset_split", "")
        )

    padded_features, input_lengths = _pad_features(
        feature_list,
        pad_value=feature_pad_value
    )

    padded_labels, label_lengths, flat_labels = _pad_labels(
        label_list,
        pad_value=label_pad_value
    )

    batch_data = {

        "features": padded_features,

        "labels": padded_labels,

        "flat_labels": flat_labels,

        "input_lengths": input_lengths,

        "label_lengths": label_lengths,

        "batch_size": len(valid_batch),

        "max_input_length": int(padded_features.shape[1]),

        "feature_dim": int(padded_features.shape[2]),

        "max_label_length": int(padded_labels.shape[1]),

        "audio_files": audio_files,

        "feature_files": feature_files,

        "label_texts": label_texts,

        "pair_ids": pair_ids,

        "dataset_splits": dataset_splits,

        "collate_status": "Created",

        "collate_message": "Batch collated successfully"

    }

    return batch_data


def create_sample_batch(dataset, batch_size=4):

    if dataset is None:

        raise ValueError("Dataset is None")

    if len(dataset) == 0:

        raise ValueError("Dataset is empty")

    actual_batch_size = min(
        batch_size,
        len(dataset)
    )

    batch = []

    for index in range(actual_batch_size):

        batch.append(
            dataset[index]
        )

    return batch


def test_collate_function(
    train_dataset,
    val_dataset,
    test_dataset,
    blank_id=0,
    batch_size=4
):

    logger.info("=" * 70)
    logger.info("Testing Batch Collate Function")
    logger.info("=" * 70)

    collate_results = {}

    summaries = {}

    datasets = {

        "train": train_dataset,

        "val": val_dataset,

        "test": test_dataset

    }

    total_tested = 0
    total_failed = 0

    for split_name, dataset in datasets.items():

        logger.info("-" * 70)
        logger.info(f"Testing collate function for split: {split_name}")
        logger.info("-" * 70)

        try:

            sample_batch = create_sample_batch(
                dataset,
                batch_size=batch_size
            )

            collated_batch = asr_collate_fn(
                sample_batch,
                feature_pad_value=0.0,
                label_pad_value=blank_id
            )

            collate_results[split_name] = collated_batch

            summaries[split_name] = {

                "split": split_name,

                "status": "Passed",

                "batch_size": collated_batch["batch_size"],

                "feature_shape": collated_batch["features"].shape,

                "label_shape": collated_batch["labels"].shape,

                "input_lengths_shape": collated_batch["input_lengths"].shape,

                "label_lengths_shape": collated_batch["label_lengths"].shape,

                "message": "Collate function test passed"

            }

            logger.info(f"{split_name} batch size       : {collated_batch['batch_size']}")
            logger.info(f"{split_name} feature shape    : {collated_batch['features'].shape}")
            logger.info(f"{split_name} label shape      : {collated_batch['labels'].shape}")
            logger.info(f"{split_name} input lengths    : {collated_batch['input_lengths']}")
            logger.info(f"{split_name} label lengths    : {collated_batch['label_lengths']}")

            total_tested += 1

        except Exception as e:

            collate_results[split_name] = None

            summaries[split_name] = {

                "split": split_name,

                "status": "Failed",

                "batch_size": 0,

                "feature_shape": "",

                "label_shape": "",

                "input_lengths_shape": "",

                "label_lengths_shape": "",

                "message": str(e)

            }

            logger.warning(
                f"Collate function failed for {split_name}: {e}"
            )

            total_failed += 1

    overall_status = (
        "Passed"
        if total_failed == 0
        else "Failed"
    )

    collate_summary = {

        "status": overall_status,

        "message": (
            "Batch collate function is working correctly"
            if overall_status == "Passed"
            else "Batch collate function has errors"
        ),

        "tested_splits": total_tested,

        "failed_splits": total_failed,

        "batch_size": batch_size,

        "train_summary": summaries.get("train", {}),

        "val_summary": summaries.get("val", {}),

        "test_summary": summaries.get("test", {})

    }

    logger.info("=" * 70)
    logger.info(f"Collate test status : {collate_summary['status']}")
    logger.info(f"Tested splits       : {collate_summary['tested_splits']}")
    logger.info(f"Failed splits       : {collate_summary['failed_splits']}")
    logger.info("=" * 70)

    return collate_results, collate_summary