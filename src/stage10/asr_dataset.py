from pathlib import Path

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("Stage10 ASR Dataset")


def _safe_text(value):

    if value is None:
        return ""

    value = str(value)

    if value.lower() == "nan":
        return ""

    return value.strip()


def _resolve_path(project_root: Path, file_path):

    file_path = _safe_text(file_path)

    if file_path == "":
        return None

    path = Path(file_path)

    if not path.is_absolute():

        path = project_root / path

    return path


def encode_text(text, char_to_id, unknown_id=1):

    text = _safe_text(text)

    label_ids = []

    for char in text:

        label_ids.append(
            char_to_id.get(char, unknown_id)
        )

    return label_ids


class ASRDataset:

    def __init__(
        self,
        dataframe,
        project_root: Path,
        char_to_id,
        unknown_id=1,
        split_name="train",
        transpose_features=True
    ):

        self.dataframe = dataframe
        self.project_root = project_root
        self.char_to_id = char_to_id
        self.unknown_id = unknown_id
        self.split_name = split_name
        self.transpose_features = transpose_features

        self.records = []

        self.summary = {

            "split": split_name,

            "total_rows": 0,

            "valid_records": 0,

            "skipped_records": 0,

            "status": "Not Built",

            "message": ""

        }

        self._build_index()

    def _build_index(self):

        logger.info("-" * 70)
        logger.info(f"Building ASR Dataset index for split: {self.split_name}")
        logger.info("-" * 70)

        if self.dataframe is None:

            self.summary["message"] = "Dataframe is None"
            self.summary["status"] = "Failed"

            return

        self.summary["total_rows"] = len(self.dataframe)

        required_columns = [

            "pair_id",

            "audio_file",

            "feature_file",

            "training_text",

            "dataset_split"

        ]

        for col in required_columns:

            if col not in self.dataframe.columns:

                self.summary["message"] = f"Missing required column: {col}"
                self.summary["status"] = "Failed"

                logger.warning(self.summary["message"])

                return

        valid = 0
        skipped = 0

        for _, row in self.dataframe.iterrows():

            pair_id = row.get("pair_id", "")

            audio_file = _safe_text(
                row.get("audio_file", "")
            )

            feature_file = _safe_text(
                row.get("feature_file", "")
            )

            training_text = _safe_text(
                row.get("training_text", "")
            )

            dataset_split = _safe_text(
                row.get("dataset_split", "")
            )

            if dataset_split != self.split_name:

                skipped += 1

                continue

            if audio_file == "":

                skipped += 1

                continue

            if feature_file == "":

                skipped += 1

                continue

            if training_text == "":

                skipped += 1

                continue

            feature_path = _resolve_path(
                self.project_root,
                feature_file
            )

            if feature_path is None or not feature_path.exists():

                logger.warning(f"Feature file not found: {feature_file}")

                skipped += 1

                continue

            label_ids = encode_text(
                training_text,
                self.char_to_id,
                self.unknown_id
            )

            if len(label_ids) == 0:

                skipped += 1

                continue

            self.records.append({

                "pair_id": pair_id,

                "audio_file": audio_file,

                "feature_file": feature_file,

                "feature_path": str(feature_path),

                "training_text": training_text,

                "label_ids": label_ids,

                "label_length": len(label_ids),

                "dataset_split": dataset_split,

                "feature_type": row.get("feature_type", ""),

                "duration": row.get("duration", ""),

                "sample_rate": row.get("sample_rate", "")

            })

            valid += 1

        self.summary["valid_records"] = valid
        self.summary["skipped_records"] = skipped

        if valid > 0:

            self.summary["status"] = "Built"
            self.summary["message"] = "ASR dataset index built successfully"

        else:

            self.summary["status"] = "Failed"
            self.summary["message"] = "No valid records found"

        logger.info(f"{self.split_name} total rows      : {self.summary['total_rows']}")
        logger.info(f"{self.split_name} valid records  : {self.summary['valid_records']}")
        logger.info(f"{self.split_name} skipped records: {self.summary['skipped_records']}")
        logger.info(f"{self.split_name} status         : {self.summary['status']}")

    def __len__(self):

        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        feature_path = Path(record["feature_path"])

        features = np.load(
            feature_path,
            allow_pickle=False
        )

        features = np.asarray(
            features,
            dtype=np.float32
        )

        if features.ndim != 2:

            raise ValueError(
                f"Invalid feature shape for {record['audio_file']}: {features.shape}"
            )

        # Original Stage 7 shape is usually:
        # (feature_dim, time_frames)
        #
        # For training, it is better to return:
        # (time_frames, feature_dim)
        if self.transpose_features:

            features = features.T

        labels = np.asarray(
            record["label_ids"],
            dtype=np.int64
        )

        sample = {

            "features": features,

            "labels": labels,

            "input_length": int(features.shape[0]),

            "feature_dim": int(features.shape[1]),

            "label_length": int(len(labels)),

            "label_text": record["training_text"],

            "audio_file": record["audio_file"],

            "feature_file": record["feature_file"],

            "dataset_split": record["dataset_split"],

            "pair_id": record["pair_id"]

        }

        return sample


def build_asr_datasets(project_root: Path, training_resources):

    logger.info("=" * 70)
    logger.info("Building ASR Datasets")
    logger.info("=" * 70)

    char_to_id = training_resources.get("char_to_id", {})

    unknown_id = training_resources.get("unknown_id", 1)

    train_df = training_resources.get("train_df")

    val_df = training_resources.get("val_df")

    test_df = training_resources.get("test_df")

    train_dataset = ASRDataset(
        dataframe=train_df,
        project_root=project_root,
        char_to_id=char_to_id,
        unknown_id=unknown_id,
        split_name="train",
        transpose_features=True
    )

    val_dataset = ASRDataset(
        dataframe=val_df,
        project_root=project_root,
        char_to_id=char_to_id,
        unknown_id=unknown_id,
        split_name="val",
        transpose_features=True
    )

    test_dataset = ASRDataset(
        dataframe=test_df,
        project_root=project_root,
        char_to_id=char_to_id,
        unknown_id=unknown_id,
        split_name="test",
        transpose_features=True
    )

    total_records = (
        len(train_dataset)
        + len(val_dataset)
        + len(test_dataset)
    )

    dataset_summary = {

        "status": "Built" if total_records > 0 else "Failed",

        "message": (
            "ASR datasets built successfully"
            if total_records > 0
            else "ASR datasets building failed"
        ),

        "train_records": len(train_dataset),

        "val_records": len(val_dataset),

        "test_records": len(test_dataset),

        "total_records": total_records,

        "train_summary": train_dataset.summary,

        "val_summary": val_dataset.summary,

        "test_summary": test_dataset.summary

    }

    logger.info(f"Train dataset records      : {dataset_summary['train_records']}")
    logger.info(f"Validation dataset records : {dataset_summary['val_records']}")
    logger.info(f"Test dataset records       : {dataset_summary['test_records']}")
    logger.info(f"Total dataset records      : {dataset_summary['total_records']}")
    logger.info(f"Dataset status             : {dataset_summary['status']}")

    logger.info("=" * 70)

    return train_dataset, val_dataset, test_dataset, dataset_summary