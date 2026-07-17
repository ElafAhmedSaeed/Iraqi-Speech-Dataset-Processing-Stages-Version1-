from pathlib import Path
import json

from src.utils.logger import get_logger

logger = get_logger("Stage9 Text Label Preparer")


def _safe_text(value):

    if value is None:
        return ""

    value = str(value)

    if value.lower() == "nan":
        return ""

    return value.strip()


def _clean_label_text(text):

    text = _safe_text(text)

    text = " ".join(text.split())

    return text


def _collect_texts(records):

    texts = []

    for record in records:

        text = _clean_label_text(
            record.get("training_text", "")
        )

        if text != "":

            texts.append(text)

    return texts


def build_character_vocabulary(all_records):

    logger.info("Building character-level vocabulary")

    texts = _collect_texts(all_records)

    characters = set()

    for text in texts:

        for char in text:

            characters.add(char)

    sorted_chars = sorted(list(characters))

    char_to_id = {

        "<blank>": 0,

        "<unk>": 1

    }

    for char in sorted_chars:

        if char not in char_to_id:

            char_to_id[char] = len(char_to_id)

    id_to_char = {

        str(index): char

        for char, index in char_to_id.items()

    }

    vocab = {

        "label_type": "character",

        "blank_token": "<blank>",

        "unknown_token": "<unk>",

        "blank_id": 0,

        "unknown_id": 1,

        "vocab_size": len(char_to_id),

        "char_to_id": char_to_id,

        "id_to_char": id_to_char

    }

    logger.info(f"Vocabulary size: {vocab['vocab_size']}")

    return vocab


def encode_text_to_ids(text, char_to_id):

    text = _clean_label_text(text)

    unknown_id = char_to_id.get("<unk>", 1)

    label_ids = []

    for char in text:

        label_ids.append(
            char_to_id.get(char, unknown_id)
        )

    return label_ids


def _prepare_split_labels(records, split_name, vocab):

    logger.info("-" * 70)
    logger.info(f"Preparing text labels for split: {split_name}")
    logger.info("-" * 70)

    prepared_records = []

    total = len(records)

    prepared = 0
    skipped = 0

    char_to_id = vocab["char_to_id"]

    for index, record in enumerate(records, start=1):

        audio_file = _safe_text(
            record.get("audio_file", "")
        )

        text = _clean_label_text(
            record.get("training_text", "")
        )

        logger.info(
            f"[{index}/{total}] Preparing label: {audio_file}"
        )

        if text == "":

            record["label_ready"] = False
            record["label_status"] = "Skipped"
            record["label_message"] = "Empty training text"
            record["label_text"] = ""
            record["label_ids"] = []
            record["label_length"] = 0
            record["label_type"] = "character"

            prepared_records.append(record)

            skipped += 1

            continue

        label_ids = encode_text_to_ids(
            text,
            char_to_id
        )

        if len(label_ids) == 0:

            record["label_ready"] = False
            record["label_status"] = "Skipped"
            record["label_message"] = "Empty encoded label"
            record["label_text"] = text
            record["label_ids"] = []
            record["label_length"] = 0
            record["label_type"] = "character"

            prepared_records.append(record)

            skipped += 1

            continue

        record["label_text"] = text
        record["label_ids"] = label_ids
        record["label_length"] = len(label_ids)
        record["label_type"] = "character"
        record["label_ready"] = True
        record["label_status"] = "Prepared"
        record["label_message"] = "Text label prepared successfully"

        prepared_records.append(record)

        prepared += 1

    split_summary = {

        "split": split_name,

        "total_records": total,

        "prepared_labels": prepared,

        "skipped_labels": skipped,

        "status": (
            "Prepared"
            if prepared > 0 and skipped == 0
            else "Prepared with warnings"
        )

    }

    logger.info(f"{split_name} total records   : {total}")
    logger.info(f"{split_name} prepared labels : {prepared}")
    logger.info(f"{split_name} skipped labels  : {skipped}")

    return prepared_records, split_summary


def save_vocabulary(project_root: Path, vocab):

    output_folder = project_root / "data" / "training_interface"

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    vocab_file = output_folder / "character_vocabulary.json"

    with open(vocab_file, "w", encoding="utf-8") as f:

        json.dump(
            vocab,
            f,
            ensure_ascii=False,
            indent=4
        )

    logger.info(f"Vocabulary saved to: {vocab_file}")

    return str(vocab_file)


def prepare_text_labels(
    project_root: Path,
    train_features,
    val_features,
    test_features
):

    logger.info("=" * 70)
    logger.info("Preparing Text Labels")
    logger.info("=" * 70)

    all_records = []

    all_records.extend(train_features)
    all_records.extend(val_features)
    all_records.extend(test_features)

    if len(all_records) == 0:

        label_summary = {

            "status": "Failed",

            "message": "No records available for text label preparation",

            "total_records": 0,

            "prepared_labels": 0,

            "skipped_labels": 0,

            "vocab_file": "",

            "vocab_size": 0,

            "label_type": "character"

        }

        return [], [], [], None, label_summary

    # --------------------------------------------------
    # Build vocabulary
    # --------------------------------------------------

    vocab = build_character_vocabulary(
        all_records
    )

    vocab_file = save_vocabulary(
        project_root,
        vocab
    )

    # --------------------------------------------------
    # Prepare labels for each split
    # --------------------------------------------------

    train_labeled, train_summary = _prepare_split_labels(
        train_features,
        "train",
        vocab
    )

    val_labeled, val_summary = _prepare_split_labels(
        val_features,
        "val",
        vocab
    )

    test_labeled, test_summary = _prepare_split_labels(
        test_features,
        "test",
        vocab
    )

    total_prepared = (
        train_summary["prepared_labels"]
        + val_summary["prepared_labels"]
        + test_summary["prepared_labels"]
    )

    total_skipped = (
        train_summary["skipped_labels"]
        + val_summary["skipped_labels"]
        + test_summary["skipped_labels"]
    )

    label_summary = {

        "status": (
            "Prepared"
            if total_prepared > 0 and total_skipped == 0
            else "Prepared with warnings"
        ),

        "message": (
            "All text labels prepared successfully"
            if total_skipped == 0
            else "Some text labels were skipped"
        ),

        "total_records": len(all_records),

        "prepared_labels": total_prepared,

        "skipped_labels": total_skipped,

        "vocab_file": vocab_file,

        "vocab_size": vocab["vocab_size"],

        "label_type": "character",

        "train_summary": train_summary,

        "val_summary": val_summary,

        "test_summary": test_summary

    }

    logger.info("=" * 70)
    logger.info(f"Total records   : {label_summary['total_records']}")
    logger.info(f"Prepared labels : {label_summary['prepared_labels']}")
    logger.info(f"Skipped labels  : {label_summary['skipped_labels']}")
    logger.info(f"Vocabulary size : {label_summary['vocab_size']}")
    logger.info(f"Label status    : {label_summary['status']}")
    logger.info("=" * 70)

    return train_labeled, val_labeled, test_labeled, vocab, label_summary