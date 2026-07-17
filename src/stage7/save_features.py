from pathlib import Path

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("Stage7 Save Features")


def save_features(project_root: Path, validated_features, feature_config):

    logger.info("=" * 70)
    logger.info("Saving Extracted Features")
    logger.info("=" * 70)

    feature_type = feature_config["feature_type"]

    output_folder = Path(feature_config["feature_output_folder"])

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    saved_features = []

    total = len(validated_features)

    saved = 0
    skipped = 0
    failed = 0

    for index, item in enumerate(validated_features, start=1):

        audio_file = item.get("audio_file", "")

        logger.info(f"[{index}/{total}] Saving features: {audio_file}")

        if not item.get("feature_valid", False):

            item["feature_saved"] = False
            item["feature_file"] = ""
            item["save_status"] = "Skipped"
            item["save_message"] = "Feature is not valid"

            saved_features.append(item)

            skipped += 1

            continue

        try:

            features = item.get("features", None)

            if features is None:

                raise ValueError("Feature array is None")

            features = np.asarray(features)

            feature_file_name = Path(audio_file).stem + ".npy"

            feature_file_path = output_folder / feature_file_name

            np.save(
                feature_file_path,
                features
            )

            item["feature_saved"] = True
            item["feature_file"] = str(feature_file_path)
            item["feature_file_name"] = feature_file_name
            item["save_status"] = "Saved"
            item["save_message"] = "Feature file saved successfully"
            item["feature_type"] = feature_type
            item["feature_shape"] = features.shape

            saved_features.append(item)

            saved += 1

        except Exception as e:

            logger.warning(f"Saving failed for {audio_file}: {e}")

            item["feature_saved"] = False
            item["feature_file"] = ""
            item["feature_file_name"] = ""
            item["save_status"] = "Failed"
            item["save_message"] = str(e)

            saved_features.append(item)

            failed += 1

    logger.info(f"Total features : {total}")
    logger.info(f"Saved          : {saved}")
    logger.info(f"Skipped        : {skipped}")
    logger.info(f"Failed         : {failed}")

    logger.info("=" * 70)

    return saved_features