import numpy as np

from src.utils.logger import get_logger

logger = get_logger("Stage7 Feature Validator")


def validate_single_feature(feature_item):

    audio_file = feature_item.get("audio_file", "")

    features = feature_item.get("features", None)

    feature_status = feature_item.get("feature_status", "")

    # --------------------------------------------------
    # Check extraction status
    # --------------------------------------------------

    if feature_status != "Extracted":

        return {

            **feature_item,

            "feature_valid": False,

            "validation_status": "Failed",

            "validation_message": "Feature extraction was not successful"

        }

    # --------------------------------------------------
    # Check feature object
    # --------------------------------------------------

    if features is None:

        return {

            **feature_item,

            "feature_valid": False,

            "validation_status": "Failed",

            "validation_message": "Feature array is None"

        }

    # --------------------------------------------------
    # Convert to numpy array
    # --------------------------------------------------

    try:

        features = np.asarray(features)

    except Exception as e:

        return {

            **feature_item,

            "feature_valid": False,

            "validation_status": "Failed",

            "validation_message": f"Cannot convert features to numpy array: {e}"

        }

    # --------------------------------------------------
    # Check dimensions
    # --------------------------------------------------

    if features.ndim != 2:

        return {

            **feature_item,

            "feature_valid": False,

            "validation_status": "Failed",

            "validation_message": f"Invalid feature dimensions: {features.ndim}"

        }

    # --------------------------------------------------
    # Check empty array
    # --------------------------------------------------

    if features.size == 0:

        return {

            **feature_item,

            "feature_valid": False,

            "validation_status": "Failed",

            "validation_message": "Feature array is empty"

        }

    # --------------------------------------------------
    # Check number of frames
    # --------------------------------------------------

    if features.shape[1] == 0:

        return {

            **feature_item,

            "feature_valid": False,

            "validation_status": "Failed",

            "validation_message": "Feature has zero frames"

        }

    # --------------------------------------------------
    # Check NaN values
    # --------------------------------------------------

    if np.isnan(features).any():

        return {

            **feature_item,

            "feature_valid": False,

            "validation_status": "Failed",

            "validation_message": "Feature contains NaN values"

        }

    # --------------------------------------------------
    # Check infinite values
    # --------------------------------------------------

    if np.isinf(features).any():

        return {

            **feature_item,

            "feature_valid": False,

            "validation_status": "Failed",

            "validation_message": "Feature contains infinite values"

        }

    # --------------------------------------------------
    # Passed
    # --------------------------------------------------

    return {

        **feature_item,

        "features": features,

        "feature_shape": features.shape,

        "feature_valid": True,

        "validation_status": "Passed",

        "validation_message": "Feature validation passed"

    }


def validate_features(extracted_features):

    logger.info("=" * 70)
    logger.info("Validating Extracted Features")
    logger.info("=" * 70)

    validated_features = []

    total = len(extracted_features)

    passed = 0
    failed = 0

    for index, feature_item in enumerate(extracted_features, start=1):

        audio_file = feature_item.get("audio_file", "")

        logger.info(f"[{index}/{total}] Validating features: {audio_file}")

        result = validate_single_feature(feature_item)

        validated_features.append(result)

        if result["feature_valid"]:

            passed += 1

        else:

            failed += 1

            logger.warning(
                f"Feature validation failed for {audio_file}: "
                f"{result['validation_message']}"
            )

    logger.info(f"Total features   : {total}")
    logger.info(f"Validation passed: {passed}")
    logger.info(f"Validation failed: {failed}")

    logger.info("=" * 70)

    return validated_features