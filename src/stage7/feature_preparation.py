from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger("Stage7 Feature Preparation")


def prepare_feature_extraction(project_root: Path, feature_type: str):

    logger.info("=" * 70)
    logger.info("Preparing Feature Extraction Configuration")
    logger.info("=" * 70)

    # --------------------------------------------------
    # Supported feature types
    # --------------------------------------------------

    supported_features = [

        "mfcc",

        "mel",

        "combined"

    ]

    feature_type = feature_type.lower().strip()

    if feature_type not in supported_features:

        logger.warning(f"Unsupported feature type: {feature_type}")
        logger.warning(f"Supported feature types: {supported_features}")

        return None

    # --------------------------------------------------
    # Output folders
    # --------------------------------------------------

    features_root = project_root / "data" / "features"

    feature_output_folder = features_root / feature_type

    feature_output_folder.mkdir(

        parents=True,

        exist_ok=True

    )

    # --------------------------------------------------
    # Common audio settings
    # --------------------------------------------------

    config = {

        "feature_type": feature_type,

        "features_root": str(features_root),

        "feature_output_folder": str(feature_output_folder),

        "sample_rate": 16000,

        "mono": True,

        "file_format": "npy"

    }

    # --------------------------------------------------
    # MFCC settings
    # --------------------------------------------------

    if feature_type == "mfcc":

        config.update({

            "n_mfcc": 13,

            "n_fft": 2048,

            "hop_length": 512,

            "win_length": 2048,

            "include_delta": True,

            "include_delta_delta": True

        })

    # --------------------------------------------------
    # Mel Spectrogram settings
    # --------------------------------------------------

    elif feature_type == "mel":

        config.update({

            "n_mels": 128,

            "n_fft": 2048,

            "hop_length": 512,

            "win_length": 2048,

            "power": 2.0,

            "log_scale": True

        })

    # --------------------------------------------------
    # Combined settings
    # --------------------------------------------------

    elif feature_type == "combined":

        config.update({

            "n_mfcc": 13,

            "n_mels": 128,

            "n_fft": 2048,

            "hop_length": 512,

            "win_length": 2048,

            "include_delta": True,

            "include_delta_delta": True,

            "include_rms": True,

            "include_zcr": True,

            "include_spectral_centroid": True

        })

    logger.info(f"Feature type          : {config['feature_type']}")
    logger.info(f"Sample rate           : {config['sample_rate']}")
    logger.info(f"Output folder         : {config['feature_output_folder']}")
    logger.info(f"Feature file format   : {config['file_format']}")

    logger.info("=" * 70)

    return config