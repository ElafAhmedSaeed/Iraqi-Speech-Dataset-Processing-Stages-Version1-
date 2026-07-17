from pathlib import Path

import numpy as np
import librosa

from src.utils.logger import get_logger

logger = get_logger("Stage7 Feature Extractor")


def extract_mfcc(y, sr, config):

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=config.get("n_mfcc", 13),
        n_fft=config.get("n_fft", 2048),
        hop_length=config.get("hop_length", 512),
        win_length=config.get("win_length", 2048)
    )

    features = [mfcc]

    if config.get("include_delta", False):

        delta = librosa.feature.delta(mfcc)

        features.append(delta)

    if config.get("include_delta_delta", False):

        delta_delta = librosa.feature.delta(mfcc, order=2)

        features.append(delta_delta)

    combined_mfcc = np.vstack(features)

    return combined_mfcc


def extract_mel(y, sr, config):

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=config.get("n_mels", 128),
        n_fft=config.get("n_fft", 2048),
        hop_length=config.get("hop_length", 512),
        win_length=config.get("win_length", 2048),
        power=config.get("power", 2.0)
    )

    if config.get("log_scale", True):

        mel = librosa.power_to_db(
            mel,
            ref=np.max
        )

    return mel


def extract_combined_features(y, sr, config):

    mfcc = extract_mfcc(
        y,
        sr,
        config
    )

    rms = librosa.feature.rms(
        y=y,
        frame_length=config.get("n_fft", 2048),
        hop_length=config.get("hop_length", 512)
    )

    zcr = librosa.feature.zero_crossing_rate(
        y=y,
        frame_length=config.get("n_fft", 2048),
        hop_length=config.get("hop_length", 512)
    )

    spectral_centroid = librosa.feature.spectral_centroid(
        y=y,
        sr=sr,
        n_fft=config.get("n_fft", 2048),
        hop_length=config.get("hop_length", 512)
    )

    combined = np.vstack([
        mfcc,
        rms,
        zcr,
        spectral_centroid
    ])

    return combined


def extract_audio_features(valid_dataset, feature_config):

    logger.info("=" * 70)
    logger.info("Extracting Audio Features")
    logger.info("=" * 70)

    extracted_features = []

    feature_type = feature_config["feature_type"]

    total = len(valid_dataset)

    success = 0
    failed = 0

    for index, sample in enumerate(valid_dataset, start=1):

        audio_path = Path(sample["audio_path"])
        audio_file = sample["audio_file"]

        try:

            logger.info(f"[{index}/{total}] Extracting features: {audio_file}")

            y, sr = librosa.load(
                audio_path,
                sr=feature_config.get("sample_rate", 16000),
                mono=feature_config.get("mono", True)
            )

            if y is None or len(y) == 0:

                raise ValueError("Empty audio signal")

            if feature_type == "mfcc":

                features = extract_mfcc(
                    y,
                    sr,
                    feature_config
                )

            elif feature_type == "mel":

                features = extract_mel(
                    y,
                    sr,
                    feature_config
                )

            elif feature_type == "combined":

                features = extract_combined_features(
                    y,
                    sr,
                    feature_config
                )

            else:

                raise ValueError(f"Unsupported feature type: {feature_type}")

            extracted_features.append({

                "pair_id": sample["pair_id"],

                "audio_file": audio_file,

                "audio_path": str(audio_path),

                "alignment_file": sample["alignment_file"],

                "feature_type": feature_type,

                "features": features,

                "feature_shape": features.shape,

                "sample_rate": sr,

                "duration": librosa.get_duration(
                    y=y,
                    sr=sr
                ),

                "feature_status": "Extracted",

                "message": "Feature extraction completed successfully"

            })

            success += 1

        except Exception as e:

            logger.warning(f"Feature extraction failed for {audio_file}: {e}")

            extracted_features.append({

                "pair_id": sample.get("pair_id", ""),

                "audio_file": audio_file,

                "audio_path": str(audio_path),

                "alignment_file": sample.get("alignment_file", ""),

                "feature_type": feature_type,

                "features": None,

                "feature_shape": None,

                "sample_rate": None,

                "duration": None,

                "feature_status": "Failed",

                "message": str(e)

            })

            failed += 1

    logger.info(f"Total samples     : {total}")
    logger.info(f"Extracted success : {success}")
    logger.info(f"Extraction failed : {failed}")

    logger.info("=" * 70)

    return extracted_features