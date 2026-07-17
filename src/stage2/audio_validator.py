#بعد تشغيله سيولد:
#
# metadata/
# │
# ├── metadata.csv
# │
# └── audio_validation.csv

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.utils.logger import get_logger

logger = get_logger("Audio Validator")


SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a"
}


# --------------------------------------------------
# RMS
# --------------------------------------------------

def calculate_rms(audio):

    if len(audio) == 0:
        return 0.0

    return float(np.sqrt(np.mean(np.square(audio))))


# --------------------------------------------------
# Peak
# --------------------------------------------------

def calculate_peak(audio):

    if len(audio) == 0:
        return 0.0

    return float(np.max(np.abs(audio)))


# --------------------------------------------------
# Silence Ratio
# --------------------------------------------------

def calculate_silence_ratio(audio, threshold=0.001):

    if len(audio) == 0:
        return 1.0

    silent = np.sum(np.abs(audio) < threshold)

    return float(silent / len(audio))


# --------------------------------------------------
# Validate One Audio File
# --------------------------------------------------

def validate_audio_file(audio_path: Path):

    result = {

        "audio_valid": False,
        "duration_sec": None,
        "sample_rate": None,
        "channels": None,
        "bit_depth": None,
        "rms": None,
        "peak": None,
        "silence_ratio": None,
        "validation_message": ""

    }

    # ---------------- File Exists ----------------

    if not audio_path.exists():

        result["validation_message"] = "File Not Found"

        return result

    # ---------------- Extension ----------------

    if audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:

        result["validation_message"] = "Unsupported Extension"

        return result

    try:

        audio, sample_rate = sf.read(audio_path)

        info = sf.info(audio_path)

    except Exception as e:

        result["validation_message"] = str(e)

        return result

    # ---------------- Empty ----------------

    if len(audio) == 0:

        result["validation_message"] = "Empty Audio"

        return result

    # ---------------- Channels ----------------

    if audio.ndim == 1:

        channels = 1

    else:

        channels = audio.shape[1]

        # للتحليل نحول إلى Mono
        audio = np.mean(audio, axis=1)

    duration = len(audio) / sample_rate

    result["audio_valid"] = True

    result["duration_sec"] = round(duration, 3)

    result["sample_rate"] = sample_rate

    result["channels"] = channels

    result["bit_depth"] = info.subtype

    result["rms"] = round(calculate_rms(audio), 6)

    result["peak"] = round(calculate_peak(audio), 6)

    result["silence_ratio"] = round(calculate_silence_ratio(audio), 6)

    result["validation_message"] = "OK"

    return result


# --------------------------------------------------
# Validate Dataset
# --------------------------------------------------

def validate_audio_dataset(metadata_path: Path, audio_folder: Path):

    logger.info("Starting Audio Validation...")

    metadata = pd.read_csv(metadata_path)

    results = []

    for _, row in metadata.iterrows():

        audio_name = row["audio_file"]

        audio_path = audio_folder / audio_name

        logger.info(f"Checking: {audio_name}")

        validation = validate_audio_file(audio_path)

        validation["pair_id"] = row["pair_id"]

        validation["audio_file"] = audio_name

        results.append(validation)

    df = pd.DataFrame(results)

    logger.info(f"Validated {len(df)} audio files.")

    return df


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    project_root = Path(__file__).resolve().parents[2]

    metadata_path = project_root / "metadata" / "metadata.csv"

    audio_folder = project_root / "data" / "raw" / "audio"

    validation = validate_audio_dataset(

        metadata_path,
        audio_folder

    )

    output = project_root / "metadata" / "audio_validation.csv"

    validation.to_csv(

        output,

        index=False,

        encoding="utf-8-sig"

    )

    logger.info(f"Saved: {output}")