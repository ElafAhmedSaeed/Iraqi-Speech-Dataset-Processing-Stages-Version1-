from pathlib import Path

import pandas as pd
import soundfile as sf

from src.utils.logger import get_logger

logger = get_logger("Stage6 Load Audio & Transcript")


def load_audio_transcript(project_root: Path):

    logger.info("=" * 70)
    logger.info("Loading Audio & Transcript")
    logger.info("=" * 70)

    metadata_file = project_root / "metadata" / "metadata.csv"
    audio_folder = project_root / "data" / "processed_audio"

    if not metadata_file.exists():

        logger.error("metadata.csv not found.")

        return []

    df = pd.read_csv(metadata_file)

    dataset = []

    loaded = 0
    missing = 0

    for _, row in df.iterrows():

        audio_path = audio_folder / row["audio_file"]

        if not audio_path.exists():

            logger.warning(f"Missing audio: {audio_path.name}")

            missing += 1
            continue

        try:

            audio_info = sf.info(audio_path)

        except Exception as e:

            logger.warning(f"Cannot read {audio_path.name}: {e}")

            missing += 1
            continue

        dataset.append({

            "pair_id": row["pair_id"],

            "audio_file": row["audio_file"],

            "audio_path": audio_path,

            "transcript": row["normalized_text"],

            "sample_rate": audio_info.samplerate,

            "duration": audio_info.duration,

            "channels": audio_info.channels

        })

        loaded += 1

    logger.info(f"Loaded : {loaded}")
    logger.info(f"Missing: {missing}")

    logger.info("=" * 70)

    return dataset