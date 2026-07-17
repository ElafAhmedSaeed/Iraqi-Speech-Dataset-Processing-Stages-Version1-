from pathlib import Path

import librosa

from src.utils.logger import get_logger

logger = get_logger("Stage5 Audio Loader")


def load_processed_audio(project_root: Path):

    logger.info("=" * 70)
    logger.info("Loading Processed Audio Files")
    logger.info("=" * 70)

    audio_folder = project_root / "data" / "processed_audio"

    if not audio_folder.exists():

        logger.warning("Processed audio folder not found.")

        return []

    audio_files = sorted(audio_folder.glob("*.wav"))

    if len(audio_files) == 0:

        logger.warning("No processed audio files found.")

        return []

    dataset = []

    for audio_path in audio_files:

        try:

            audio, sample_rate = librosa.load(

                audio_path,

                sr=None,

                mono=False

            )

            if audio.ndim == 1:

                channels = 1
                samples = len(audio)

            else:

                channels = audio.shape[0]
                samples = audio.shape[1]

            duration = samples / sample_rate

            dataset.append({

                "file_name": audio_path.name,

                "file_path": str(audio_path),

                "audio": audio,

                "sample_rate": sample_rate,

                "channels": channels,

                "samples": samples,

                "duration": duration

            })

        except Exception as e:

            logger.error(f"Cannot load {audio_path.name}")

            logger.error(str(e))

    logger.info(f"Loaded {len(dataset)} processed audio file(s).")

    logger.info("=" * 70)

    return dataset