from pathlib import Path

import soundfile as sf

from src.utils.logger import get_logger

logger = get_logger("Load Audio")


SUPPORTED_AUDIO = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a"
}


def load_audio_files(project_root: Path):

    logger.info("=" * 70)
    logger.info("Loading Audio Files...")
    logger.info("=" * 70)

    audio_folder = project_root / "data" / "raw" / "audio"

    audio_files = []

    files = sorted(audio_folder.glob("*"))

    if len(files) == 0:

        logger.warning("No audio files found.")

        return audio_files

    for file in files:

        if file.suffix.lower() not in SUPPORTED_AUDIO:
            continue

        try:

            audio, sample_rate = sf.read(file)

            if audio.ndim == 1:
                channels = 1
            else:
                channels = audio.shape[1]

            duration = len(audio) / sample_rate

            audio_files.append({

                "file_name": file.name,

                "file_path": str(file),

                "audio": audio,

                "sample_rate": sample_rate,

                "channels": channels,

                "duration": duration,

                "samples": len(audio),

                "dtype": str(audio.dtype)

            })

        except Exception as e:

            logger.error(f"Cannot read {file.name}")

            logger.error(str(e))

    logger.info(f"Loaded {len(audio_files)} audio file(s).")

    logger.info("=" * 70)

    return audio_files