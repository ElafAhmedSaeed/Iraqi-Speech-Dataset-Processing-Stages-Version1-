import numpy as np
import librosa

from src.utils.logger import get_logger

logger = get_logger("Audio Standardizer")


TARGET_SAMPLE_RATE = 16000


def standardize_audio(audio_data: dict):

    audio = audio_data["audio"]

    sample_rate = audio_data["sample_rate"]

    original_sample_rate = sample_rate

    original_channels = audio_data["channels"]

    # ------------------------------------------
    # Stereo → Mono
    # ------------------------------------------

    if audio.ndim > 1:

        audio = np.mean(audio, axis=1)

    # ------------------------------------------
    # Convert to float32
    # ------------------------------------------

    audio = audio.astype(np.float32)

    # ------------------------------------------
    # Remove DC Offset
    # ------------------------------------------

    audio = audio - np.mean(audio)

    # ------------------------------------------
    # Resample
    # ------------------------------------------

    if sample_rate != TARGET_SAMPLE_RATE:

        audio = librosa.resample(
            y=audio,
            orig_sr=sample_rate,
            target_sr=TARGET_SAMPLE_RATE
        )

        sample_rate = TARGET_SAMPLE_RATE

    # ------------------------------------------
    # Peak Normalization
    # ------------------------------------------

    peak = np.max(np.abs(audio))

    if peak > 0:

        audio = audio / peak

    return {

        **audio_data,

        "audio": audio,

        "sample_rate": sample_rate,

        "channels": 1,

        "samples": len(audio),

        "duration": len(audio) / sample_rate,

        "original_sample_rate": original_sample_rate,

        "original_channels": original_channels,

        "standardized": True

    }