import numpy as np

from src.utils.logger import get_logger

logger = get_logger("Audio Enhancer")


def enhance_audio(audio_data: dict):

    audio = audio_data["audio"]

    # ------------------------------------------
    # RMS
    # ------------------------------------------

    rms = float(np.sqrt(np.mean(audio ** 2)))

    # ------------------------------------------
    # Peak
    # ------------------------------------------

    peak = float(np.max(np.abs(audio)))

    # ------------------------------------------
    # Peak Normalization
    # ------------------------------------------

    if peak > 0:

        audio = audio / peak

    # ------------------------------------------
    # Clipping Detection
    # ------------------------------------------

    clipping_samples = int(np.sum(np.abs(audio) >= 0.999))

    clipping_ratio = clipping_samples / len(audio)

    # ------------------------------------------
    # Silence Ratio
    # ------------------------------------------

    silence_threshold = 0.001

    silence_samples = int(np.sum(np.abs(audio) < silence_threshold))

    silence_ratio = silence_samples / len(audio)

    # ------------------------------------------
    # DC Offset
    # ------------------------------------------

    dc_offset = float(np.mean(audio))

    # ------------------------------------------
    # NaN / Inf
    # ------------------------------------------

    has_nan = bool(np.isnan(audio).any())

    has_inf = bool(np.isinf(audio).any())

    return {

        **audio_data,

        "audio": audio,

        "rms": rms,

        "peak": peak,

        "clipping_ratio": clipping_ratio,

        "silence_ratio": silence_ratio,

        "dc_offset": dc_offset,

        "has_nan": has_nan,

        "has_inf": has_inf,

        "enhanced": True

    }