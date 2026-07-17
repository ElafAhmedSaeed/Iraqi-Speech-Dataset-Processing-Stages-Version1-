from src.utils.logger import get_logger

logger = get_logger("Audio Validator")


TARGET_SAMPLE_RATE = 16000

MIN_DURATION = 0.5      # seconds
MAX_DURATION = 60.0     # seconds

MAX_SILENCE_RATIO = 0.70

MAX_CLIPPING_RATIO = 0.01

MAX_DC_OFFSET = 0.01

MIN_RMS = 0.005


def validate_audio(audio_data: dict):

    issues = []

    warnings = []

    # ------------------------------------------
    # Sample Rate
    # ------------------------------------------

    if audio_data["sample_rate"] != TARGET_SAMPLE_RATE:

        issues.append("Invalid Sample Rate")

    # ------------------------------------------
    # Channels
    # ------------------------------------------

    if audio_data["channels"] != 1:

        issues.append("Audio is not Mono")

    # ------------------------------------------
    # Duration
    # ------------------------------------------

    duration = audio_data["duration"]

    if duration < MIN_DURATION:

        issues.append("Audio too short")

    elif duration > MAX_DURATION:

        warnings.append("Long audio")

    # ------------------------------------------
    # RMS
    # ------------------------------------------

    if audio_data["rms"] < MIN_RMS:

        warnings.append("Low audio level")

    # ------------------------------------------
    # Clipping
    # ------------------------------------------

    if audio_data["clipping_ratio"] > MAX_CLIPPING_RATIO:

        warnings.append("Possible clipping")

    # ------------------------------------------
    # Silence Ratio
    # ------------------------------------------

    if audio_data["silence_ratio"] > MAX_SILENCE_RATIO:

        warnings.append("Too much silence")

    # ------------------------------------------
    # DC Offset
    # ------------------------------------------

    if abs(audio_data["dc_offset"]) > MAX_DC_OFFSET:

        warnings.append("High DC Offset")

    # ------------------------------------------
    # NaN / Inf
    # ------------------------------------------

    if audio_data["has_nan"]:

        issues.append("NaN values detected")

    if audio_data["has_inf"]:

        issues.append("Infinite values detected")

    # ------------------------------------------
    # Final Status
    # ------------------------------------------

    if len(issues) > 0:

        status = "Invalid"

    elif len(warnings) > 0:

        status = "Warning"

    else:

        status = "Valid"

    return {

        **audio_data,

        "validation_status": status,

        "issues": "; ".join(issues),

        "warnings": "; ".join(warnings)

    }