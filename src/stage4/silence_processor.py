import librosa
import numpy as np

from src.utils.logger import get_logger

logger = get_logger("Silence Processor")


TOP_DB = 30


def process_silence(audio_data: dict):

    audio = audio_data["audio"]

    sample_rate = audio_data["sample_rate"]

    original_duration = len(audio) / sample_rate

    # ------------------------------------------
    # Detect Non-Silent Region
    # ------------------------------------------

    intervals = librosa.effects.split(

        audio,

        top_db=TOP_DB

    )

    # ------------------------------------------
    # No Speech Found
    # ------------------------------------------

    if len(intervals) == 0:

        return {

            **audio_data,

            "silence_processed": False,

            "removed_duration": 0.0,

            "processed_duration": original_duration

        }

    start = intervals[0][0]

    end = intervals[-1][1]

    processed_audio = audio[start:end]

    processed_duration = len(processed_audio) / sample_rate

    removed_duration = original_duration - processed_duration

    return {

        **audio_data,

        "audio": processed_audio,

        "samples": len(processed_audio),

        "duration": processed_duration,

        "processed_duration": processed_duration,

        "removed_duration": removed_duration,

        "silence_processed": True

    }