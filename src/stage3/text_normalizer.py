import re

from src.utils.logger import get_logger

logger = get_logger("Text Normalizer")


def normalize_text(text: str):

    """
    Normalize Arabic characters without changing words.
    """

    original_text = text

    # ------------------------------------------------
    # Remove Arabic Tatweel
    # ------------------------------------------------

    text = text.replace("ـ", "")

    # ------------------------------------------------
    # Normalize Alef
    # ------------------------------------------------

    text = re.sub("[أإآ]", "ا", text)

    # ------------------------------------------------
    # Normalize Ya
    # ------------------------------------------------

    text = text.replace("ى", "ي")

    # ------------------------------------------------
    # Normalize Hamza Forms
    # ------------------------------------------------

    text = text.replace("ؤ", "و")

    text = text.replace("ئ", "ي")

    # ------------------------------------------------
    # Remove Arabic Diacritics
    # ------------------------------------------------

    arabic_diacritics = re.compile(
        r"""
        ّ|َ|ً|ُ|ٌ|ِ|ٍ|ْ|ـ
        """,
        re.VERBOSE
    )

    text = re.sub(arabic_diacritics, "", text)

    text = text.strip()

    return {

        "original_text": original_text,

        "normalized_text": text,

        "changed": original_text != text

    }