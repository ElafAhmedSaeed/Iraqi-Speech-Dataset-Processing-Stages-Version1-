import re

from src.utils.logger import get_logger

logger = get_logger("Transcript Quality")


ARABIC_PATTERN = re.compile(r"[\u0600-\u06FF]")


def validate_transcript(text: str):

    characters = len(text)

    words = len(text.split())

    arabic_chars = len(ARABIC_PATTERN.findall(text))

    english_chars = len(re.findall(r"[A-Za-z]", text))

    numbers = len(re.findall(r"\d", text))

    weird_symbols = len(
        re.findall(r"[^ء-ي0-9A-Za-z\s.,!?؟،؛:]", text)
    )

    multiple_spaces = bool(
        re.search(r"\s{2,}", text)
    )

    if characters == 0:

        status = "Invalid"

        message = "Empty transcript"

    elif words == 0:

        status = "Invalid"

        message = "No words"

    elif arabic_chars == 0:

        status = "Invalid"

        message = "No Arabic text"

    elif weird_symbols > 0:

        status = "Warning"

        message = "Contains unusual symbols"

    else:

        status = "Valid"

        message = ""

    return {

        "characters": characters,

        "words": words,

        "arabic_characters": arabic_chars,

        "english_characters": english_chars,

        "numbers": numbers,

        "weird_symbols": weird_symbols,

        "multiple_spaces": multiple_spaces,

        "status": status,

        "message": message

    }