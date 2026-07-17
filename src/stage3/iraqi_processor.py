import re

from src.utils.logger import get_logger

logger = get_logger("Iraqi Processor")


IRAQI_MAP = {

    # Time
    "هسة": "هسه",

    # Demonstratives
    "هيچي": "هيج",
    "هيجية": "هيج",

    # Common spelling
    "موو": "مو",
    "ايي": "ايي",
    "إيي": "ايي",

    # Pronunciation
    "شلونج": "شلونج",
    "شلونك": "شلونك"

}


def process_iraqi_text(text: str):

    original = text

    # ------------------------------------
    # Reduce repeated letters
    # ------------------------------------

    text = re.sub(

        r"(.)\1{2,}",

        r"\1",

        text

    )

    # ------------------------------------
    # Dictionary normalization
    # ------------------------------------

    words = []

    for word in text.split():

        words.append(

            IRAQI_MAP.get(

                word,

                word

            )

        )

    text = " ".join(words)

    return {

        "original_text": original,

        "processed_text": text,

        "changed": original != text

    }