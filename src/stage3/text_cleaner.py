import re

from src.utils.logger import get_logger

logger = get_logger("Text Cleaner")


def clean_text(text: str):

    """
    Basic text cleaning without changing the content.
    """

    original_text = text

    # ------------------------------------------
    # Remove UTF-8 BOM
    # ------------------------------------------

    text = text.replace("\ufeff", "")

    # ------------------------------------------
    # Convert Tabs to Spaces
    # ------------------------------------------

    text = text.replace("\t", " ")

    # ------------------------------------------
    # Remove Control Characters
    # ------------------------------------------

    text = "".join(

        ch

        for ch in text

        if ch.isprintable() or ch in "\n"

    )

    # ------------------------------------------
    # Normalize Line Endings
    # ------------------------------------------

    text = text.replace("\r\n", "\n")

    text = text.replace("\r", "\n")

    # ------------------------------------------
    # Remove Empty Lines
    # ------------------------------------------

    text = "\n".join(

        line.strip()

        for line in text.split("\n")

        if line.strip()

    )

    # ------------------------------------------
    # Collapse Multiple Spaces
    # ------------------------------------------

    text = re.sub(

        r"\s+",

        " ",

        text

    )

    # ------------------------------------------
    # Remove Spaces Before Punctuation
    # ------------------------------------------

    text = re.sub(

        r"\s+([.,!?،؛:؟])",

        r"\1",

        text

    )

    # ------------------------------------------
    # Trim Text
    # ------------------------------------------

    text = text.strip()

    return {

        "original_text": original_text,

        "clean_text": text,

        "changed": original_text != text

    }