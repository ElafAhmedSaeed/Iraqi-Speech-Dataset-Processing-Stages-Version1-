from pathlib import Path
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("Transcript Validator")


def validate_transcripts(metadata_path: Path,
                         transcript_folder: Path):

    logger.info("=" * 70)
    logger.info("Validating Transcript Files...")
    logger.info("=" * 70)

    metadata = pd.read_csv(metadata_path)

    results = []

    seen_text = set()

    for _, row in metadata.iterrows():

        transcript_name = row["transcript_file"]

        transcript_path = transcript_folder / transcript_name

        logger.info(f"Checking: {transcript_name}")

        status = "Valid"

        message = ""

        exists = transcript_path.exists()

        readable = False

        empty = False

        utf8 = True

        duplicated = False

        characters = 0

        words = 0

        lines = 0

        if not exists:

            status = "Invalid"

            message = "File Not Found"

        else:

            try:

                with open(transcript_path,
                          "r",
                          encoding="utf-8") as f:

                    text = f.read()

                readable = True

            except UnicodeDecodeError:

                utf8 = False

                status = "Invalid"

                message = "Encoding Error"

                text = ""

            except Exception as e:

                status = "Invalid"

                message = str(e)

                text = ""

            if readable:

                characters = len(text)

                words = len(text.split())

                lines = len(text.splitlines())

                if characters == 0:

                    empty = True

                    status = "Invalid"

                    message = "Empty File"

                if text in seen_text:

                    duplicated = True

                else:

                    seen_text.add(text)

        results.append({

            "pair_id": row["pair_id"],

            "transcript_file": transcript_name,

            "exists": exists,

            "readable": readable,

            "utf8": utf8,

            "empty": empty,

            "duplicated": duplicated,

            "characters": characters,

            "words": words,

            "lines": lines,

            "status": status,

            "message": message

        })

    df = pd.DataFrame(results)

    logger.info(f"Validated {len(df)} transcript files.")

    logger.info("=" * 70)

    return df