"""
Stage 1
Pair Generator

Generate permanent Pair IDs for audio/text pairs.
"""

from pathlib import Path
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("Pair Generator")


def generate_pair_index():

    project_root = Path(__file__).resolve().parents[2]

    mapping_file = project_root / "metadata" / "rename_mapping.csv"

    if not mapping_file.exists():

        logger.error("rename_mapping.csv not found.")

        return

    df = pd.read_csv(mapping_file)

    audio_df = df[df["file_type"] == "Audio"].copy()
    text_df = df[df["file_type"] == "Transcript"].copy()

    # استخراج الاسم بدون الامتداد
    audio_df["stem"] = audio_df["new_name"].apply(lambda x: Path(x).stem)
    text_df["stem"] = text_df["new_name"].apply(lambda x: Path(x).stem)

    pairs = []

    pair_counter = 1

    for _, audio in audio_df.iterrows():

        stem = audio["stem"]

        transcript = text_df[text_df["stem"] == stem]

        if len(transcript) > 0:

            transcript_name = transcript.iloc[0]["new_name"]

            status = "Matched"

        else:

            transcript_name = ""

            status = "Missing Transcript"

        pairs.append({

            "pair_id": f"PAIR_{pair_counter:06d}",

            "audio_file": audio["new_name"],

            "transcript_file": transcript_name,

            "status": status

        })

        pair_counter += 1

    pair_df = pd.DataFrame(pairs)

    output = project_root / "metadata" / "pair_index.csv"

    pair_df.to_csv(output, index=False, encoding="utf-8-sig")

    logger.info(f"{len(pair_df)} pairs generated.")

    logger.info(f"Saved: {output}")

    return pair_df