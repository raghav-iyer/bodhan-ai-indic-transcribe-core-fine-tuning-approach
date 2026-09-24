"""Conservative Unicode handling; preserve Marathi vowel signs and diacritics."""

import re
import unicodedata


def clean_transcript(text):
    if not isinstance(text, str):
        raise ValueError("Transcript must be a string")
    text = unicodedata.normalize("NFC", text).replace("\ufeff", "")
    return re.sub(r"\s+", " ", text).strip()


def normalize_for_scoring(text):
    text = clean_transcript(text).casefold()
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return re.sub(r"\s+", " ", text).strip()
