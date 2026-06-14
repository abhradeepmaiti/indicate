from __future__ import annotations

from .transliterator import Seq2SeqTransliterator


class HindiToEnglish(Seq2SeqTransliterator):
    """Hindi (Devanagari) → English transliteration model."""

    SUBDIR = "hindi_to_english"
    INPUT_VOCAB = "hindi_tokens.json"
    TARGET_VOCAB = "english_tokens.json"

    max_length_input = 47
    max_length_output = 173
