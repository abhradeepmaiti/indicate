from __future__ import annotations

from .transliterator import Seq2SeqTransliterator


class PunjabiToEnglish(Seq2SeqTransliterator):
    """Punjabi (Gurmukhi) → English transliteration model."""

    SUBDIR = "punjabi_to_english"
    INPUT_VOCAB = "punjabi_tokens.json"
    TARGET_VOCAB = "english_tokens.json"

    # Word-level training data: Gurmukhi/English words are short (max ~20 chars).
    max_length_input = 32
    max_length_output = 32
