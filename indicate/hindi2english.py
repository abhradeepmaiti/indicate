from __future__ import annotations

from .transliterator import Seq2SeqTransliterator


class HindiToEnglish(Seq2SeqTransliterator):
    """Hindi (Devanagari) → English transliteration model."""

    MODELFN = "data/hindi_to_english/saved_weights/"
    INPUT_VOCAB = "data/hindi_to_english/hindi_tokens.json"
    TARGET_VOCAB = "data/hindi_to_english/english_tokens.json"

    max_length_input = 47
    max_length_output = 173
