from __future__ import annotations

from .transliterator import Seq2SeqTransliterator


class PunjabiToEnglish(Seq2SeqTransliterator):
    """Punjabi (Gurmukhi) → English transliteration model."""

    MODELFN = "data/punjabi_to_english/saved_weights/"
    INPUT_VOCAB = "data/punjabi_to_english/punjabi_tokens.json"
    TARGET_VOCAB = "data/punjabi_to_english/english_tokens.json"

    # Word-level training data: Gurmukhi/English words are short (max ~20 chars).
    max_length_input = 32
    max_length_output = 32
