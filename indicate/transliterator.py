from __future__ import annotations

import os
from importlib.resources import files
from typing import TYPE_CHECKING, Self, cast

from func_timeout import FunctionTimedOut, func_timeout
from safetensors.torch import load_file

from .decoder import Decoder
from .encoder import Encoder
from .logging import get_logger
from .utils import CharTokenizer, load_tokenizer, translate

if TYPE_CHECKING:
    from .rerank import Reranker

logger = get_logger()


class Seq2SeqTransliterator:
    """Base char-level seq2seq transliterator: lazy-loaded singleton.

    Subclasses point ``MODELFN`` / ``INPUT_VOCAB`` / ``TARGET_VOCAB`` (and the max
    lengths) at a specific language's safetensors weights and tokenizer JSONs.
    All mutable state is assigned on the concrete subclass, so each language's
    model is loaded and cached independently.
    """

    MODELFN: str = ""
    INPUT_VOCAB: str = ""
    TARGET_VOCAB: str = ""

    embedding_dim: int = 256
    units: int = 1024

    max_length_input: int = 64
    max_length_output: int = 64
    START_TOKEN: str = "^"
    END_TOKEN: str = "$"

    # Decoding: 1 = greedy; >1 enables length-normalized beam search. Beam search
    # is the shipped default (+1.5 pts exact-match on Hindi Dakshina vs greedy).
    BEAM_WIDTH: int = 5
    # Optional LM re-ranker for beam candidates (opt-in; None = ship default).
    RERANKER: Reranker | None = None
    # True only for weights trained with attention padding masks (train.py masks
    # by default); must match the shipped weights or inference degrades.
    MASK_PADDING: bool = False

    _instance: Seq2SeqTransliterator | None = None
    _weights_loaded: bool = False

    input_lang_tokenizer: CharTokenizer | None = None
    target_lang_tokenizer: CharTokenizer | None = None
    encoder: Encoder | None = None
    decoder: Decoder | None = None

    def __new__(cls) -> Self:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cast(Self, cls._instance)

    @classmethod
    def _resource(cls, rel: str) -> str:
        return os.path.join(str(files(__package__)), rel)

    @classmethod
    def get_model_path(cls) -> str:
        return cls._resource(cls.MODELFN)

    @classmethod
    def get_input_vocab(cls) -> str:
        return cls._resource(cls.INPUT_VOCAB)

    @classmethod
    def get_target_vocab(cls) -> str:
        return cls._resource(cls.TARGET_VOCAB)

    @classmethod
    def _load_weights(cls) -> None:
        try:
            model_path = cls.get_model_path()
            cls.input_lang_tokenizer = load_tokenizer(cls.get_input_vocab())
            cls.target_lang_tokenizer = load_tokenizer(cls.get_target_vocab())

            cls.encoder = Encoder(
                cls.input_lang_tokenizer.vocab_size, cls.embedding_dim, cls.units
            )
            cls.decoder = Decoder(
                cls.target_lang_tokenizer.vocab_size, cls.embedding_dim, cls.units
            )

            logger.debug(f"Restoring model weights from {model_path}")
            cls.encoder.load_state_dict(load_file(f"{model_path}/encoder.safetensors"))
            cls.decoder.load_state_dict(load_file(f"{model_path}/decoder.safetensors"))
            cls.encoder.eval()
            cls.decoder.eval()
            cls._weights_loaded = True
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

    @classmethod
    def transliterate(cls, input: str) -> str:
        """
        Transliterate the input text to English, one whitespace word at a time.

        Args:
            input (str): source-language text
        Returns:
            output (str): English transliteration
        Raises:
            TypeError: If input is None
            ValueError: If input is not a string
            RuntimeError: If model loading fails
        """
        if input is None:
            raise TypeError("Input cannot be None")
        if not isinstance(input, str):
            raise ValueError("Input must be a string")
        if not input.strip():
            # Handle whitespace-only input gracefully
            return ""

        if not cls._weights_loaded:
            cls._load_weights()

        # Split on spaces and transliterate each word independently. This avoids
        # the decoder running away on multi-word inputs and bounds each word to
        # the 10-second timeout below.
        words = input.split(" ") if " " in input else [input]

        output = []
        for word in words:
            target = ""
            try:
                target = func_timeout(
                    10,
                    translate,
                    args=(
                        word,
                        cls.input_lang_tokenizer,
                        cls.target_lang_tokenizer,
                        cls.encoder,
                        cls.decoder,
                        cls.max_length_input,
                        cls.max_length_output,
                        cls.BEAM_WIDTH,
                        cls.RERANKER,
                        cls.MASK_PADDING,
                    ),
                )
                logger.debug(f"Model predicted {target}")
            except FunctionTimedOut as fex:
                logger.error(
                    f"Not able to transliterate {input} within 10 seconds, exiting with {fex}"
                )
            except Exception as exe:
                logger.error(f"Not able to transliterate {input} due to {exe}")
            output.append(target)

        return " ".join(output)
