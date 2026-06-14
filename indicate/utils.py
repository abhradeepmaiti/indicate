from __future__ import annotations

import json
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

from .decoder import Decoder
from .encoder import Encoder

if TYPE_CHECKING:
    from .rerank import Reranker

START_TOKEN = "^"
END_TOKEN = "$"


class CharTokenizer:
    """Character-level tokenizer holding the ``word_index``/``index_word`` maps.

    Loaded from the JSON files that were serialised by the original Keras
    ``Tokenizer`` so the vocabulary indices stay identical across the migration.
    """

    def __init__(self, word_index: dict[str, int]) -> None:
        self.word_index = word_index
        self.index_word = {index: word for word, index in word_index.items()}

    @property
    def vocab_size(self) -> int:
        # +1 for the reserved padding index (0).
        return len(self.word_index) + 1


def load_tokenizer(path: str) -> CharTokenizer:
    """Load a character tokenizer from a Keras-serialised tokenizer JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # The Keras export is a JSON-encoded string whose ``config.word_index`` is
    # itself a JSON-encoded string.
    if isinstance(data, str):
        data = json.loads(data)
    word_index = json.loads(data["config"]["word_index"])
    return CharTokenizer(word_index)


def sequence_to_chars(tokenizer: CharTokenizer, sequence: Iterable[int]) -> str:
    """Convert a sequence of indices back to characters, skipping padding (0)."""
    index_word = tokenizer.index_word
    return "".join(index_word[int(q)] for q in sequence if int(q) != 0)


def _greedy_decode(
    encoder_outputs: torch.Tensor,
    state: tuple[torch.Tensor, torch.Tensor],
    start_id: int,
    end_id: int,
    decoder: Decoder,
    max_length_output: int,
    src_mask: torch.Tensor | None,
) -> list[int]:
    dec_input = torch.tensor([[start_id]], dtype=torch.long)
    outputs: list[int] = []
    for _ in range(max_length_output):
        logits, state = decoder(dec_input, encoder_outputs, state, src_mask)
        predicted_id = int(logits[0, -1].argmax())
        outputs.append(predicted_id)
        if predicted_id == end_id:
            break
        dec_input = torch.tensor([[predicted_id]], dtype=torch.long)
    return outputs


def _beam_decode(
    encoder_outputs: torch.Tensor,
    state: tuple[torch.Tensor, torch.Tensor],
    start_id: int,
    end_id: int,
    decoder: Decoder,
    max_length_output: int,
    beam_width: int,
    src_mask: torch.Tensor | None,
) -> list[tuple[float, list[int]]]:
    """Batched beam search with length-normalized scores.

    The ``beam_width`` hypotheses are stacked into the batch dimension and the
    decoder is run once per step for all of them. Returns the finished
    hypotheses as ``(score, tokens)`` sorted best-first (tokens exclude the
    start token); falls back to the single best unfinished beam.
    """
    k = beam_width
    h, c = state  # each [1, 1, U]
    enc = encoder_outputs.expand(k, -1, -1).contiguous()  # [k, S, U]
    mask = src_mask.expand(k, -1).contiguous() if src_mask is not None else None
    h = h.expand(-1, k, -1).contiguous()
    c = c.expand(-1, k, -1).contiguous()

    seqs = torch.full((k, 1), start_id, dtype=torch.long)
    scores = torch.full((k,), float("-inf"))
    scores[0] = 0.0  # only one live beam at the start (avoids k identical beams)
    finished: list[tuple[float, list[int]]] = []

    for _ in range(max_length_output):
        last = seqs[:, -1:]  # [k, 1]
        logits, (h, c) = decoder(last, enc, (h, c), mask)
        logp = torch.log_softmax(logits[:, -1, :], dim=-1)  # [k, V]
        vocab = logp.size(-1)
        cand = scores.unsqueeze(1) + logp  # [k, V]
        top_scores, top_idx = cand.view(-1).topk(k)
        beam_idx = torch.div(top_idx, vocab, rounding_mode="floor")
        tok_idx = top_idx % vocab

        seqs = torch.cat([seqs[beam_idx], tok_idx.unsqueeze(1)], dim=1)
        h = h[:, beam_idx, :].contiguous()
        c = c[:, beam_idx, :].contiguous()
        scores = top_scores.clone()

        for i in range(k):
            if int(tok_idx[i]) == end_id:
                # Length-normalized score; drop the leading start token.
                finished.append((float(scores[i]) / seqs.size(1), seqs[i, 1:].tolist()))
                scores[i] = float("-inf")
        if bool(torch.isinf(scores).all()):
            break

    if not finished:
        return [(float(scores.max()), seqs[int(scores.argmax()), 1:].tolist())]
    finished.sort(key=lambda x: x[0], reverse=True)
    return finished


def word_candidates(
    sentence: str,
    input_lang_tokenizer: CharTokenizer,
    target_lang_tokenizer: CharTokenizer,
    encoder: Encoder,
    decoder: Decoder,
    max_length_input: int,
    max_length_output: int,
    beam_width: int = 1,
    mask_padding: bool = False,
) -> list[tuple[str, float]]:
    """Ranked ``(text, score)`` candidates for one word, best-first.

    With ``beam_width > 1`` this is the beam's length-normalized hypotheses;
    with greedy decoding it is a single candidate. Empty input -> ``[]``.
    ``mask_padding`` must match how the weights were trained.
    """
    start_id = target_lang_tokenizer.word_index[START_TOKEN]
    end_id = target_lang_tokenizer.word_index[END_TOKEN]

    # Character -> index, post-padded/truncated to the fixed input length.
    ids = [input_lang_tokenizer.word_index[ch] for ch in sentence][:max_length_input]
    if not ids:
        return []
    ids = ids + [0] * (max_length_input - len(ids))
    inputs = torch.tensor([ids], dtype=torch.long)
    src_mask = (inputs != 0) if mask_padding else None

    with torch.no_grad():
        encoder_outputs, state = encoder(inputs)
        if beam_width and beam_width > 1:
            hyps = _beam_decode(
                encoder_outputs,
                state,
                start_id,
                end_id,
                decoder,
                max_length_output,
                beam_width,
                src_mask,
            )
            return [
                (sequence_to_chars(target_lang_tokenizer, toks).strip(END_TOKEN), score)
                for score, toks in hyps
            ]
        tokens = _greedy_decode(
            encoder_outputs, state, start_id, end_id, decoder, max_length_output, src_mask
        )
        return [(sequence_to_chars(target_lang_tokenizer, tokens).strip(END_TOKEN), 0.0)]


def translate(
    sentence: str,
    input_lang_tokenizer: CharTokenizer,
    target_lang_tokenizer: CharTokenizer,
    encoder: Encoder,
    decoder: Decoder,
    max_length_input: int,
    max_length_output: int,
    beam_width: int = 1,
    reranker: Reranker | None = None,
    mask_padding: bool = False,
) -> str:
    """Best single-word transliteration (greedy, or beam if width > 1).

    With ``reranker`` set and beam on, the top-k beam hypotheses are re-scored by
    interpolating the model score with a language-model score.
    """
    cands = word_candidates(
        sentence,
        input_lang_tokenizer,
        target_lang_tokenizer,
        encoder,
        decoder,
        max_length_input,
        max_length_output,
        beam_width,
        mask_padding,
    )
    if not cands:
        return ""
    if reranker is not None and beam_width and beam_width > 1:
        return reranker.best(cands)
    return cands[0][0]
