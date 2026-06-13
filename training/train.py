#!/usr/bin/env python
"""Train the Hindi -> English transliteration model (PyTorch).

Reads the committed parallel corpus ``data/hindi.csv.gz`` (columns ``hindi`` and
``english``), trains the encoder-decoder with Luong attention using teacher
forcing, and saves the best weights as safetensors plus the character
tokenizers used.

The vocabulary is taken from the existing tokenizer JSON files so the indices
stay identical to the original model; pass ``--rebuild-vocab`` to instead fit
fresh character vocabularies from the data.

Example:
    python training/train.py --epochs 25 --batch-size 64
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from indicate.decoder import Decoder  # noqa: E402
from indicate.encoder import Encoder  # noqa: E402
from indicate.utils import (  # noqa: E402
    END_TOKEN,
    START_TOKEN,
    CharTokenizer,
    load_tokenizer,
)

DEFAULT_DATA = REPO_ROOT / "data" / "hindi.csv.gz"
WEIGHTS_DIR = REPO_ROOT / "indicate" / "data" / "hindi_to_english" / "saved_weights"
INPUT_VOCAB = REPO_ROOT / "indicate" / "data" / "hindi_to_english" / "hindi_tokens.json"
TARGET_VOCAB = (
    REPO_ROOT / "indicate" / "data" / "hindi_to_english" / "english_tokens.json"
)

EMBEDDING_DIM = 256
UNITS = 1024
MAX_LENGTH_INPUT = 47
MAX_LENGTH_OUTPUT = 173
PAD_ID = 0


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_vocab(words: list[str], extra: tuple[str, ...] = ()) -> CharTokenizer:
    """Fit a character vocabulary (index 0 reserved for padding)."""
    counts: dict[str, int] = {}
    for token in extra:
        counts[token] = counts.get(token, 0) + 1
    for word in words:
        for ch in word:
            counts[ch] = counts.get(ch, 0) + 1
    # Most frequent first, mirroring Keras Tokenizer ordering.
    ordered = sorted(counts, key=lambda c: counts[c], reverse=True)
    word_index = {ch: i + 1 for i, ch in enumerate(ordered)}
    return CharTokenizer(word_index)


def load_pairs(path: Path) -> list[tuple[str, str]]:
    """Read (source, english) pairs from the first two CSV columns.

    Column-name agnostic so the same loader handles ``hindi.csv`` (hindi,english)
    and ``punjabi.csv`` (punjabi,english). Transparently reads gzip (``.csv.gz``)
    or plain ``.csv``.
    """
    opener = gzip.open if str(path).endswith(".gz") else open
    pairs: list[tuple[str, str]] = []
    with opener(path, "rt", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header row
        for row in reader:
            if len(row) < 2:
                continue
            source, english = row[0].strip(), row[1].strip()
            if source and english:
                pairs.append((source, english))
    return pairs


class TransliterationDataset(Dataset):
    """Tokenises pairs and yields ``(input_ids, dec_in_ids, dec_target_ids)``.

    The target is wrapped as ``^ <english> $``; the decoder input is the target
    shifted right (``^ ...``) and the decoder target is shifted left (``... $``).
    Rows with characters outside the vocabulary, or longer than the fixed max
    lengths, are dropped at construction time.
    """

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        input_tok: CharTokenizer,
        target_tok: CharTokenizer,
        max_input: int = MAX_LENGTH_INPUT,
        max_output: int = MAX_LENGTH_OUTPUT,
    ) -> None:
        self.samples: list[tuple[list[int], list[int], list[int]]] = []
        # Raw (source, english) pairs that survived filtering, aligned 1:1 with
        # ``samples`` so callers can recover the held-out split's source strings.
        self.kept_pairs: list[tuple[str, str]] = []
        in_index = input_tok.word_index
        tgt_index = target_tok.word_index
        start_id = tgt_index[START_TOKEN]
        end_id = tgt_index[END_TOKEN]
        dropped = 0
        for source, english in pairs:
            try:
                src = [in_index[ch] for ch in source]
                tgt = [start_id] + [tgt_index[ch] for ch in english] + [end_id]
            except KeyError:
                dropped += 1
                continue
            if len(src) > max_input or len(tgt) > max_output + 1:
                dropped += 1
                continue
            self.samples.append((src, tgt[:-1], tgt[1:]))
            self.kept_pairs.append((source, english))
        self.dropped = dropped

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[list[int], list[int], list[int]]:
        return self.samples[idx]


def collate(
    batch: list[tuple[list[int], list[int], list[int]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad each field to the longest sequence in the batch (pad id = 0)."""

    def pad(seqs: list[list[int]]) -> torch.Tensor:
        width = max(len(s) for s in seqs)
        return torch.tensor(
            [s + [PAD_ID] * (width - len(s)) for s in seqs], dtype=torch.long
        )

    src = pad([b[0] for b in batch])
    dec_in = pad([b[1] for b in batch])
    dec_tgt = pad([b[2] for b in batch])
    return src, dec_in, dec_tgt


def run_epoch(
    encoder: Encoder,
    decoder: Decoder,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    desc: str,
) -> float:
    train = optimizer is not None
    encoder.train(train)
    decoder.train(train)
    total_loss = 0.0
    total_tokens = 0
    for src, dec_in, dec_tgt in tqdm(loader, desc=desc, leave=False):
        src = src.to(device)
        dec_in = dec_in.to(device)
        dec_tgt = dec_tgt.to(device)
        src_mask = src != PAD_ID
        with torch.set_grad_enabled(train):
            enc_out, state = encoder(src)
            logits, _ = decoder(dec_in, enc_out, state, src_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), dec_tgt.reshape(-1))
        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()), 5.0
            )
            optimizer.step()
        n_tokens = int((dec_tgt != PAD_ID).sum())
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens
    return total_loss / max(total_tokens, 1)


def save_tokenizer_json(tok: CharTokenizer, path: Path) -> None:
    """Persist a tokenizer in the same doubly-encoded JSON shape we read."""
    config = {
        "num_words": None,
        "filters": "",
        "lower": True,
        "split": " ",
        "char_level": True,
        "oov_token": None,
        "document_count": 0,
        "word_index": json.dumps(tok.word_index),
        "index_word": json.dumps({str(i): w for w, i in tok.word_index.items()}),
    }
    payload = json.dumps({"class_name": "Tokenizer", "config": config})
    path.write_text(json.dumps(payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=REPO_ROOT / "indicate" / "data" / "hindi_to_english",
        help="dir holding the tokenizer JSONs and saved_weights/",
    )
    parser.add_argument("--input-vocab-name", default="hindi_tokens.json")
    parser.add_argument("--target-vocab-name", default="english_tokens.json")
    parser.add_argument("--max-input", type=int, default=MAX_LENGTH_INPUT)
    parser.add_argument("--max-output", type=int, default=MAX_LENGTH_OUTPUT)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="cap #pairs (smoke test)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--rebuild-vocab",
        action="store_true",
        help="fit fresh char vocabularies instead of reusing the JSON files",
    )
    args = parser.parse_args()

    weights_dir = args.model_dir / "saved_weights"
    input_vocab = args.model_dir / args.input_vocab_name
    target_vocab = args.model_dir / args.target_vocab_name

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = pick_device(args.device)
    print(f"Device: {device}")

    pairs = load_pairs(args.data)
    random.shuffle(pairs)
    if args.limit:
        pairs = pairs[: args.limit]
    print(f"Loaded {len(pairs)} pairs from {args.data}")

    if args.rebuild_vocab:
        input_tok = build_vocab([h for h, _ in pairs])
        target_tok = build_vocab([e for _, e in pairs], extra=(START_TOKEN, END_TOKEN))
    else:
        input_tok = load_tokenizer(str(input_vocab))
        target_tok = load_tokenizer(str(target_vocab))
    print(
        f"Vocab sizes - input: {input_tok.vocab_size}, target: {target_tok.vocab_size}"
    )

    dataset = TransliterationDataset(
        pairs, input_tok, target_tok, args.max_input, args.max_output
    )
    print(f"Usable samples: {len(dataset)} (dropped {dataset.dropped})")

    val_size = int(len(dataset) * args.val_frac)
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
    )

    encoder = Encoder(input_tok.vocab_size, EMBEDDING_DIM, UNITS).to(device)
    decoder = Decoder(target_tok.vocab_size, EMBEDDING_DIM, UNITS).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr
    )

    weights_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            encoder,
            decoder,
            train_loader,
            criterion,
            device,
            optimizer,
            f"epoch {epoch} train",
        )
        val_loss = run_epoch(
            encoder,
            decoder,
            val_loader,
            criterion,
            device,
            None,
            f"epoch {epoch} val",
        )
        print(f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            save_file(
                {k: v.cpu() for k, v in encoder.state_dict().items()},
                str(weights_dir / "encoder.safetensors"),
            )
            save_file(
                {k: v.cpu() for k, v in decoder.state_dict().items()},
                str(weights_dir / "decoder.safetensors"),
            )
            if args.rebuild_vocab:
                save_tokenizer_json(input_tok, input_vocab)
                save_tokenizer_json(target_tok, target_vocab)
            print(f"  saved best weights (val_loss={best_val:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    print(f"Done. Best val_loss={best_val:.4f}. Weights in {weights_dir}")


if __name__ == "__main__":
    main()
