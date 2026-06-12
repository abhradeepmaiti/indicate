#!/usr/bin/env python
"""Extract clean (Gurmukhi -> English) word pairs for training a Punjabi model.

Source: ``data/punjab_transliteration_subset.parquet`` (not committed) — Punjab
electoral-roll text whose Gurmukhi runs were transliterated to English by GPT-4o
(see ``examples/punjab_transliteration.ipynb``). Each ``<field>`` has a matching
``<field>_transliterated`` where every Gurmukhi run was replaced *in place*, so
the two are positionally aligned and word pairs can be recovered.

For every row/field we extract the Gurmukhi runs from the source and the Latin
runs from the target; when their counts match we zip them into word pairs. Rows
that don't align (e.g. pre-existing Latin inflating the target) are skipped.

Output: ``data/punjabi.csv`` (columns ``punjabi,english``), deduped to unique
pairs and committed as the Punjabi analogue of ``data/hindi.csv``.

Example:
    python training/extract_punjabi.py
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARQUET = REPO_ROOT / "data" / "punjab_transliteration_subset.parquet"
DEFAULT_OUT = REPO_ROOT / "data" / "punjabi.csv"

GURMUKHI_RUN = re.compile(r"[਀-੿]+")
LATIN_RUN = re.compile(r"[A-Za-z]+")

# Source fields whose ``<field>_transliterated`` partner holds the romanization.
SOURCE_FIELDS = [
    "elector_name",
    "father_or_husband_name",
    "ac_name",
    "parl_constituency",
    "main_town",
    "police_station",
    "mandal",
    "revenue_division",
    "district",
    "polling_station_name",
    "polling_station_address",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument(
        "--max-len", type=int, default=40, help="drop word pairs longer than this"
    )
    args = parser.parse_args()

    pf = pq.ParquetFile(args.parquet)
    columns = [c for f in SOURCE_FIELDS for c in (f, f"{f}_transliterated")]
    n_rows = pf.metadata.num_rows

    # For each Gurmukhi word, count how often each English mapping is observed.
    # The GPT-4o pipeline used a deterministic 1:1 word map, so the *modal*
    # English per word is the intended one; rare alternates are alignment noise.
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    rows_seen = aligned = mismatched = 0

    with tqdm(total=n_rows, desc="rows", unit="row") as bar:
        for batch in pf.iter_batches(batch_size=args.batch_size, columns=columns):
            cols = batch.to_pydict()
            batch_len = len(cols[SOURCE_FIELDS[0]])
            rows_seen += batch_len
            for field in SOURCE_FIELDS:
                src_col = cols[field]
                tgt_col = cols[f"{field}_transliterated"]
                for src, tgt in zip(src_col, tgt_col, strict=False):
                    if not src or not tgt:
                        continue
                    gur = GURMUKHI_RUN.findall(src)
                    if not gur:
                        continue
                    lat = LATIN_RUN.findall(tgt)
                    if len(gur) != len(lat):
                        mismatched += 1
                        continue
                    aligned += 1
                    for g, latin in zip(gur, lat, strict=False):
                        eng = latin.lower()
                        if 0 < len(g) <= args.max_len and 0 < len(eng) <= args.max_len:
                            counts[g][eng] += 1
            bar.update(batch_len)

    # Collapse to the modal English per Gurmukhi word.
    ambiguous = 0
    pairs: list[tuple[str, str]] = []
    for g, eng_counts in counts.items():
        if len(eng_counts) > 1:
            ambiguous += 1
        best_eng = eng_counts.most_common(1)[0][0]
        pairs.append((g, best_eng))
    pairs.sort()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["punjabi", "english"])
        writer.writerows(pairs)

    gur_lens = sorted(len(p) for p, _ in pairs)
    eng_lens = sorted(len(e) for _, e in pairs)

    def pct(xs: list[int], q: float) -> int:
        return xs[min(len(xs) - 1, int(len(xs) * q))] if xs else 0

    print(f"\nRows scanned        : {rows_seen:,}")
    print(f"Aligned field cells : {aligned:,} (skipped {mismatched:,} mismatched)")
    print(
        f"Unique Gurmukhi words: {len(pairs):,}  -> {args.out}  "
        f"({ambiguous:,} had >1 candidate; kept modal)"
    )
    print(
        f"Gurmukhi word length: max {gur_lens[-1] if gur_lens else 0}, "
        f"p99 {pct(gur_lens, 0.99)}, p50 {pct(gur_lens, 0.50)}"
    )
    print(
        f"English  word length: max {eng_lens[-1] if eng_lens else 0}, "
        f"p99 {pct(eng_lens, 0.99)}, p50 {pct(eng_lens, 0.50)}"
    )


if __name__ == "__main__":
    main()
