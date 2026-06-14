#!/usr/bin/env python
"""Build v2 training corpora + a blended multi-reference eval set per language.

For Hindi (hi/hin) and Punjabi (pa/pan):

  * **Blended eval** (`data/eval/<lang>_blended.tsv`) — union, by source string, of
    (a) Dakshina test, (b) Aksharantar test, and (c) a held-out slice of our own
    corpus. Multiple romanizations per source are all kept (multi-reference).

  * **Training corpus** (`data/<lang>_train_v2.csv.gz`) — our corpus + Aksharantar
    train + val, deduped on exact (source, english), with every pair whose *source*
    appears in the blended eval removed (leakage filter — Aksharantar overlaps
    Dakshina, so this is mandatory for a clean benchmark).

Both outputs are gitignored. Run after `fetch_aksharantar.py`.

Example:
    python training/build_v2.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.train import load_pairs  # noqa: E402

DATA = REPO_ROOT / "data"
AKS = DATA / "aksharantar"
DAK = DATA / "dakshina"
EVAL_DIR = DATA / "eval"

LANGS = {
    "hi": {
        "cfg": "hin",
        "ours": DATA / "hindi.csv.gz",
        "dakshina": DAK / "hi.translit.sampled.test.tsv",
    },
    "pa": {
        "cfg": "pan",
        "ours": DATA / "punjabi.csv.gz",
        "dakshina": DAK / "pa.translit.sampled.test.tsv",
    },
}


def read_dakshina(path: Path) -> dict[str, set[str]]:
    refs: dict[str, set[str]] = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                refs[row[0].strip()].add(row[1].strip())
    return refs


def build_lang(lang: str, cfg: dict, heldout_n: int, rng: random.Random) -> None:
    c = cfg["cfg"]
    ours = load_pairs(cfg["ours"])
    aks_train = load_pairs(AKS / f"{c}.train.csv.gz")
    aks_val = load_pairs(AKS / f"{c}.val.csv.gz")
    aks_test = load_pairs(AKS / f"{c}.test.csv.gz")
    dak_test = read_dakshina(cfg["dakshina"])

    # Held-out slice of our own corpus: sample unique source strings, keep all refs.
    our_sources = sorted({s for s, _ in ours})
    k = min(heldout_n, len(our_sources))
    heldout = set(rng.sample(our_sources, k))
    our_heldout_refs: dict[str, set[str]] = defaultdict(set)
    for s, e in ours:
        if s in heldout:
            our_heldout_refs[s].add(e)

    # Blended eval = Dakshina test ∪ Aksharantar test ∪ held-out-own (multi-reference).
    eval_refs: dict[str, set[str]] = defaultdict(set)
    for s, rs in dak_test.items():
        eval_refs[s] |= rs
    for s, e in aks_test:
        eval_refs[s].add(e)
    for s, rs in our_heldout_refs.items():
        eval_refs[s] |= rs
    eval_sources = set(eval_refs)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    eval_path = EVAL_DIR / f"{lang}_blended.tsv"
    with open(eval_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        for s in sorted(eval_sources):
            for r in sorted(eval_refs[s]):
                w.writerow([s, r])

    # Training corpus: ours + Aksharantar train + val, dedupe, drop eval sources.
    seen: set[tuple[str, str]] = set()
    train_pairs: list[tuple[str, str]] = []
    removed = 0
    for s, e in [*ours, *aks_train, *aks_val]:
        if s in eval_sources:
            removed += 1
            continue
        key = (s, e)
        if key in seen:
            continue
        seen.add(key)
        train_pairs.append((s, e))

    train_path = DATA / f"{lang}_train_v2.csv.gz"
    with gzip.open(train_path, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "english"])
        w.writerows(train_pairs)

    leak = {s for s, _ in train_pairs} & eval_sources
    assert not leak, f"LEAKAGE: {len(leak)} eval sources remain in {lang} train"

    aks_test_words = len({s for s, _ in aks_test})
    print(
        f"[{lang}] ours={len(ours):,}  aks_train={len(aks_train):,}  aks_val={len(aks_val):,}"
    )
    print(
        f"  blended eval sources: {len(eval_sources):,} "
        f"(dakshina {len(dak_test):,} + aks_test {aks_test_words:,} + heldout {k:,})  "
        f"-> {eval_path.relative_to(REPO_ROOT)}"
    )
    print(
        f"  train pairs: {len(train_pairs):,} (removed {removed:,} leakage/dupe)  "
        f"-> {train_path.relative_to(REPO_ROOT)}"
    )
    print(f"  leakage check: {len(leak)} (must be 0) ✓")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--heldout", type=int, default=2500, help="own-corpus held-out source strings"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    for lang, cfg in LANGS.items():
        build_lang(lang, cfg, args.heldout, rng)


if __name__ == "__main__":
    main()
