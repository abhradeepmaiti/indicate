#!/usr/bin/env python
"""Head-to-head: indicate v2 vs IndicXlit on a language's blended eval, by subset.

Scores two prediction files (``source<TAB>pred``) against the blended
multi-reference eval, broken down by origin: Aksharantar-test, Dakshina-test, and
the held-out-own slice (the cleanest comparison — IndicXlit never trained on our
electoral/affidavit names). Exact-match top-1 accuracy.

    python training/compare.py --lang hi
    python training/compare.py --lang pa
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
DATA = REPO_ROOT / "data"

# Same insertion order build_v2.py uses (shared RNG), so held-out slices reproduce.
LANGS = {
    "hi": {"cfg": "hin", "ours": "hindi.csv.gz", "dak": "hi.translit.sampled.test.tsv"},
    "pa": {
        "cfg": "pan",
        "ours": "punjabi.csv.gz",
        "dak": "pa.translit.sampled.test.tsv",
    },
}


def col0(
    path: Path, gz: bool = False, delim: str = "\t", hdr: bool = False
) -> set[str]:
    op = gzip.open if gz else open
    out: set[str] = set()
    with op(path, "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delim)
        if hdr:
            next(reader, None)
        for row in reader:
            if row and row[0].strip():
                out.add(row[0].strip())
    return out


def read_refs(path: Path) -> dict[str, set[str]]:
    refs: dict[str, set[str]] = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2:
                refs[row[0]].add(row[1])
    return refs


def read_preds(path: Path) -> dict[str, str]:
    preds: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if row:
                preds[row[0]] = row[1] if len(row) > 1 else ""
    return preds


def reproduce_heldout(lang: str) -> set[str]:
    """Replay build_v2's shared RNG (in language order) to recover the slice."""
    rng = random.Random(42)
    held: set[str] = set()
    for la in LANGS:  # insertion order == build_v2 order
        srcs = sorted(col0(DATA / LANGS[la]["ours"], gz=True, delim=",", hdr=True))
        sample = rng.sample(srcs, min(2500, len(srcs)))
        if la == lang:
            held = set(sample)
            break
    return held


def acc(sources, refs, preds) -> tuple[float, int]:
    sources = [s for s in sources if s in refs and s in preds]
    if not sources:
        return 0.0, 0
    hit = sum(preds[s] in refs[s] for s in sources)
    return 100.0 * hit / len(sources), len(sources)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=sorted(LANGS), required=True)
    args = parser.parse_args()
    cfg = LANGS[args.lang]

    refs = read_refs(DATA / "eval" / f"{args.lang}_blended.tsv")
    v2 = read_preds(DATA / "eval" / f"{args.lang}_v2_pred.tsv")
    xlit = read_preds(DATA / "eval" / f"{args.lang}_indicxlit_pred.tsv")

    dakshina = col0(DATA / "dakshina" / cfg["dak"], delim="\t")
    aksh = col0(
        DATA / "aksharantar" / f"{cfg['cfg']}.test.csv.gz", gz=True, delim=",", hdr=True
    )
    heldout_only = reproduce_heldout(args.lang) - dakshina - aksh

    all_sources = set(refs)
    subsets = {
        "Overall (blended)": all_sources,
        "Aksharantar-test": all_sources & aksh,
        "Dakshina-test": all_sources & dakshina,
        "Held-out-own (clean)": all_sources & heldout_only,
    }

    print(
        f"[{args.lang}]  {'subset':22s} {'n':>6}   {'indicate v2':>12} {'IndicXlit':>12}"
    )
    print("-" * 64)
    for name, srcs in subsets.items():
        a2, n2 = acc(srcs, refs, v2)
        ax, _ = acc(srcs, refs, xlit)
        print(f"      {name:22s} {n2:>6}   {a2:>11.2f}% {ax:>11.2f}%")


if __name__ == "__main__":
    main()
