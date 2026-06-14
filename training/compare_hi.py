#!/usr/bin/env python
"""Head-to-head: indicate v2 vs IndicXlit on the Hindi blended eval, by subset.

Scores two prediction files (``source<TAB>pred``) against the blended
multi-reference eval, broken down by origin: Aksharantar-test, Dakshina-test, and
the held-out-own slice (the cleanest comparison — IndicXlit never trained on our
electoral/affidavit names). Exact-match top-1 accuracy.

    python training/compare_hi.py
"""

from __future__ import annotations

import csv
import gzip
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
DATA = REPO_ROOT / "data"


def read_col0(
    path: Path, gz: bool = False, tsv: bool = True, skip_header: bool = False
):
    op = gzip.open if gz else open
    out = set()
    with op(path, "rt", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t" if tsv else ",")
        if skip_header:
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
    preds = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if row:
                preds[row[0]] = row[1] if len(row) > 1 else ""
    return preds


def acc(sources, refs, preds) -> tuple[float, int]:
    sources = [s for s in sources if s in refs and s in preds]
    if not sources:
        return 0.0, 0
    hit = sum(preds[s] in refs[s] for s in sources)
    return 100.0 * hit / len(sources), len(sources)


def main() -> None:
    refs = read_refs(DATA / "eval" / "hi_blended.tsv")
    v2 = read_preds(DATA / "eval" / "hi_v2_pred.tsv")
    xlit = read_preds(DATA / "eval" / "hi_indicxlit_pred.tsv")

    dakshina = read_col0(DATA / "dakshina" / "hi.translit.sampled.test.tsv")
    aksh = read_col0(
        DATA / "aksharantar" / "hin.test.csv.gz", gz=True, tsv=False, skip_header=True
    )
    our_sources = sorted(
        read_col0(DATA / "hindi.csv.gz", gz=True, tsv=False, skip_header=True)
    )
    heldout = set(random.Random(42).sample(our_sources, 2500))
    heldout_only = heldout - dakshina - aksh

    all_sources = set(refs)
    subsets = {
        "Overall (blended)": all_sources,
        "Aksharantar-test": all_sources & aksh,
        "Dakshina-test": all_sources & dakshina,
        "Held-out-own (clean)": all_sources & heldout_only,
    }

    print(f"{'subset':24s} {'n':>6}   {'indicate v2':>12} {'IndicXlit':>12}")
    print("-" * 60)
    for name, srcs in subsets.items():
        a2, n2 = acc(srcs, refs, v2)
        ax, _ = acc(srcs, refs, xlit)
        print(f"{name:24s} {n2:>6}   {a2:>11.2f}% {ax:>11.2f}%")


if __name__ == "__main__":
    main()
