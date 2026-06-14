#!/usr/bin/env python
"""Score a predictions TSV against a blended multi-reference eval set.

Used to score external baselines (e.g. IndicXlit, from baseline_indicxlit.py) on
the exact same eval set and metrics as our models — exact-match ACC, Acc@<=1, CER.

Example:
    python training/score_preds.py --eval data/eval/hi_blended.tsv \
        --pred data/eval/hi_indicxlit_pred.tsv --label "IndicXlit hi"
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from training.metrics import format_summary, score_word, summarize  # noqa: E402


def read_refs(path: Path) -> dict[str, set[str]]:
    refs: dict[str, set[str]] = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) >= 2 and row[0] and row[1]:
                refs[row[0]].add(row[1])
    return refs


def read_preds(path: Path) -> dict[str, str]:
    preds: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if row:
                preds[row[0]] = row[1] if len(row) > 1 else ""
    return preds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()

    refs = read_refs(args.eval)
    preds = read_preds(args.pred)

    dists: list[int] = []
    ref_lens: list[int] = []
    missing = 0
    for src, ref_set in refs.items():
        if src not in preds:
            missing += 1
            continue
        dist, ref_len = score_word(preds[src], ref_set)
        dists.append(dist)
        ref_lens.append(ref_len)

    stats = summarize(dists, ref_lens)
    print(format_summary(stats, args.label))
    if missing:
        print(f"  (warning: {missing} eval sources had no prediction)")


if __name__ == "__main__":
    main()
