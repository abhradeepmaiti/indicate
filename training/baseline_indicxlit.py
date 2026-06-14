#!/usr/bin/env python
"""Generate AI4Bharat IndicXlit (Indic->Roman) predictions on a blended eval set.

Runs in an ISOLATED environment (IndicXlit needs fairseq + an old TF/keras/torch
stack that conflicts with this package's Python 3.13 / torch 2.x). Reads the
``source`` column of an eval TSV and writes ``source<TAB>prediction`` (top-1).
Multi-word sources are transliterated word-by-word and re-joined, mirroring how
``indicate`` handles them. Score the output with ``training/score_preds.py``.

Run (ephemeral env):
    uv run --python 3.10 --no-project \
      --with ai4bharat-transliteration --with "gevent==24.11.1" \
      --with "tensorflow==2.15.1" --with "keras==2.15.0" \
      --with "tensorflow-addons==0.23.0" --with "torch==2.2.2" \
      python training/baseline_indicxlit.py --eval data/eval/hi_blended.tsv \
      --lang hi --out data/eval/hi_indicxlit_pred.tsv
"""

from __future__ import annotations

import argparse
import csv

from ai4bharat.transliteration import XlitEngine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval", required=True, help="blended eval TSV (source<TAB>ref)"
    )
    parser.add_argument("--lang", required=True, help="language code, e.g. hi / pa")
    parser.add_argument("--out", required=True, help="output TSV (source<TAB>pred)")
    parser.add_argument("--beam", type=int, default=4)
    args = parser.parse_args()

    sources: list[str] = []
    seen: set[str] = set()
    with open(args.eval, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t"):
            if row and row[0] and row[0] not in seen:
                seen.add(row[0])
                sources.append(row[0])

    engine = XlitEngine(src_script_type="indic", beam_width=args.beam)

    def translit(word: str) -> str:
        try:
            out = engine.translit_word(word, lang_code=args.lang, topk=1)
            return out[0] if out else word
        except Exception:
            return word

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        for i, src in enumerate(sources):
            # Word-by-word for multi-token sources (matches indicate's handling).
            pred = " ".join(translit(w) for w in src.split(" ") if w)
            writer.writerow([src, pred])
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{len(sources)}", flush=True)

    print(f"wrote {len(sources)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
