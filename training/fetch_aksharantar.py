#!/usr/bin/env python
"""Download AI4Bharat Aksharantar (Hindi + Punjabi) and cache as (native, english) CSVs.

Aksharantar (https://huggingface.co/datasets/ai4bharat/Aksharantar, CC-BY/CC0) ships
one ``<lang>.zip`` per language, each containing ``<lang>_{train,valid,test}.json``
(JSONL with ``native word`` / ``english word`` / ``source`` / ``score``).

This writes gzipped two-column CSVs (``source,english``) per split to
``data/aksharantar/`` — large/external, gitignored. Used by ``build_v2.py``.

Example:
    python training/fetch_aksharantar.py
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "aksharantar"
LANGS = {"hindi": "hin", "punjabi": "pan"}
SPLITS = {"train": "train", "val": "valid", "test": "test"}


def write_split(zf: zipfile.ZipFile, member: str, out_path: Path) -> int:
    n = 0
    with (
        zf.open(member) as raw,
        gzip.open(out_path, "wt", encoding="utf-8", newline="") as out,
    ):
        writer = csv.writer(out)
        writer.writerow(["source", "english"])
        for line in io.TextIOWrapper(raw, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            native = (row.get("native word") or "").strip()
            english = (row.get("english word") or "").strip()
            if native and english:
                writer.writerow([native, english])
                n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--langs", nargs="+", default=list(LANGS), choices=list(LANGS))
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for lang in args.langs:
        cfg = LANGS[lang]
        print(f"Downloading Aksharantar [{cfg}] ...")
        zip_path = hf_hub_download(
            "ai4bharat/Aksharantar", f"{cfg}.zip", repo_type="dataset"
        )
        with zipfile.ZipFile(zip_path) as zf:
            members = {m.rsplit("/", 1)[-1]: m for m in zf.namelist()}
            for short, tag in SPLITS.items():
                member = members.get(f"{cfg}_{tag}.json")
                if not member:
                    print(f"  ({cfg}_{tag}.json missing)")
                    continue
                out_path = OUT_DIR / f"{cfg}.{short}.csv.gz"
                count = write_split(zf, member, out_path)
                print(f"  {short:5s} {count:>9,} -> {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
