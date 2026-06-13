# Data

This directory holds **training & source data only** — it is excluded from the
installable package (`tool.uv.build-backend.source-exclude` drops `/data`), so
nothing here ships in the PyPI wheel. The runtime model artifacts (tokenizers +
safetensors) live under `indicate/data/` instead.

## What lives where

**In the repo** (small, derived, reproducible):
- `hindi.csv.gz` — Hindi→English training corpus, 371k char-level pairs (gzipped)
- `punjabi.csv.gz` — Punjabi→English training corpus, 287k pairs (gzipped)
- All collection/processing code: `get_affidavits/`, `get_espncricinfo/`,
  `railway_stations/`, `wikipedia_interwiki/`, and the notebooks
  (`notebooks/Data_Preparation.ipynb`, `examples/punjab_transliteration.ipynb`)

**On Dataverse** (source data, for reproducibility):
- **Hindi source** (public) — `affidavits.csv`, `players_with_hindi_names.json`,
  `en-hi.mined-pairs` (IIT Bombay, CC-BY-NC). New deposit DOI: `TODO`.
- **Punjab electoral rolls** (restricted: form + IRB) — *super-upstream* raw rolls.
  *Parsed Indian Electoral Rolls*, Sood et al., https://doi.org/10.7910/DVN/MUEGDT.
  File `punjab_all_clean+t13n.csv.gz` is the raw input to the LLM annotation.
- **Punjab LLM transliterations** (restricted) — `punjab_transliteration_subset.parquet`,
  the GPT-4o-annotated output (the direct source for `punjabi.csv.gz`). New
  restricted deposit DOI: `TODO`. Gitignored locally (201 MB, PII).

**Download from origin** (third-party, not redistributed):
- Google Dakshina benchmark → `data/dakshina/` (see `training/README.md`).

## Pipelines (source → repo → model)

**Hindi**
```
[Dataverse: affidavits.csv, players_with_hindi_names.json, en-hi.mined-pairs]
  + Google Dakshina hi lexicon (download)
    → notebooks/Data_Preparation.ipynb        (merge, dedupe, clean)
    → data/hindi.csv.gz                        (committed)
    → training/train.py                        → indicate/data/hindi_to_english/
```

**Punjabi**
```
Parsed Indian Electoral Rolls (10.7910/DVN/MUEGDT, restricted)   [super-upstream raw]
  → examples/punjab_transliteration.ipynb     (extract Gurmukhi words, GPT-4o annotate)
  → punjab_transliteration_subset.parquet      [deposited: new restricted Dataverse]
  → training/extract_punjabi.py                (align word pairs, modal-dedupe)
  → data/punjabi.csv.gz                        (committed)
  → training/train.py                          → indicate/data/punjabi_to_english/
```

## Reproduce / download

```bash
# Train directly from the committed corpora (no download needed):
python training/train.py                                   # Hindi
python training/train.py --data data/punjabi.csv.gz \
    --model-dir indicate/data/punjabi_to_english --rebuild-vocab \
    --input-vocab-name punjabi_tokens.json --max-input 32 --max-output 32

# Re-derive a corpus from source: fetch the Dataverse files above, then run the
# corresponding notebook / training/extract_punjabi.py.
```
