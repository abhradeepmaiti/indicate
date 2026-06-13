# Training the transliteration models

This directory holds the PyTorch training, extraction, and evaluation scripts for
the encoder-decoder (LSTM + Luong attention) transliteration models shipped in
`indicate/` — currently **Hindi** (Devanagari) and **Punjabi** (Gurmukhi), both
→ English.

The same `train.py` trains either model; `--model-dir` selects where the
tokenizers and `saved_weights/` live.

## Hindi

Training uses the committed parallel corpus `data/hindi.csv.gz` (371k Hindi/English
character pairs from Indian election affidavits, ESPN Cricinfo, the Google
Dakshina dataset, and the IIT Bombay corpus). **No download is required** — the
merged corpus is already in the repo. By default the existing tokenizer JSONs are
reused (so vocabulary indices stay stable); pass `--rebuild-vocab` to refit.

```bash
# Full run (auto-selects cuda > mps > cpu)
python training/train.py --epochs 25 --batch-size 64

# Quick smoke test on a subset
python training/train.py --limit 5000 --epochs 1
```

## Punjabi

The Punjabi pairs are distilled from GPT-4o. The raw source
`data/punjab_transliteration_subset.parquet` (Punjab electoral rolls, ~19M rows,
**not committed**) has each Gurmukhi run transliterated in place, so aligned word
pairs can be recovered:

```bash
# 1. Extract unique (gurmukhi -> english) word pairs -> data/punjabi.csv.gz (committed)
python training/extract_punjabi.py

# 2. Train (own Gurmukhi + English vocabularies)
python training/train.py \
    --data data/punjabi.csv.gz \
    --model-dir indicate/data/punjabi_to_english \
    --input-vocab-name punjabi_tokens.json --target-vocab-name english_tokens.json \
    --max-input 32 --max-output 32 --rebuild-vocab \
    --epochs 25
```

Best weights (lowest validation loss) are written to
`<model-dir>/saved_weights/{encoder,decoder}.safetensors`.

## Evaluate

The canonical benchmark uses the **Google Dakshina** romanization lexicons. The
2 GB dataset is not committed; download it once
(https://github.com/google-research-datasets/dakshina, release tarball
`dakshina_dataset_v1.0.tar`) and place the per-language test splits under
`data/dakshina/`:

- Hindi:   `dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv`
- Punjabi: `dakshina_dataset_v1.0/pa/lexicons/pa.translit.sampled.test.tsv`

```bash
python training/eval.py --model hindi      # data/dakshina/hi.translit.sampled.test.tsv
python training/eval.py --model punjabi    # data/dakshina/pa.translit.sampled.test.tsv
```

A native word counts as correct if the prediction matches any reference
romanization (the test set lists multiple valid spellings per word).

| Model | Dakshina exact-match | Notes |
|-------|----------------------|-------|
| Hindi (PyTorch) | **75.80%** (1895/2500) | beats the old TensorFlow model's 73.64% |
| Punjabi (PyTorch) | _see `eval.py --model punjabi`_ | trained on electoral-roll names; Dakshina pa is general vocabulary, so expect a domain gap |

`oos_eval.py` additionally reports held-out accuracy on the training corpus
itself (`--model {hindi,punjabi}`), reproducing the exact train/val split.
