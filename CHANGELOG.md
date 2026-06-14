# Changelog

## 0.7.0

### Added
- **Data-scaled v2 models** — both Hindi and Punjabi retrained on the public
  **Aksharantar** corpus (AI4Bharat) merged with our own data: Hindi ~1.5M pairs,
  Punjabi ~773k. A strict **leakage filter** drops every eval word from training.
- **Batched inference** — `transliterate_batch(list[str])` is the decode engine
  (batched greedy + beam); `transliterate(str)` is a thin wrapper. The CLI
  `--batch` mode and bulk eval use it (≫ faster than one-at-a-time).
- **Top-k / n-best** — `transliterate(text, n=k)` returns up to `k` ranked
  candidates (a `list[str]`); `n=1` (default) still returns a `str`.

### Changed
- **Model weights hosted on Hugging Face** (`soodoku/indicate`, tag `v0.7.0`) and
  lazy-downloaded (cached) on first use, so the wheel is ~50 KB instead of ~107 MB
  (PyPI's 100 MB limit). Tokenizer JSONs still ship in the package; a local copy of
  the weights, if present, is used in preference to the download. Weights are fp32
  (`safetensors`) — fp16 was tried and rejected (it perturbed ~0.05% of outputs for
  no size benefit worth keeping, since hosting solves the size problem).
- **Input-adaptive decode cap** (`min(max_length_output, 2·len(input)+8)`).
- Removed `func_timeout` (the decoder is hard-bounded, so the wall-clock guard
  was dead weight — and a perf/flakiness drag); dropped the dependency.
- Coverage reports to the terminal only (no `htmlcov`/`coverage.xml`).

### Accuracy (Dakshina test, exact-match; vs AI4Bharat IndicXlit, same direction/data)
| Model | Dakshina (gold) | Held-out-own names | 
|-------|-----------------|--------------------|
| Hindi v2 | **74.4%** (IndicXlit 73.2%) | **52.8%** (IndicXlit 49.7%) |
| Punjabi v2 | 71.9% (IndicXlit 73.2%) | **56.9%** (IndicXlit 53.5%) |

v2 matches/edges SOTA IndicXlit on the gold benchmark and **beats it on the
deployment domain** (names IndicXlit never trained on). Eval is leakage-filtered;
see `training/IMPROVEMENTS.md` and `training/compare.py`.

## 0.6.0

### Added
- **Punjabi (Gurmukhi → English) transliteration model** — new `PunjabiToEnglish`
  class, `indicate.punjabi2english(...)` API, and `indicate punjabi2english` CLI
  command. Distilled from GPT-4o-labelled Punjab data
  (`training/extract_punjabi.py` → `data/punjabi.csv.gz`).
- **Beam-search decoding** (length-normalized, width 5) is now the default for
  both models — +1.5 pts exact-match on the Hindi Dakshina test set vs greedy.
- Evaluation tooling: `training/eval.py` (gold Dakshina), `training/oos_eval.py`
  (held-out), and `training/metrics.py` reporting exact-match accuracy, CER, and
  Acc@≤1 with edit-distance histograms.

### Changed
- **Migrated the modeling stack from TensorFlow to PyTorch** and the package to
  **Python 3.13-only**. Weights now ship as `safetensors`.
- Refactored the per-language logic into a shared `Seq2SeqTransliterator` base.

### Accuracy (Google Dakshina test, exact-match / Acc@≤1 / CER)
- Hindi → English: **77.32% / 91.16% / 5.65%**
- Punjabi → English: **71.24% / 91.56% / 6.42%**

### Notes
- Char-LM beam re-ranking and attention padding masks are implemented but gated
  off (no measured win for the name-heavy, native→Latin case); see
  `training/IMPROVEMENTS.md`. The next milestone is a data-scale retrain on the
  public Aksharantar corpus.
