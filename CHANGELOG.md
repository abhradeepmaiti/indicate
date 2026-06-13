# Changelog

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
