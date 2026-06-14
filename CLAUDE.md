# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Indicate is a Python package for transliterating Indic text to English using PyTorch-based encoder-decoder models with attention. It ships local models for **Hindi** (Devanagari) and **Punjabi** (Gurmukhi), plus an LLM backend for other languages. The local models are custom-trained neural networks with pre-trained weights.

## Development Commands

### Testing
```bash
python -m unittest tests.test_010_hindi_translate
python -m unittest discover tests/
```

### Build and Installation
```bash
uv sync           # Install dependencies with uv (recommended)
uv build          # Build package
pip install -e .  # Install in development mode (alternative)
```

### CLI Usage
```bash
# Modern Click-based CLI
indicate hindi2english "राजशेखर चिंतालपति"
indicate hindi2english --input file.txt --output result.txt
indicate info
```

### Documentation (Sphinx)
```bash
cd docs/
make html         # Build HTML documentation
make clean        # Clean build artifacts
```

## Architecture

### Core Components

1. **Seq2SeqTransliterator** (`indicate/transliterator.py`) - Base lazy-loaded singleton holding the load + greedy `transliterate()` logic, parameterized by class attrs (vocab/weights paths, max lengths)
2. **HindiToEnglish** (`indicate/hindi2english.py`) / **PunjabiToEnglish** (`indicate/punjabi2english.py`) - Thin subclasses pointing at each language's tokenizers + safetensors
3. **Encoder** (`indicate/encoder.py`) - `nn.Module` LSTM encoder
4. **Decoder** (`indicate/decoder.py`) - `nn.Module` LSTM decoder with Luong (dot-product) attention
5. **Utils** (`indicate/utils.py`) - Tokenizer loading (`load_tokenizer`) and greedy decoding (`translate`)

### Model Architecture
- Encoder-decoder with Luong attention mechanism
- Embedding dimension: 256, LSTM units: 1024
- Per-language safetensors weights + tokenizer JSONs under
  `indicate/data/{hindi,punjabi}_to_english/`

### Training
- PyTorch training/extraction/eval scripts live in `training/` (see `training/README.md`)
- Hindi corpus `data/hindi.csv.gz` and Punjabi corpus `data/punjabi.csv.gz` are committed
  (Punjabi is extracted from the Dataverse-hosted parquet via `training/extract_punjabi.py`)
- Raw/large source data and the Dakshina benchmark live on Dataverse (see `data/README.md`)

### Data Pipeline
- Training data from ESPN Cricinfo, election affidavits, Google Dakshina dataset, and IIT Bombay corpus
- Character-level tokenization with special start (^) and end ($) tokens
- Decoding is hard-bounded by an input-adaptive step cap (no wall-clock timeout)

### Key Entry Points
- CLI: `indicate` command (Click group) with `hindi2english` / `punjabi2english` subcommands
- API: `indicate.hindi2english(text)` and `indicate.punjabi2english(text)` functions
- Main module: `indicate/__init__.py` exposes both transliteration functions

## Dependencies
- Python 3.13+ (modern Python with enhanced type hints)
- Click 8.0+ (modern CLI framework)
- PyTorch 2.6+ (core ML framework)
- safetensors 0.4+ (model weight serialization)
- func-timeout 4.3.0+ (prevents hanging translations)
- tqdm 4.60.0+ (progress bars)