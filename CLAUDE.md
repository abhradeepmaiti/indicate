# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Indicate is a Python package for transliterating Hindi text to English using a TensorFlow-based encoder-decoder model with attention mechanism. The project uses a custom trained neural network model with pre-trained weights for transliteration.

## Development Commands

### Testing
```bash
python -m unittest indicate.tests.test_010_hindi_translate
python -m unittest discover indicate/tests/
```

### Build and Installation
```bash
pip install -e .  # Install in development mode
pip install .     # Regular installation
```

### Documentation (Sphinx)
```bash
cd docs/
make html         # Build HTML documentation
make clean        # Clean build artifacts
```

## Architecture

### Core Components

1. **HindiToEnglish** (`indicate/hindi2english.py:17`) - Main transliteration class with lazy-loaded TensorFlow model
2. **Encoder** (`indicate/encoder.py:3`) - LSTM-based encoder for processing Hindi input
3. **Decoder** (`indicate/decoder.py:7`) - LSTM-based decoder with attention mechanism for generating English output
4. **Utils** (`indicate/utils.py`) - Translation utilities and tokenization helpers

### Model Architecture
- Encoder-decoder with Luong attention mechanism
- Embedding dimension: 256, LSTM units: 1024
- Pre-trained weights stored in `indicate/data/model/hindi_to_english/saved_weights/`
- Tokenizers for Hindi and English stored as JSON files

### Data Pipeline
- Training data from ESPN Cricinfo, election affidavits, Google Dakshina dataset, and IIT Bombay corpus
- Character-level tokenization with special start (^) and end ($) tokens
- 10-second timeout per translation to prevent infinite loops

### Key Entry Points
- CLI: `hindi2english` command (defined in pyproject.toml)
- API: `indicate.transliterate.hindi2english(text)` function
- Main module: `indicate/__init__.py` exposes the transliteration function

## Dependencies
- TensorFlow 2.18.0 (core ML framework)
- func-timeout (prevents hanging translations)
- tqdm (progress bars)