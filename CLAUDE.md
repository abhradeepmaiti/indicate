# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Indicate is a Python package for transliterating Hindi text to English using a TensorFlow-based encoder-decoder model with attention mechanism. The project uses a custom trained neural network model with pre-trained weights for transliteration.

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

# Legacy CLI (backward compatibility)
hindi2english --type hin2eng --input "हिंदी"
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
- Pre-trained weights stored in `data/model/hindi_to_english/saved_weights/`
- Tokenizers for Hindi and English stored as JSON files

### Data Pipeline
- Training data from ESPN Cricinfo, election affidavits, Google Dakshina dataset, and IIT Bombay corpus
- Character-level tokenization with special start (^) and end ($) tokens
- 10-second timeout per translation to prevent infinite loops

### Key Entry Points
- Modern CLI: `indicate` command with Click-based interface (defined in pyproject.toml)
- Legacy CLI: `hindi2english` command for backward compatibility
- API: `indicate.transliterate.hindi2english(text)` function
- Main module: `indicate/__init__.py` exposes the transliteration function

## Dependencies
- Click 8.0+ (modern CLI framework)
- TensorFlow 2.13.0-2.16.0 (core ML framework)
- func-timeout 4.3.0+ (prevents hanging translations)
- tqdm 4.60.0+ (progress bars)