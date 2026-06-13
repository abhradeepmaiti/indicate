# Indicate: Transliterate Indic Languages with PyTorch and LLMs

[![Notary Badge](https://notarypy.soodoku.workers.dev/badge/indicate/0.2.1/indicate-0.2.1-py3-none-any.whl)](https://pypi.org/integrity/indicate/0.2.1/indicate-0.2.1-py3-none-any.whl/provenance)
[![PyPI Version](https://img.shields.io/pypi/v/indicate.svg)](https://pypi.python.org/pypi/indicate)
[![Downloads](https://static.pepy.tech/badge/indicate)](https://pepy.tech/project/indicate)
[![Tests](https://github.com/in-rolls/indicate/workflows/test/badge.svg)](https://github.com/in-rolls/indicate/actions?query=workflow%3Atest)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://in-rolls.github.io/indicate/)

**Indicate** provides high-quality transliteration between Indic languages and English using both a traditional PyTorch model and state-of-the-art LLMs (Large Language Models).

## 🚀 Features

- **🧠 Dual Backend Support**: Choose between the PyTorch model or LLM-based transliteration
- **🌍 Multi-Language**: 12+ Indic languages (Hindi, Tamil, Telugu, Bengali, etc.)
- **🔄 Bidirectional**: Supports both Indic→English and English→Indic transliteration
- **🛡️ Production Ready**: Safe file handling, atomic writes, backup support
- **📊 Structured Output**: Rich JSON format with metadata and error handling
- **⚡ Batch Processing**: Efficient processing of large files with progress tracking

## 🎯 Supported Languages

Hindi • Tamil • Telugu • Bengali • Gujarati • Kannada • Malayalam • Punjabi • Marathi • Odia • Urdu • Sanskrit ↔ English

## Install

We strongly recommend installing `indicate` inside a Python virtual environment (see [venv documentation](https://docs.python.org/3/library/venv.html#creating-virtual-environments))

**Requirements:** Python 3.13+

```bash
pip install indicate
```

## 🔧 Quick Setup

### For LLM-based transliteration (recommended):
```bash
pip install indicate

# Set your API key (choose one):
export OPENAI_API_KEY=your-key
export ANTHROPIC_API_KEY=your-key  
export GOOGLE_API_KEY=your-key
```

### For the local model (no API key):
```bash
pip install indicate
# No API key needed - uses the bundled pre-trained PyTorch model
```

## 🎯 Usage

### 🧠 LLM-Based Transliteration (New!)

The LLM backend provides higher accuracy and supports all Indic languages:

```bash
# Simple transliteration (auto-detects Hindi)
indicate llm "राजशेखर चिंतालपति"
# Output: Rajashekar Chintalapati

# Specify languages explicitly  
indicate llm "முருகன்" --source tamil --target english
# Output: Murugan

# Between Indic languages
indicate llm "नमस्ते" --source hindi --target tamil  
# Output: நமஸ்தே

# Safe batch processing with structured JSON output
indicate llm --input names.txt --output results.json --format json --batch --backup

# Dry run to preview changes
indicate llm --input large_file.txt --dry-run
```

**Python API:**
```python
from indicate import IndicLLMTransliterator

# Initialize for any language pair
transliterator = IndicLLMTransliterator('hindi', 'english')
result = transliterator.transliterate('राजशेखर चिंतालपति')
print(result)  # Output: Rajashekar Chintalapati

# Batch processing
texts = ["राजेश", "गौरव", "प्रिया"]
results = transliterator.transliterate_batch(texts)
print(results)  # ['Rajesh', 'Gaurav', 'Priya']
```

### 🤖 PyTorch Backend (Traditional)

Local offline models are available for **Hindi** (Devanagari) and **Punjabi**
(Gurmukhi):

```bash
# Hindi to English using the local PyTorch model
indicate hindi2english "राजशेखर चिंतालपति"
# Output: rajshekhar chintapalati

# Punjabi (Gurmukhi) to English
indicate punjabi2english "ਰਵਿ ਸ਼ਰਮਾ"
# Output: ravi sharma

# From file
indicate hindi2english --input hindi.txt --output english.txt

# Batch processing
indicate hindi2english --input large_file.txt --batch
```

**Python API:**
```python
from indicate import hindi2english, punjabi2english
print(hindi2english("हिंदी"))      # hindi
print(punjabi2english("ਰਵਿ"))      # ravi
```

## 📊 JSON Output Format

The LLM backend provides rich, structured output perfect for data processing:

```json
{
  "metadata": {
    "source_language": "hindi",
    "target_language": "english", 
    "timestamp": "2024-12-09T12:00:00Z",
    "total_lines": 3,
    "successful_lines": 3,
    "failed_lines": 0,
    "encoding": "utf-8"
  },
  "results": [
    {
      "line_number": 1,
      "input_text": "राजेश कुमार",
      "output_text": "Rajesh Kumar", 
      "source_lang": "hindi",
      "target_lang": "english",
      "confidence": "high",
      "processing_time": 1.2,
      "timestamp": "2024-12-09T12:00:01Z"
    }
  ]
}
```

## 🛡️ Safety Features

- **🔒 Input/Output Validation**: Prevents accidental file overwrites
- **⚛️ Atomic Writing**: Safe file operations using temporary files
- **💾 Automatic Backups**: Optional timestamped backups of existing files
- **🔄 Resume Support**: Resume interrupted batch operations
- **👁️ Dry Run Mode**: Preview operations before execution

## 🎛️ Advanced Usage

```bash
# Show few-shot examples being used
indicate llm --show-examples --source bengali --target english

# Resume interrupted batch job
indicate llm --input large_file.txt --output results.txt --resume

# Use specific LLM provider/model
indicate llm "text" --provider anthropic --model claude-3-opus

# Process JSON from previous results
indicate llm --input results.json --source english --target hindi
```

## 🔄 Backend Comparison

| Feature | PyTorch Backend | LLM Backend |
|---------|------------------|-------------|
| **Languages** | Hindi ↔ English only | 12+ Indic languages ↔ English + Inter-Indic |
| **Setup** | No API key needed | Requires LLM API key |
| **Speed** | Very fast (local) | Moderate (API calls) |
| **Accuracy** | Good for common words | Excellent for all types |
| **Cost** | Free | Pay per API call |
| **Offline** | ✅ Works offline | ❌ Requires internet |
| **Batch Processing** | ✅ | ✅ with safety features |

## 🧪 Testing Locally

1. **Clone and install**:
   ```bash
   git clone https://github.com/in-rolls/indicate.git
   cd indicate
   uv sync  # or pip install -e .
   ```

2. **Run tests**:
   ```bash
   # All tests
   python -m pytest
   
   # Specific tests
   python -m pytest tests/test_llm_indic.py
   python -m pytest tests/test_file_safety.py
   ```

3. **Test both backends**:
   ```bash
   # PyTorch backend
   indicate hindi2english "हिंदी"
   
   # LLM backend (set API key first)
   export OPENAI_API_KEY=your-key
   indicate llm "हिंदी"
   ```

## Data

The datasets used to train the model:

- [Indian Election affidavits](https://affidavit.eci.gov.in/CandidateCustomFilter)
- [Google Dakshina dataset](https://github.com/google-research-datasets/dakshina)
- [ESPN Cric Info](https://www.espncricinfo.com/hindi/series/pakistan-tour-of-england-2021-1239529/england-vs-pakistan-1st-odi-1239537/full-scorecard) for hindi version of the [english scorecard](https://www.espncricinfo.com/series/pakistan-tour-of-england-2021-1239529/england-vs-pakistan-1st-odi-1239537/full-scorecard)
- [IIT Bombay English-Hindi Corpus](https://www.cfilt.iitb.ac.in/iitb_parallel/)

## Evaluation

Both PyTorch models were evaluated on the Google Dakshina test sets (2,500 unique
words each; a word counts as correct if the prediction matches any reference
romanization). Beam search (the shipped default) is reported alongside greedy:

| Model | Exact-match (greedy → beam) | Acc@≤1 | CER |
|-------|-----------------------------|--------|-----|
| Hindi → English | 75.80% → **77.32%** | 91.16% | 5.65% |
| Punjabi → English | 70.96% → **71.24%** | 91.56% | 6.42% |

The earlier TensorFlow Hindi model scored 73.64%;
[Indic-trans](https://github.com/libindic/indic-trans) scored 63.12% on the same
Hindi set. Primary metric is Top-1 exact-match accuracy; CER (character error
rate) is the soft companion that credits near-misses.

Below is the edit-distance distribution on the test set (0 = exact match):

![Edit distance metrics of model on Google Dakshina test dataset](https://github.com/in-rolls/indicate/raw/master/images/h2e_ed.png)

## Authors

Rajashekar Chintalapati and Gaurav Sood

## Contributor Code of Conduct

The project welcomes contributions from everyone! In fact, it depends on it. To maintain this welcoming atmosphere, and to collaborate in a fun and productive way, we expect contributors to the project to abide by the [Contributor Code of Conduct](http://contributor-covenant.org/version/1/0/0/).

## License

The package is released under the [MIT License](https://opensource.org/licenses/MIT).