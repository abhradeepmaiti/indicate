# Examples

This directory contains practical examples for using the `indicate` package.

## Files

### 📝 `basic_llm_usage.py`
Demonstrates the core LLM transliteration features:
- Basic Hindi-English transliteration
- Multi-language support (Tamil, Telugu, Bengali, etc.)
- Automatic language detection
- Batch processing
- Custom provider/model selection
- Error handling

**Run it:**
```bash
# Set your API key first
export OPENAI_API_KEY=your-key

# Run the example
python examples/basic_llm_usage.py
```

### 🗂️ `file_processing.py`
Shows production-ready file handling:
- Safe file path validation
- Automatic backup creation
- Structured JSON output
- Round-trip processing (JSON as input)
- Error recovery and reporting
- Atomic file operations

**Run it:**
```bash
python examples/file_processing.py
```

## Quick Examples

### Simple CLI Usage

```bash
# Basic transliteration
indicate llm "राजशेखर चिंतालपति"
# Output: Rajashekar Chintalapati

# Tamil to English  
indicate llm "முருகன்" --source tamil --target english
# Output: Murugan

# Safe batch processing with JSON output
indicate llm --input names.txt --output results.json --format json --batch --backup
```

### Python API Usage

```python
from indicate import IndicLLMTransliterator

# Initialize for any language pair
trans = IndicLLMTransliterator('hindi', 'english')

# Single transliteration
result = trans.transliterate('नमस्ते')
print(result)  # Namaste

# Batch processing
texts = ['राजेश', 'गौरव', 'प्रिया']
results = trans.transliterate_batch(texts)
print(results)  # ['Rajesh', 'Gaurav', 'Priya']
```

## Prerequisites

### For LLM Examples
Set one of these API keys:
```bash
export OPENAI_API_KEY=your-openai-key
export ANTHROPIC_API_KEY=your-anthropic-key  
export GOOGLE_API_KEY=your-google-key
```

### For TensorFlow Examples
No setup needed - uses pre-trained models.

## Supported Language Pairs

### LLM Backend (Full Support)
- **Hindi** ↔ English
- **Tamil** ↔ English  
- **Telugu** ↔ English
- **Bengali** ↔ English
- **Gujarati** ↔ English
- **Kannada** ↔ English
- **Malayalam** ↔ English
- **Punjabi** ↔ English
- **Marathi** ↔ English
- **Odia** ↔ English
- **Urdu** ↔ English
- **Sanskrit** ↔ English
- **Inter-Indic**: Any Indic language to any other Indic language

### TensorFlow Backend
- **Hindi** → English only

## File Formats

### Input
- **Text files**: Plain UTF-8 text, one item per line
- **JSON files**: Structured format from previous results

### Output  
- **Text format**: Plain transliterated text
- **JSON format**: Rich metadata with error handling

```json
{
  "metadata": {
    "source_language": "hindi",
    "target_language": "english",
    "total_lines": 3,
    "successful_lines": 3,
    "failed_lines": 0
  },
  "results": [
    {
      "line_number": 1,
      "input_text": "राजेश",
      "output_text": "Rajesh",
      "processing_time": 1.2,
      "confidence": "high"
    }
  ]
}
```

## Safety Features

All examples demonstrate:
- ✅ **Path validation** - prevents accidental file overwrites
- ✅ **Atomic writing** - no partial/corrupted files
- ✅ **Automatic backups** - protects existing data
- ✅ **Error recovery** - handles API failures gracefully
- ✅ **Progress tracking** - resume interrupted operations