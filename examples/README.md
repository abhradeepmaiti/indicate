# Indicate Examples

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

### 📊 `pandas_usage.py`
**DataFrame processing with pandas** (reference implementation):
- Process entire DataFrame columns
- Batch processing for efficiency
- Multiple column transliteration
- Error handling and progress tracking
- Optimal settings for large datasets

**Note**: This is a reference implementation. Copy and adapt the `transliterate_dataframe()` function to your needs.

**Run it:**
```bash
# Install dependencies
pip install pandas tqdm

# Set API key and run
export OPENAI_API_KEY=your-key
python examples/pandas_usage.py
```

### 💾 `large_dataset_with_checkpoints.py`
**Processing large datasets with reliability features**:
- Automatic checkpointing to Parquet files
- Resume from interruptions
- Progress saving and recovery
- Emergency saves on Ctrl+C
- Parallel processing strategies

Perfect for datasets with hundreds of thousands of entries where processing might take hours.

**Run it:**
```bash
# Install dependencies
pip install pandas tqdm pyarrow

# Process large dataset
python examples/large_dataset_with_checkpoints.py
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

### DataFrame Processing (from pandas_usage.py)

```python
import pandas as pd
# Copy the transliterate_dataframe function from pandas_usage.py

df = pd.read_csv('your_data.csv')
result = transliterate_dataframe(
    df, 
    source_column='hindi_text',
    target_column='english_text',
    batch_size=100
)
```

### Large Dataset with Checkpoints

```python
# Use for datasets that take long to process
from large_dataset_with_checkpoints import transliterate_with_checkpoints

result = transliterate_with_checkpoints(
    df,
    source_column='text',
    checkpoint_path='progress.parquet',
    save_every=50,  # Save every 50 batches
    resume=True      # Resume if interrupted
)
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

### For DataFrame Examples
```bash
pip install pandas tqdm          # Basic DataFrame processing
pip install pandas tqdm pyarrow  # With checkpointing support
```

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

## Performance Guidelines

### For Large Datasets (300k+ words)

1. **Batch Size**: Use 100-200 for optimal API efficiency
2. **Delay**: Set to 0.05-0.1 seconds to avoid rate limits
3. **Checkpointing**: Save every 50-100 batches for safety
4. **Model Selection**:
   - `gpt-4o`: Best quality
   - `gpt-3.5-turbo`: Cost-effective
   - `claude-3-haiku`: Fast and cheap
5. **Parallel Processing**: Split large datasets and process in parallel

### Expected Performance

For 300,000 words:
- **Batch size 200**: ~1,500 API calls
- **Time**: 5-10 minutes (depends on API)
- **Cost**: $15-30 (depends on model)
- **With checkpointing**: Safe to interrupt and resume

## File Formats

### Input
- **Text files**: Plain UTF-8 text, one item per line
- **JSON files**: Structured format from previous results
- **CSV files**: For DataFrame processing

### Output  
- **Text format**: Plain transliterated text
- **JSON format**: Rich metadata with error handling
- **Parquet format**: For checkpointing and large datasets

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
- ✅ **Checkpointing** - save progress for long-running tasks

## Important Notes

1. The DataFrame functions (`pandas_usage.py`) are **reference implementations** - not part of the core package
2. Always test with a small sample first
3. Monitor your API usage and costs
4. Use checkpointing for any dataset that takes >5 minutes
5. These examples are designed to be copied and adapted to your needs

## Support

For issues or questions:
- GitHub: https://github.com/in-rolls/indicate
- Documentation: See main README.md