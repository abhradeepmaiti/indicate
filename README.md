# Indicate: Transliterate Indic Languages to English

[![Notary Badge](https://notarypy.soodoku.workers.dev/badge/indicate/0.2.1/indicate-0.2.1-py3-none-any.whl)](https://pypi.org/integrity/indicate/0.2.1/indicate-0.2.1-py3-none-any.whl/provenance)
[![PyPI Version](https://img.shields.io/pypi/v/indicate.svg)](https://pypi.python.org/pypi/indicate)
[![Downloads](https://static.pepy.tech/badge/indicate)](https://pepy.tech/project/indicate)
[![Tests](https://github.com/in-rolls/indicate/workflows/test/badge.svg)](https://github.com/in-rolls/indicate/actions?query=workflow%3Atest)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://in-rolls.github.io/indicate/)

Transliterations to/from Indian languages are still generally low quality. One problem is access to data. Another is that there is no standard transliteration.

For Hindi--English, we build novel dataset for names using the ESPNcricinfo. For instance, see [here](https://www.espncricinfo.com/hindi/series/pakistan-tour-of-england-2021-1239529/england-vs-pakistan-1st-odi-1239537/full-scorecard) for hindi version of the [english scorecard](https://www.espncricinfo.com/series/pakistan-tour-of-england-2021-1239529/england-vs-pakistan-1st-odi-1239537/full-scorecard).

We also create a dataset from [election affidavits](https://affidavit.eci.gov.in/CandidateCustomFilter) and exploit the [Google Dakshina dataset](https://github.com/google-research-datasets/dakshina).

To overcome the fact that there isn't one standard way of transliteration, we provide k-best transliterations.

## Install

We strongly recommend installing `indicate` inside a Python virtual environment (see [venv documentation](https://docs.python.org/3/library/venv.html#creating-virtual-environments))

**Requirements:** Python 3.10 or higher

```bash
pip install indicate
```

## Usage

### Python API

```python
from indicate import transliterate
english_translated = transliterate.hindi2english("हिंदी")
print(english_translated)
# Output: hindi
```

### Command Line Interface

The package provides both modern and legacy CLI interfaces:

#### Modern CLI (Recommended)

```bash
# Basic usage
indicate hindi2english "राजशेखर चिंतालपति"

# From file
indicate hindi2english --input hindi.txt --output english.txt

# From stdin
echo "गौरव सूद" | indicate hindi2english

# Batch processing for large files
indicate hindi2english --input large_file.txt --batch --quiet

# Get help
indicate hindi2english --help

# Package information
indicate info
```

#### Legacy CLI (Backward Compatibility)

```bash
# Still supported for backward compatibility
hindi2english --type hin2eng --input "हिंदी"
```

## Functions

We expose 1 function, which will take Hindi text and transliterate it to English.

- **transliterate.hindi2english(input)**
  - What it does: Converts given hindi text into English alphabet
  - Output: Returns text in English

## Testing Locally

To test the package locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/in-rolls/indicate.git
   cd indicate
   ```

2. **Install with uv (recommended)**:
   ```bash
   uv sync
   ```
   
   Or with pip:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .
   ```

3. **Run tests**:
   ```bash
   # Run all tests
   python -m unittest discover tests/
   
   # Run specific test
   python -m unittest tests.test_010_hindi_translate
   ```

4. **Test the transliteration**:
   ```bash
   # Modern CLI
   indicate hindi2english "हिंदी"
   
   # Legacy CLI
   hindi2english --type hin2eng --input "हिंदी"
   
   # Python usage
   python -c "from indicate import transliterate; print(transliterate.hindi2english('हिंदी'))"
   ```

## Data

The datasets used to train the model:

- [Indian Election affidavits](https://affidavit.eci.gov.in/CandidateCustomFilter)
- [Google Dakshina dataset](https://github.com/google-research-datasets/dakshina)
- [ESPN Cric Info](https://www.espncricinfo.com/hindi/series/pakistan-tour-of-england-2021-1239529/england-vs-pakistan-1st-odi-1239537/full-scorecard) for hindi version of the [english scorecard](https://www.espncricinfo.com/series/pakistan-tour-of-england-2021-1239529/england-vs-pakistan-1st-odi-1239537/full-scorecard)
- [IIT Bombay English-Hindi Corpus](https://www.cfilt.iitb.ac.in/iitb_parallel/)

## Evaluation

Model was evaluated on test dataset of Google Dakshina dataset, Model predicted 73.64% exact matches.
[Indic-trans](https://github.com/libindic/indic-trans) predicted 63.12% exact matches on Google Dakshina dataset.

Below is the edit distance metrics on test dataset (0.0 mean exact match, the farther away from 0.0, the difference is more between predicted text and actual text):

![Edit distance metrics of model on Google Dakshina test dataset](https://github.com/in-rolls/indicate/raw/master/images/h2e_ed.png)

## Authors

Rajashekar Chintalapati and Gaurav Sood

## Contributor Code of Conduct

The project welcomes contributions from everyone! In fact, it depends on it. To maintain this welcoming atmosphere, and to collaborate in a fun and productive way, we expect contributors to the project to abide by the [Contributor Code of Conduct](http://contributor-covenant.org/version/1/0/0/).

## License

The package is released under the [MIT License](https://opensource.org/licenses/MIT).