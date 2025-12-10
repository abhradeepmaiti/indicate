#!/usr/bin/env python3
"""
DataFrame Processing with Indicate

This example shows how to use indicate for transliterating pandas DataFrames.
This is a reference implementation - adapt it to your specific needs.

Requirements:
    pip install indicate pandas tqdm

Usage:
    python examples/pandas_usage.py
"""

import os
import time
from typing import Optional

# Check for required packages
try:
    import pandas as pd
    from tqdm import tqdm
except ImportError:
    print("This example requires pandas and tqdm. Install with:")
    print("  pip install pandas tqdm")
    exit(1)

from indicate import IndicLLMTransliterator


def transliterate_dataframe(
    df: pd.DataFrame,
    source_column: str,
    target_column: str = "transliterated",
    source_lang: Optional[str] = None,
    target_lang: str = "english",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    batch_size: int = 100,
    delay: float = 0.1,
    use_few_shot: bool = True,
    overwrite: bool = True,
) -> pd.DataFrame:
    """
    Transliterate a column of text in a DataFrame using indicate LLM.
    
    This is a reference implementation showing how to process DataFrames
    with indicate. Adapt it to your specific needs.
    
    Args:
        df: Input DataFrame
        source_column: Name of column containing text to transliterate
        target_column: Name of new column for transliterated text
        source_lang: Source language (auto-detected if None)
        target_lang: Target language (default: english)
        provider: LLM provider (openai, anthropic, google)
        model: Specific model to use (e.g., 'gpt-4o', 'claude-3-opus')
        api_key: API key (uses environment variables if None)
        batch_size: Number of texts to process in one API call
        delay: Delay between API calls to avoid rate limits
        use_few_shot: Whether to use few-shot examples
        overwrite: Whether to overwrite existing target_column
    
    Returns:
        DataFrame with new transliterated column
    """
    # Validate input
    if source_column not in df.columns:
        raise ValueError(f"Source column '{source_column}' not found in DataFrame")
    
    if target_column in df.columns and not overwrite:
        raise ValueError(
            f"Target column '{target_column}' already exists. "
            f"Set overwrite=True to overwrite it."
        )
    
    # Get non-null, non-empty texts
    if len(df) == 0:
        mask = pd.Series([], dtype=bool)
    else:
        mask = df[source_column].notna() & (df[source_column].str.strip() != "")
    texts_to_process = df.loc[mask, source_column].tolist()
    
    if not texts_to_process:
        print("No text to transliterate (all entries are null or empty)")
        df[target_column] = ""
        return df
    
    print(f"Processing {len(texts_to_process)} texts in batches of {batch_size}...")
    
    # Initialize transliterator
    try:
        transliterator = IndicLLMTransliterator(
            source_lang=source_lang or "hindi",
            target_lang=target_lang,
            provider=provider,
            model=model,
            api_key=api_key,
        )
        print(f"✓ Using {transliterator.provider} with model {transliterator.model}")
    except Exception as e:
        raise ValueError(f"Failed to initialize transliterator: {e}") from e
    
    # Process in batches with progress bar
    results = {}
    successful_count = 0
    failed_count = 0
    
    with tqdm(total=len(texts_to_process), desc="Transliterating", unit="texts") as pbar:
        for i in range(0, len(texts_to_process), batch_size):
            batch_texts = texts_to_process[i:i + batch_size]
            batch_indices = df.loc[mask].index[i:i + batch_size]
            
            try:
                if batch_size == 1 or len(batch_texts) == 1:
                    # Single text processing
                    for text, idx in zip(batch_texts, batch_indices):
                        try:
                            result = transliterator.transliterate(text, use_few_shot=use_few_shot)
                            results[idx] = result
                            successful_count += 1
                        except Exception as e:
                            print(f"\nError processing text: {e}")
                            results[idx] = f"[ERROR: {str(e)}]"
                            failed_count += 1
                        
                        pbar.update(1)
                        if delay > 0:
                            time.sleep(delay)
                else:
                    # Batch processing
                    try:
                        batch_results = transliterator.transliterate_batch(
                            batch_texts, 
                            batch_size=len(batch_texts), 
                            use_few_shot=use_few_shot
                        )
                        
                        for idx, result in zip(batch_indices, batch_results):
                            results[idx] = result
                            if result and not result.startswith("[ERROR:"):
                                successful_count += 1
                            else:
                                failed_count += 1
                        
                        pbar.update(len(batch_texts))
                        
                    except Exception as e:
                        print(f"\nBatch processing failed, falling back to individual: {e}")
                        # Fallback to individual processing
                        for text, idx in zip(batch_texts, batch_indices):
                            try:
                                result = transliterator.transliterate(text, use_few_shot=use_few_shot)
                                results[idx] = result
                                successful_count += 1
                            except Exception as e:
                                results[idx] = f"[ERROR: {str(e)}]"
                                failed_count += 1
                            
                            pbar.update(1)
                            if delay > 0:
                                time.sleep(delay)
                
                # Rate limiting between batches
                if delay > 0 and len(batch_texts) > 1:
                    time.sleep(delay)
                    
            except KeyboardInterrupt:
                print("\nProcessing interrupted by user")
                break
            except Exception as e:
                print(f"\nUnexpected error: {e}")
                continue
    
    # Create result column
    if target_column not in df.columns:
        df[target_column] = ""
    
    # Update with results
    for idx, result in results.items():
        df.at[idx, target_column] = result
    
    print(f"\n✓ Completed! Successful: {successful_count}, Failed: {failed_count}")
    
    if failed_count > 0:
        print(f"⚠ {failed_count} texts failed. Check '[ERROR: ...]' entries.")
    
    return df


def create_sample_data():
    """Create sample Hindi data for demonstration."""
    return pd.DataFrame({
        'names': [
            'राजेश कुमार',
            'प्रिया शर्मा', 
            'अमित पटेल',
            'सुनीता देवी',
            'विकास सिंह'
        ],
        'cities': [
            'मुंबई',
            'दिल्ली',
            'बैंगलोर',
            'कोलकाता',
            'चेन्नई'
        ],
        'descriptions': [
            'सॉफ्टवेयर इंजीनियर',
            'डॉक्टर',
            'शिक्षक',
            'व्यापारी',
            'कलाकार'
        ]
    })


def example_basic():
    """Basic DataFrame transliteration example."""
    print("=== Basic DataFrame Transliteration ===\n")
    
    # Create sample data
    df = create_sample_data()
    print("Original DataFrame:")
    print(df)
    print()
    
    # Check for API key
    if not any(os.environ.get(key) for key in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY']):
        print("⚠ No API key found. Set one of:")
        print("  OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY")
        print("\nExample usage:")
        print("  export OPENAI_API_KEY='your-key-here'")
        return
    
    # Transliterate names column
    try:
        result_df = transliterate_dataframe(
            df,
            source_column='names',
            target_column='names_english',
            source_lang='hindi',
            batch_size=5,
            delay=0.1
        )
        
        print("\nResult:")
        print(result_df[['names', 'names_english']])
        
    except Exception as e:
        print(f"Error: {e}")


def example_multiple_columns():
    """Transliterate multiple columns."""
    print("\n=== Multiple Columns Example ===\n")
    
    df = create_sample_data()
    
    # Check for API key
    if not any(os.environ.get(key) for key in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY']):
        print("⚠ API key required. Skipping example.")
        return
    
    # Process multiple columns
    columns_to_transliterate = ['names', 'cities', 'descriptions']
    
    for col in columns_to_transliterate:
        print(f"Transliterating column: {col}")
        df = transliterate_dataframe(
            df,
            source_column=col,
            target_column=f"{col}_english",
            batch_size=5
        )
    
    print("\nFinal DataFrame:")
    print(df)


def example_large_dataset():
    """Example for large datasets with progress tracking."""
    print("\n=== Large Dataset Processing ===\n")
    
    print("For large datasets (e.g., 300k words), use this approach:")
    print("""
# Load your data
df = pd.read_csv('large_dataset.csv')

# Process with optimal settings
result_df = transliterate_dataframe(
    df,
    source_column='hindi_text',
    target_column='english_text',
    model='gpt-4o',        # Or 'gpt-3.5-turbo' for lower cost
    batch_size=200,        # Large batches for efficiency
    delay=0.05,           # Minimal delay (adjust based on rate limits)
    use_few_shot=True     # Better accuracy
)

# Save results
result_df.to_csv('translated_results.csv', index=False)

# Expected performance for 300k words:
# - API calls: ~1,500 (at 200 words/batch)
# - Time: 5-10 minutes
# - Cost: $15-30 (depending on model)
""")


def example_custom_model():
    """Example with specific model selection."""
    print("\n=== Custom Model Selection ===\n")
    
    print("Available models by provider:")
    print("""
OpenAI:
  - gpt-4o (best quality)
  - gpt-4-turbo 
  - gpt-3.5-turbo (cost-effective)

Anthropic:
  - claude-3-opus-20240229 (best)
  - claude-3-sonnet-20240229
  - claude-3-haiku-20240307 (fast)

Google:
  - gemini-pro
  - gemini-1.5-pro

Example usage:
  result = transliterate_dataframe(
      df,
      source_column='text',
      provider='openai',
      model='gpt-4o'
  )
""")


def example_error_handling():
    """Show error handling patterns."""
    print("\n=== Error Handling ===\n")
    
    print("The function handles various error cases:")
    print("""
1. Empty/null values: Skipped automatically
2. API failures: Marked as [ERROR: ...] in results
3. Rate limits: Use delay parameter
4. Interruption: Ctrl+C stops gracefully

# Check for errors after processing:
errors = df[df['translated'].str.startswith('[ERROR:')]
if len(errors) > 0:
    print(f"Failed translations: {len(errors)}")
    # Retry failed ones with different settings
""")


def main():
    """Run all examples."""
    print("=" * 60)
    print("Indicate DataFrame Processing Examples")
    print("=" * 60)
    
    example_basic()
    example_multiple_columns()
    example_large_dataset()
    example_custom_model()
    example_error_handling()
    
    print("\n" + "=" * 60)
    print("Adapt these examples to your specific needs!")
    print("For more info: https://github.com/in-rolls/indicate")


if __name__ == "__main__":
    main()