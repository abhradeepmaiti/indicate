#!/usr/bin/env python3
"""
DataFrame Transliteration Script using indicate LLM

This script transliterates a column of text in a pandas DataFrame using
the indicate package's LLM functionality, adding the results as a new column.

Example usage:
    python dataframe_transliteration.py
"""

import os
import time
from typing import Optional

import pandas as pd
from tqdm import tqdm

from indicate.llm_indic import IndicLLMTransliterator


def transliterate_dataframe(
    df: pd.DataFrame,
    source_column: str,
    target_column: str = "transliterated",
    source_lang: Optional[str] = None,
    target_lang: str = "english",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    batch_size: int = 1,
    delay: float = 0.1,
) -> pd.DataFrame:
    """
    Transliterate a column of text in a DataFrame using indicate LLM.
    
    Args:
        df: Input DataFrame
        source_column: Name of column containing text to transliterate
        target_column: Name of new column for transliterated text
        source_lang: Source language (auto-detected if None)
        target_lang: Target language (default: english)
        provider: LLM provider (auto-detected if None)
        model: Specific model to use (uses provider default if None)
        api_key: API key (uses environment variables if None)
        batch_size: Number of texts to process at once
        delay: Delay between API calls (to avoid rate limits)
    
    Returns:
        DataFrame with new transliterated column
    
    Raises:
        ValueError: If source column doesn't exist or API setup fails
    """
    # Validate input
    if source_column not in df.columns:
        raise ValueError(f"Source column '{source_column}' not found in DataFrame")
    
    if target_column in df.columns:
        print(f"Warning: Target column '{target_column}' already exists and will be overwritten")
    
    # Get non-null, non-empty texts
    mask = df[source_column].notna() & (df[source_column].str.strip() != "")
    texts_to_process = df.loc[mask, source_column].tolist()
    
    if not texts_to_process:
        print("No text to transliterate (all entries are null or empty)")
        df[target_column] = ""
        return df
    
    print(f"Processing {len(texts_to_process)} texts...")
    
    # Initialize transliterator
    try:
        transliterator = IndicLLMTransliterator(
            source_lang=source_lang or "hindi",  # Default to hindi if not specified
            target_lang=target_lang,
            provider=provider,
            model=model,
            api_key=api_key,
        )
        print(f"✓ Initialized {transliterator.provider} transliterator")
    except Exception as e:
        raise ValueError(f"Failed to initialize transliterator: {e}")
    
    # Process in batches with progress bar
    results = {}
    
    with tqdm(total=len(texts_to_process), desc="Transliterating") as pbar:
        for i in range(0, len(texts_to_process), batch_size):
            batch_texts = texts_to_process[i:i + batch_size]
            batch_indices = df.loc[mask].index[i:i + batch_size]
            
            if batch_size == 1:
                # Single text processing
                for text, idx in zip(batch_texts, batch_indices):
                    try:
                        result = transliterator.transliterate(text, use_few_shot=True)
                        results[idx] = result
                    except Exception as e:
                        print(f"Error processing '{text[:50]}...': {e}")
                        results[idx] = f"[ERROR: {str(e)}]"
                    
                    pbar.update(1)
                    
                    # Rate limiting
                    if delay > 0:
                        time.sleep(delay)
            else:
                # Batch processing
                try:
                    batch_results = transliterator.transliterate_batch(
                        batch_texts, 
                        batch_size=batch_size, 
                        use_few_shot=True
                    )
                    
                    for idx, result in zip(batch_indices, batch_results):
                        results[idx] = result
                    
                    pbar.update(len(batch_texts))
                    
                except Exception as e:
                    print(f"Batch error, falling back to individual processing: {e}")
                    # Fallback to individual processing
                    for text, idx in zip(batch_texts, batch_indices):
                        try:
                            result = transliterator.transliterate(text, use_few_shot=True)
                            results[idx] = result
                        except Exception as e:
                            print(f"Error processing '{text[:50]}...': {e}")
                            results[idx] = f"[ERROR: {str(e)}]"
                        
                        pbar.update(1)
                        
                        if delay > 0:
                            time.sleep(delay)
                
                # Rate limiting between batches
                if delay > 0:
                    time.sleep(delay)
    
    # Create result column
    df[target_column] = ""
    for idx, result in results.items():
        df.at[idx, target_column] = result
    
    print(f"✓ Completed! Results saved to column '{target_column}'")
    return df


def create_sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    sample_data = {
        'name': [
            'राजशेखर चिंतालपति',
            'गौरव सूद',
            'नरेंद्र मोदी',
            'आमिर खान',
            'दीपिका पादुकोण',
            'विराट कोहली',
            None,  # Test null value
            '',    # Test empty string
            'सचिन तेंदुलकर'
        ],
        'city': [
            'मुंबई',
            'दिल्ली',
            'अहमदाबाद',
            'मुंबई',
            'बैंगलोर',
            'दिल्ली',
            'चेन्नई',
            'कोलकाता',
            'मुंबई'
        ]
    }
    return pd.DataFrame(sample_data)


def main():
    """Main function demonstrating usage."""
    print("DataFrame Transliteration using indicate LLM")
    print("=" * 50)
    
    # Check for API key
    api_keys = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY']
    available_keys = [key for key in api_keys if os.environ.get(key)]
    
    if not available_keys:
        print("\n⚠️  No LLM API keys found in environment variables!")
        print("Please set one of the following:")
        for key in api_keys:
            print(f"  export {key}=your-api-key")
        print("\nUsing sample data for demonstration (transliteration will fail)")
        print("-" * 50)
    else:
        print(f"✓ Found API keys: {', '.join(available_keys)}")
        print("-" * 50)
    
    # Create sample DataFrame
    print("\n1. Creating sample DataFrame...")
    df = create_sample_dataframe()
    print("Sample data:")
    print(df)
    
    # Example 1: Transliterate names
    print("\n2. Transliterating names (Hindi → English)...")
    try:
        df_names = transliterate_dataframe(
            df.copy(),
            source_column='name',
            target_column='name_english',
            source_lang='hindi',
            target_lang='english',
            delay=0.5  # Half-second delay between API calls
        )
        
        print("\nResults:")
        print(df_names[['name', 'name_english']])
        
    except Exception as e:
        print(f"Error during name transliteration: {e}")
        df_names = df.copy()
        df_names['name_english'] = "[API Error - Check your API key]"
    
    # Example 2: Transliterate cities
    print("\n3. Transliterating cities (Hindi → English)...")
    try:
        df_complete = transliterate_dataframe(
            df_names,
            source_column='city',
            target_column='city_english',
            source_lang='hindi',
            target_lang='english',
            delay=0.5
        )
        
        print("\nComplete results:")
        print(df_complete)
        
        # Save to CSV
        output_file = 'transliterated_data.csv'
        df_complete.to_csv(output_file, index=False)
        print(f"\n✓ Results saved to {output_file}")
        
    except Exception as e:
        print(f"Error during city transliteration: {e}")
        print("Partial results from name transliteration:")
        print(df_names)


if __name__ == "__main__":
    main()