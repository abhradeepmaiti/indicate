#!/usr/bin/env python3
"""
Quick DataFrame Transliteration Script

Simple script to transliterate a column in your DataFrame.
"""

import pandas as pd
from indicate.llm_indic import IndicLLMTransliterator


def quick_transliterate(df, column_name, new_column_name=None):
    """
    Quick transliteration of a DataFrame column.
    
    Args:
        df: pandas DataFrame
        column_name: name of column to transliterate
        new_column_name: name for new column (default: column_name + '_english')
    
    Returns:
        DataFrame with new transliterated column
    """
    if new_column_name is None:
        new_column_name = f"{column_name}_english"
    
    # Initialize transliterator (will auto-detect provider from env vars)
    transliterator = IndicLLMTransliterator(
        source_lang="hindi",  # Change this if needed
        target_lang="english"
    )
    
    # Get texts to process (skip empty/null)
    texts = df[column_name].dropna()
    texts = texts[texts.str.strip() != ""]
    
    print(f"Transliterating {len(texts)} texts...")
    
    # Process each text
    results = {}
    for idx, text in texts.items():
        try:
            result = transliterator.transliterate(text)
            results[idx] = result
            print(f"✓ {text} → {result}")
        except Exception as e:
            results[idx] = f"[ERROR: {e}]"
            print(f"✗ Failed: {text}")
    
    # Add results to DataFrame
    df[new_column_name] = ""
    for idx, result in results.items():
        df.at[idx, new_column_name] = result
    
    return df


# Example usage
if __name__ == "__main__":
    # Create sample data
    data = {
        'names': ['राजशेखर', 'गौरव', 'नरेंद्र', 'आमिर']
    }
    df = pd.DataFrame(data)
    
    print("Original DataFrame:")
    print(df)
    print()
    
    # Transliterate
    df_result = quick_transliterate(df, 'names')
    
    print("\nResult:")
    print(df_result)