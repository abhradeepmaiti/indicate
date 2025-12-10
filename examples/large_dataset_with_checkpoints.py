#!/usr/bin/env python3
"""
Large Dataset Processing with Checkpointing

This example shows how to process large DataFrames with automatic checkpointing
for reliability and the ability to resume interrupted processing.

Requirements:
    pip install indicate pandas tqdm pyarrow

Features:
    - Automatic checkpointing to Parquet files
    - Resume from interruption
    - Progress tracking
    - Error recovery

Usage:
    python examples/large_dataset_with_checkpoints.py
"""

import os
import shutil
import time
from pathlib import Path
from typing import Optional

try:
    import pandas as pd
    from tqdm import tqdm
except ImportError:
    print("This example requires pandas and tqdm. Install with:")
    print("  pip install pandas tqdm pyarrow")
    exit(1)

from indicate import IndicLLMTransliterator


def transliterate_with_checkpoints(
    df: pd.DataFrame,
    source_column: str,
    target_column: str = "transliterated",
    checkpoint_path: str = "checkpoint.parquet",
    save_every: int = 50,
    resume: bool = True,
    backup_on_complete: bool = True,
    source_lang: Optional[str] = None,
    target_lang: str = "english",
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    batch_size: int = 100,
    delay: float = 0.1,
    use_few_shot: bool = True,
) -> pd.DataFrame:
    """
    Transliterate DataFrame with automatic checkpointing for reliability.
    
    This function is designed for large datasets where processing might take
    a long time or be interrupted. It saves progress periodically and can
    resume from where it left off.
    
    Args:
        df: Input DataFrame
        source_column: Column containing text to transliterate
        target_column: Column for transliterated text
        checkpoint_path: Path to save checkpoint file
        save_every: Save checkpoint every N batches
        resume: Resume from checkpoint if it exists
        backup_on_complete: Keep backup after completion
        source_lang: Source language (auto-detected if None)
        target_lang: Target language
        provider: LLM provider
        model: Specific model to use
        api_key: API key
        batch_size: Texts per API call
        delay: Delay between API calls
        use_few_shot: Use few-shot examples
    
    Returns:
        DataFrame with transliterated column
    """
    
    # Validate input
    if source_column not in df.columns:
        raise ValueError(f"Source column '{source_column}' not found")
    
    # Setup checkpoint path
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try to resume from checkpoint
    start_index = 0
    if resume and checkpoint_path.exists():
        try:
            checkpoint_df = pd.read_parquet(checkpoint_path)
            if target_column in checkpoint_df.columns:
                # Copy existing results
                df[target_column] = checkpoint_df[target_column]
                
                # Find where we left off
                processed_mask = checkpoint_df[target_column].notna()
                start_index = processed_mask.sum()
                
                print(f"✓ Resumed from checkpoint: {start_index} texts already processed")
        except Exception as e:
            print(f"⚠ Failed to load checkpoint: {e}. Starting fresh.")
            start_index = 0
    
    # Get texts to process
    if len(df) == 0:
        mask = pd.Series([], dtype=bool)
    else:
        mask = df[source_column].notna() & (df[source_column].str.strip() != "")
    
    texts_to_process = df.loc[mask, source_column].tolist()
    indices_to_process = df.loc[mask].index.tolist()
    
    # Skip already processed texts
    if start_index > 0:
        texts_to_process = texts_to_process[start_index:]
        indices_to_process = indices_to_process[start_index:]
    
    if not texts_to_process:
        print("No texts to process")
        return df
    
    print(f"Processing {len(texts_to_process)} texts (batch size: {batch_size})")
    
    # Initialize column if needed
    if target_column not in df.columns:
        df[target_column] = ""
    
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
        raise ValueError(f"Failed to initialize transliterator: {e}")
    
    # Process with progress bar and checkpointing
    batch_count = 0
    successful = 0
    failed = 0
    
    try:
        with tqdm(total=len(texts_to_process), desc="Transliterating") as pbar:
            for i in range(0, len(texts_to_process), batch_size):
                batch_texts = texts_to_process[i:i + batch_size]
                batch_indices = indices_to_process[i:i + batch_size]
                
                # Process batch
                for text, idx in zip(batch_texts, batch_indices):
                    try:
                        result = transliterator.transliterate(
                            text, 
                            use_few_shot=use_few_shot
                        )
                        df.at[idx, target_column] = result
                        successful += 1
                    except Exception as e:
                        print(f"\n⚠ Error: {e}")
                        df.at[idx, target_column] = f"[ERROR: {str(e)}]"
                        failed += 1
                    
                    pbar.update(1)
                    
                    if delay > 0:
                        time.sleep(delay)
                
                batch_count += 1
                
                # Save checkpoint periodically
                if batch_count % save_every == 0:
                    try:
                        df.to_parquet(checkpoint_path, index=False)
                        print(f"\n✓ Checkpoint saved (batch {batch_count})")
                    except Exception as e:
                        print(f"\n⚠ Failed to save checkpoint: {e}")
    
    except KeyboardInterrupt:
        print("\n\n⚠ Processing interrupted!")
        # Save emergency checkpoint
        try:
            df.to_parquet(checkpoint_path, index=False)
            print(f"✓ Emergency checkpoint saved to {checkpoint_path}")
            print(f"  Resume by running the script again with resume=True")
        except Exception as e:
            print(f"⚠ Failed to save emergency checkpoint: {e}")
        raise
    
    # Final save
    try:
        df.to_parquet(checkpoint_path, index=False)
        print(f"\n✓ Final checkpoint saved")
        
        # Handle backup
        if backup_on_complete:
            backup_path = checkpoint_path.with_suffix('.completed.parquet')
            shutil.copy2(checkpoint_path, backup_path)
            print(f"✓ Backup saved to {backup_path}")
        else:
            checkpoint_path.unlink()
            print(f"✓ Checkpoint file removed")
    except Exception as e:
        print(f"⚠ Failed to handle final checkpoint: {e}")
    
    print(f"\n✓ Completed! Successful: {successful}, Failed: {failed}")
    
    return df


def example_basic_checkpoint():
    """Basic example with checkpointing."""
    print("=== Basic Checkpointing Example ===\n")
    
    # Create sample data
    df = pd.DataFrame({
        'text': [f'नमस्ते {i}' for i in range(20)],
        'id': range(20)
    })
    
    print(f"Sample DataFrame with {len(df)} rows")
    
    # Check for API key
    if not any(os.environ.get(key) for key in ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'GOOGLE_API_KEY']):
        print("\n⚠ No API key found. Set one of:")
        print("  OPENAI_API_KEY, ANTHROPIC_API_KEY, or GOOGLE_API_KEY")
        return
    
    # Process with checkpointing
    result = transliterate_with_checkpoints(
        df,
        source_column='text',
        target_column='translated',
        checkpoint_path='demo_checkpoint.parquet',
        save_every=5,  # Save every 5 batches
        batch_size=2,  # Small batch for demo
        delay=0.1
    )
    
    print("\nResults:")
    print(result[['text', 'translated']].head(10))


def example_large_dataset():
    """Example configuration for very large datasets."""
    print("\n=== Large Dataset Configuration ===\n")
    
    print("""
# For a 300k word dataset:

import pandas as pd

# Load your large dataset
df = pd.read_csv('hindi_words_300k.csv')

# Process with aggressive checkpointing
result = transliterate_with_checkpoints(
    df,
    source_column='hindi_word',
    target_column='english_word',
    checkpoint_path='./checkpoints/300k_words.parquet',
    save_every=50,        # Save every 50 batches (~10,000 words)
    resume=True,          # Resume if interrupted
    backup_on_complete=True,  # Keep backup when done
    model='gpt-4o',       # Best quality (or gpt-3.5-turbo for cost)
    batch_size=200,       # Large batches for efficiency
    delay=0.05,          # Minimal delay
    use_few_shot=True    # Better accuracy
)

# Save final results
result.to_csv('translated_300k.csv', index=False)

Benefits:
- Checkpoint every ~1 minute of processing
- Can safely interrupt with Ctrl+C
- Automatically resumes from last checkpoint
- No data loss even if process crashes
- Final backup kept for safety
""")


def example_resume_interrupted():
    """Show how to resume interrupted processing."""
    print("\n=== Resuming Interrupted Processing ===\n")
    
    print("""
If processing was interrupted:

1. The checkpoint file contains all progress
2. Simply run the same command again with resume=True
3. Processing will continue from where it left off

Example:
    # Original run (interrupted at 50,000 words)
    result = transliterate_with_checkpoints(
        df, 
        'text', 
        'translated',
        checkpoint_path='progress.parquet',
        resume=True  # This flag enables resuming
    )
    
    # Run again - automatically continues from word 50,001!

You can also check progress manually:
    checkpoint_df = pd.read_parquet('progress.parquet')
    processed = checkpoint_df['translated'].notna().sum()
    print(f"Processed: {processed} out of {len(checkpoint_df)}")
""")


def example_parallel_processing():
    """Show how to split dataset for parallel processing."""
    print("\n=== Parallel Processing Strategy ===\n")
    
    print("""
For maximum speed, split your dataset and process in parallel:

import pandas as pd
from multiprocessing import Pool

def process_chunk(chunk_data):
    chunk_df, chunk_id = chunk_data
    return transliterate_with_checkpoints(
        chunk_df,
        'text',
        'translated',
        checkpoint_path=f'checkpoint_{chunk_id}.parquet',
        # ... other params
    )

# Split dataset into chunks
df = pd.read_csv('large_dataset.csv')
chunks = np.array_split(df, 4)  # Split into 4 parts

# Process in parallel
with Pool(4) as pool:
    results = pool.map(
        process_chunk,
        [(chunk, i) for i, chunk in enumerate(chunks)]
    )

# Combine results
final_df = pd.concat(results, ignore_index=True)

Note: Each chunk gets its own checkpoint file for safety!
""")


def main():
    """Run examples."""
    print("=" * 60)
    print("Large Dataset Processing with Checkpoints")
    print("=" * 60)
    
    example_basic_checkpoint()
    example_large_dataset()
    example_resume_interrupted()
    example_parallel_processing()
    
    print("\n" + "=" * 60)
    print("Key takeaways:")
    print("- Always use checkpointing for large datasets")
    print("- Set save_every based on your risk tolerance")
    print("- Keep resume=True to handle interruptions")
    print("- Consider parallel processing for maximum speed")
    print("\nFor more info: https://github.com/in-rolls/indicate")


if __name__ == "__main__":
    main()