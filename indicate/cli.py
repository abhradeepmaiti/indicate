#!/usr/bin/env python3
"""
Modern CLI for the indicate package using Click.
"""
from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path

import click

from .hindi2english import HindiToEnglish


def _get_version() -> str:
    """Get package version from metadata."""
    try:
        return metadata.version("indicate")
    except Exception:
        return "unknown"


@click.group()
@click.version_option(version=_get_version())
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Indicate: Transliterations to/from Indian languages.
    
    A Python package for transliterating Hindi text to English using
    a TensorFlow-based encoder-decoder model with attention mechanism.
    """
    ctx.ensure_object(dict)


@cli.command()
@click.argument('text', required=False)
@click.option(
    '--input', '-i', 'input_file',
    type=click.File('r', encoding='utf-8'),
    help='Read Hindi text from file instead of argument'
)
@click.option(
    '--output', '-o', 'output_file', 
    type=click.File('w', encoding='utf-8'),
    help='Write transliterated text to file instead of stdout'
)
@click.option(
    '--batch/--no-batch', default=False,
    help='Process input line by line (useful for large files)'
)
@click.option(
    '--quiet', '-q', is_flag=True,
    help='Suppress progress output'
)
def hindi2english(
    text: str | None,
    input_file: click.File | None,
    output_file: click.File | None, 
    batch: bool,
    quiet: bool,
) -> None:
    """Transliterate Hindi text to English.
    
    TEXT: Hindi text to transliterate. If not provided, will read from --input or stdin.
    
    Examples:
    
        # Transliterate text directly
        indicate hindi2english "राजशेखर चिंतालपति"
        
        # From file
        indicate hindi2english --input hindi.txt --output english.txt
        
        # From stdin
        echo "गौरव सूद" | indicate hindi2english
        
        # Batch processing
        indicate hindi2english --input large_file.txt --batch --quiet
    """
    try:
        # Determine input source
        if text:
            input_text = text
        elif input_file:
            if batch:
                # Process line by line for large files
                _process_batch(input_file, output_file, quiet)
                return
            else:
                input_text = input_file.read().strip()
        else:
            # Read from stdin
            if not sys.stdin.isatty():
                input_text = sys.stdin.read().strip()
            else:
                click.echo("No input provided. Use --help for usage information.", err=True)
                sys.exit(1)
        
        if not input_text:
            click.echo("No text to transliterate.", err=True)
            sys.exit(1)
        
        # Transliterate
        if not quiet:
            click.echo("Transliterating...", err=True)
        
        result = HindiToEnglish.transliterate(input_text)
        
        # Output result
        if output_file:
            output_file.write(result)
            if not quiet:
                click.echo(f"Result written to {output_file.name}", err=True)
        else:
            click.echo(result)
            
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled.", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


def _process_batch(
    input_file: click.File,
    output_file: click.File | None,
    quiet: bool,
) -> None:
    """Process input file line by line."""
    lines = input_file.readlines()
    
    if not quiet:
        from tqdm import tqdm
        lines = tqdm(lines, desc="Transliterating", unit="lines")
    
    results = []
    for line in lines:
        line = line.strip()
        if line:
            result = HindiToEnglish.transliterate(line)
            results.append(result)
        else:
            results.append("")
    
    output_text = "\n".join(results)
    
    if output_file:
        output_file.write(output_text)
        if not quiet:
            click.echo(f"Results written to {output_file.name}", err=True)
    else:
        click.echo(output_text)


@cli.command()
def info() -> None:
    """Show information about the indicate package."""
    click.echo("Indicate - Hindi to English Transliteration")
    click.echo("=" * 45)
    click.echo()
    click.echo("Model Architecture: Encoder-Decoder with Luong attention")
    click.echo("Embedding dimension: 256")
    click.echo("LSTM units: 1024")
    click.echo()
    click.echo("Training Data Sources:")
    click.echo("  • ESPN Cricinfo")
    click.echo("  • Election affidavits")
    click.echo("  • Google Dakshina dataset")
    click.echo("  • IIT Bombay corpus")
    click.echo()
    
    # Try to show model path info
    try:
        model_path = HindiToEnglish.get_model_path()
        click.echo(f"Model path: {model_path}")
        
        # Check if model files exist
        from pathlib import Path
        if Path(model_path).exists():
            click.echo("✓ Model files found")
        else:
            click.echo("⚠ Model files not found")
    except Exception as e:
        click.echo(f"⚠ Could not locate model: {e}")


# Legacy entry point for backward compatibility
def main(argv: list[str] | None = None) -> int:
    """Legacy entry point for backward compatibility."""
    import sys
    if argv is None:
        argv = sys.argv[1:]
    
    # Handle old-style arguments for backward compatibility
    if argv and not any(arg.startswith('-') for arg in argv):
        # If no flags, assume it's hindi2english with text argument
        argv = ['hindi2english'] + argv
    elif '--type' in argv:
        # Handle legacy --type and --input arguments
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--type", default=None)
        parser.add_argument("--input", default=None)
        args, remaining = parser.parse_known_args(argv)
        
        if args.type == "hin2eng" and args.input:
            argv = ['hindi2english', args.input] + remaining
    
    # If no command specified, default to hindi2english
    if not argv or (argv and not argv[0] in ['hindi2english', 'info']):
        argv = ['hindi2english'] + argv
    
    try:
        cli(argv, standalone_mode=False)
        return 0
    except SystemExit as e:
        return e.code
    except Exception:
        return 1


if __name__ == '__main__':
    sys.exit(main())