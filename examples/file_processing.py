#!/usr/bin/env python3
"""
File processing examples with safe handling and JSON output.

This example demonstrates the production-ready file handling features
of the indicate package, including backup, resume, and structured output.
"""

import json
import tempfile
from pathlib import Path

from indicate.file_utils import (
    OutputFormat,
    TransliterationResult,
    create_backup,
    read_input_file,
    validate_file_paths,
    write_output_safely,
)


def safe_file_processing_example():
    """Demonstrate safe file processing with validation and backups."""
    print("=== Safe File Processing ===")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create input file
        input_file = tmp_path / "indian_names.txt"
        with open(input_file, "w", encoding="utf-8") as f:
            f.write("राजेश कुमार\n")
            f.write("गौरव सूद\n")
            f.write("प्रिया शर्मा\n")

        output_file = tmp_path / "results.json"

        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")

        # 1. Validate paths (safety check)
        try:
            validate_file_paths(input_file, output_file)
            print("✅ File path validation passed")
        except ValueError as e:
            print(f"❌ Path validation failed: {e}")
            return

        # 2. Read input safely
        lines = read_input_file(input_file)
        print(f"📖 Read {len(lines)} lines from input")

        # 3. Create mock results (simulating LLM processing)
        results = [
            TransliterationResult(
                line_number=i + 1,
                input_text=line,
                output_text=f"Result_{i + 1}",  # Mock output
                source_lang="hindi",
                target_lang="english",
                confidence="high",
                processing_time=0.5,
            )
            for i, line in enumerate(lines)
        ]

        # 4. Write output safely with atomic operation
        write_output_safely(
            results, output_file, OutputFormat.JSON, "hindi", "english", atomic=True
        )
        print("💾 Results written safely")

        # 5. Verify output
        with open(output_file, encoding="utf-8") as f:
            data = json.load(f)

        print("📊 JSON structure:")
        print(f"  - Total lines: {data['metadata']['total_lines']}")
        print(f"  - Successful: {data['metadata']['successful_lines']}")
        print(f"  - Failed: {data['metadata']['failed_lines']}")


def backup_and_overwrite_example():
    """Show how backup functionality protects existing files."""
    print("\n=== Backup Protection ===")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create existing file
        existing_file = tmp_path / "existing_results.txt"
        with open(existing_file, "w", encoding="utf-8") as f:
            f.write("Previous important results\nDo not lose this data!")

        print(f"Existing file: {existing_file}")
        print("Original content:", existing_file.read_text())

        # Create backup before overwriting
        backup_path = create_backup(existing_file)

        if backup_path:
            print(f"✅ Backup created: {backup_path}")
            print("Backup content:", backup_path.read_text())
        else:
            print("❌ Backup creation failed")

        # Now safe to overwrite
        new_results = [
            TransliterationResult(
                line_number=1,
                input_text="नया डेटा",
                output_text="New Data",
                source_lang="hindi",
                target_lang="english",
            )
        ]

        write_output_safely(
            new_results, existing_file, OutputFormat.TEXT, "hindi", "english"
        )

        print("New content:", existing_file.read_text())


def json_roundtrip_example():
    """Show how JSON output can be used as input for chaining operations."""
    print("\n=== JSON Round-trip Processing ===")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create initial JSON with Hindi names
        initial_data = {
            "metadata": {
                "source_language": "hindi",
                "target_language": "english",
                "total_lines": 3,
            },
            "results": [
                {
                    "line_number": 1,
                    "input_text": "राजेश कुमार",
                    "output_text": "Rajesh Kumar",
                    "source_lang": "hindi",
                    "target_lang": "english",
                },
                {
                    "line_number": 2,
                    "input_text": "गौरव सूद",
                    "output_text": "Gaurav Sood",
                    "source_lang": "hindi",
                    "target_lang": "english",
                },
                {
                    "line_number": 3,
                    "input_text": "प्रिया शर्मा",
                    "output_text": "Priya Sharma",
                    "source_lang": "hindi",
                    "target_lang": "english",
                },
            ],
        }

        # Save initial JSON
        initial_json = tmp_path / "initial_results.json"
        with open(initial_json, "w", encoding="utf-8") as f:
            json.dump(initial_data, f, indent=2, ensure_ascii=False)

        print(f"Created initial JSON: {initial_json}")

        # Read JSON as input (extracts original Hindi text)
        input_texts = read_input_file(initial_json)
        print(f"Extracted texts for re-processing: {input_texts}")

        # Process again (simulate reverse transliteration)
        reverse_results = [
            TransliterationResult(
                line_number=i + 1,
                input_text=text,
                output_text=f"Processed_{i + 1}",  # Mock reverse processing
                source_lang="hindi",
                target_lang="english",
                confidence="medium",
            )
            for i, text in enumerate(input_texts)
        ]

        # Save to new JSON
        output_json = tmp_path / "reprocessed_results.json"
        write_output_safely(
            reverse_results, output_json, OutputFormat.JSON, "hindi", "english"
        )

        print(f"Created reprocessed JSON: {output_json}")
        print("✅ JSON round-trip completed")


def error_recovery_example():
    """Demonstrate error handling and partial results."""
    print("\n=== Error Recovery ===")

    # Simulate processing with some failures
    mixed_results = [
        TransliterationResult(
            line_number=1,
            input_text="राजेश",
            output_text="Rajesh",
            source_lang="hindi",
            target_lang="english",
            confidence="high",
        ),
        TransliterationResult(
            line_number=2,
            input_text="invalid_input",
            output_text="",
            source_lang="hindi",
            target_lang="english",
            error="API timeout",
            confidence="none",
        ),
        TransliterationResult(
            line_number=3,
            input_text="गौरव",
            output_text="Gaurav",
            source_lang="hindi",
            target_lang="english",
            confidence="high",
        ),
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_file = Path(tmp_dir) / "mixed_results.json"

        write_output_safely(
            mixed_results, output_file, OutputFormat.JSON, "hindi", "english"
        )

        # Analyze results
        with open(output_file, encoding="utf-8") as f:
            data = json.load(f)

        print("📊 Processing Summary:")
        print(f"  - Total: {data['metadata']['total_lines']}")
        print(f"  - Successful: {data['metadata']['successful_lines']}")
        print(f"  - Failed: {data['metadata']['failed_lines']}")

        print("\n📝 Failed Items:")
        for result in data["results"]:
            if result["error"]:
                print(
                    f"  Line {result['line_number']}: {result['input_text']} - {result['error']}"
                )


def main():
    """Run all file processing examples."""
    print("Indicate File Processing Examples")
    print("=" * 40)

    safe_file_processing_example()
    backup_and_overwrite_example()
    json_roundtrip_example()
    error_recovery_example()

    print("\n✅ All file processing examples completed!")


if __name__ == "__main__":
    main()
