#!/usr/bin/env python3
"""
Basic LLM transliteration examples for the indicate package.

Before running these examples, set your API key:
    export OPENAI_API_KEY=your-key
    # OR
    export ANTHROPIC_API_KEY=your-key
    # OR
    export GOOGLE_API_KEY=your-key
"""

from indicate import IndicLLMTransliterator, detect_language_from_script


def basic_hindi_to_english():
    """Demonstrate basic Hindi to English transliteration."""
    print("=== Hindi to English ===")

    # Initialize transliterator
    transliterator = IndicLLMTransliterator("hindi", "english")

    # Single names
    names = ["राजशेखर चिंतालपति", "गौरव सूद", "प्रिया शर्मा", "अमित कुमार", "सुनीता गुप्ता"]

    for name in names:
        result = transliterator.transliterate(name)
        print(f"{name} → {result}")


def multilingual_examples():
    """Show transliteration across different Indic languages."""
    print("\n=== Multi-Language Examples ===")

    examples = [
        ("tamil", "english", "முருகன்"),
        ("telugu", "english", "హైదరాబాద్"),
        ("bengali", "english", "কলকাতা"),
        ("gujarati", "english", "અમદાવાદ"),
        ("english", "hindi", "computer"),
        ("hindi", "tamil", "नमस्ते"),
    ]

    for source, target, text in examples:
        trans = IndicLLMTransliterator(source, target)
        result = trans.transliterate(text)
        print(f"{source} → {target}: {text} → {result}")


def auto_detection_example():
    """Demonstrate automatic language detection."""
    print("\n=== Auto Language Detection ===")

    texts = [
        "राजेश कुमार",  # Hindi (Devanagari)
        "সৌরভ গাঙ্গুলী",  # Bengali
        "முருகன்",  # Tamil
        "హైదరాబాద్",  # Telugu
    ]

    for text in texts:
        # Auto-detect source language
        detected_lang = detect_language_from_script(text)
        if detected_lang:
            print(f"Detected {detected_lang}: {text}")

            # Transliterate to English
            trans = IndicLLMTransliterator(detected_lang, "english")
            result = trans.transliterate(text)
            print(f"  → {result}")


def batch_processing_example():
    """Show efficient batch processing."""
    print("\n=== Batch Processing ===")

    # Multiple Hindi names
    hindi_names = ["राजेश कुमार", "प्रिया शर्मा", "अमित पटेल", "सुनीता गुप्ता", "विकास सिंह"]

    # Initialize once, process many
    transliterator = IndicLLMTransliterator("hindi", "english")

    # Batch transliteration (more efficient)
    results = transliterator.transliterate_batch(hindi_names, batch_size=5)

    print("Batch results:")
    for original, transliterated in zip(hindi_names, results, strict=False):
        print(f"  {original} → {transliterated}")


def custom_provider_example():
    """Show how to use specific LLM providers and models."""
    print("\n=== Custom Provider/Model ===")

    # Examples with different providers
    providers = [
        ("openai", "gpt-4"),
        ("anthropic", "claude-3-opus-20240229"),
        ("google", "gemini-pro"),
    ]

    text = "राधा कृष्ण"

    for provider, model in providers:
        try:
            trans = IndicLLMTransliterator(
                "hindi", "english", provider=provider, model=model
            )
            result = trans.transliterate(text)
            print(f"{provider} ({model}): {text} → {result}")
        except Exception as e:
            print(f"{provider}: Skipped ({e})")


def error_handling_example():
    """Demonstrate error handling and robustness."""
    print("\n=== Error Handling ===")

    try:
        # Invalid language pair
        IndicLLMTransliterator("english", "french")
    except ValueError as e:
        print(f"Invalid language pair: {e}")

    try:
        # No API key set
        import os

        # Temporarily clear API keys
        old_keys = {}
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"]:
            if key in os.environ:
                old_keys[key] = os.environ.pop(key)

        IndicLLMTransliterator("hindi", "english")

        # Restore keys
        os.environ.update(old_keys)

    except ValueError as e:
        print(f"No API key detected: {e}")


def main():
    """Run all examples."""
    print("Indicate LLM Transliteration Examples")
    print("=" * 40)

    try:
        basic_hindi_to_english()
        multilingual_examples()
        auto_detection_example()
        batch_processing_example()
        custom_provider_example()
        error_handling_example()

        print("\n✅ All examples completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have set your API key:")
        print("  export OPENAI_API_KEY=your-key")
        print("  export ANTHROPIC_API_KEY=your-key")
        print("  export GOOGLE_API_KEY=your-key")


if __name__ == "__main__":
    main()
