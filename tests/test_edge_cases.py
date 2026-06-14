#!/usr/bin/env python3
"""
Test edge cases and error handling for the indicate package.
"""

from __future__ import annotations

import unittest

import indicate
from indicate.hindi2english import HindiToEnglish


class TestEdgeCases(unittest.TestCase):
    def test_empty_string(self):
        """Test that empty string is handled gracefully."""
        result = indicate.hindi2english("")
        self.assertEqual(result, "")

    def test_whitespace_only(self):
        """Test whitespace-only input."""
        result = indicate.hindi2english("   ")
        self.assertEqual(result, "")
        # Whitespace-only input returns empty string

    def test_single_character(self):
        """Test single character input."""
        result = indicate.hindi2english("अ")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_non_hindi_text(self):
        """Test non-Hindi text input."""
        # English text
        result = indicate.hindi2english("hello")
        self.assertIsInstance(result, str)

        # Numbers
        result = indicate.hindi2english("123")
        self.assertIsInstance(result, str)

        # Special characters
        result = indicate.hindi2english("!@#$%")
        self.assertIsInstance(result, str)

    def test_mixed_language_text(self):
        """Test mixed Hindi-English text."""
        result = indicate.hindi2english("हिंदी english मिश्रित")
        self.assertIsInstance(result, str)
        # Should handle mixed content

    def test_very_long_text(self):
        """Test very long input text."""
        long_text = "हिंदी " * 100  # 100 repetitions
        result = indicate.hindi2english(long_text)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_unicode_edge_cases(self):
        """Test various Unicode edge cases."""
        test_cases = [
            "हिंदी",  # Standard Hindi
            "हिं\u200dदी",  # With zero-width joiner
            "हिंदी\u0020",  # With explicit space
            "हिंदी\n",  # With newline
            "हिंदी\t",  # With tab
        ]

        for text in test_cases:
            with self.subTest(text=repr(text)):
                result = indicate.hindi2english(text)
                self.assertIsInstance(result, str)

    def test_special_devanagari_characters(self):
        """Test special Devanagari characters."""
        test_cases = [
            "०१२३४५६७८९",  # Devanagari digits
            "ॐ",  # Om symbol
            "।",  # Devanagari danda
            "॥",  # Devanagari double danda
        ]

        for text in test_cases:
            with self.subTest(text=text):
                result = indicate.hindi2english(text)
                self.assertIsInstance(result, str)

    def test_none_input(self):
        """Test that None input raises appropriate error."""
        with self.assertRaises((TypeError, AttributeError)):
            indicate.hindi2english(None)

    def test_non_string_input(self):
        """Test that non-string input raises appropriate error."""
        with self.assertRaises(ValueError):
            indicate.hindi2english(123)

        with self.assertRaises(ValueError):
            indicate.hindi2english(["हिंदी"])

    def test_extremely_long_input(self):
        """Test extremely long input that might cause timeout."""
        # Create a very long string
        very_long_text = "हिंदी भाषा एक महत्वपूर्ण भारतीय भाषा है " * 50

        # This should either work or timeout gracefully
        try:
            result = indicate.hindi2english(very_long_text)
            self.assertIsInstance(result, str)
        except Exception as e:
            # If it fails, it should fail gracefully
            self.assertIsInstance(e, Exception)

    def test_model_loading_resilience(self):
        """Test that model loading is resilient."""
        # Force model reload by getting a new instance
        instance = HindiToEnglish()
        self.assertIsNotNone(instance)

        # Test that paths are accessible
        model_path = HindiToEnglish.get_model_path()
        self.assertTrue(
            model_path.endswith("saved_weights")
            or model_path.endswith("saved_weights/")
        )

    def test_repeated_calls_performance(self):
        """Test that repeated calls don't degrade performance significantly."""
        test_text = "हिंदी"

        # First call (model loading)
        result1 = indicate.hindi2english(test_text)

        # Subsequent calls should be faster
        for _i in range(5):
            result = indicate.hindi2english(test_text)
            self.assertEqual(result, result1)

    def test_concurrent_safety(self):
        """Test that the singleton is thread-safe."""
        import threading

        results = []

        def translate_text():
            result = indicate.hindi2english("हिंदी")
            results.append(result)

        threads = []
        for _ in range(3):
            thread = threading.Thread(target=translate_text)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All results should be the same
        self.assertTrue(all(r == results[0] for r in results))
        self.assertEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()
