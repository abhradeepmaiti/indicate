#!/usr/bin/env python3
"""
Test API functions for the indicate package.
"""

from __future__ import annotations

import unittest

import indicate
from indicate import transliterate
from indicate.hindi2english import HindiToEnglish


class TestAPI(unittest.TestCase):
    def test_main_import(self):
        """Test that main import works correctly."""
        # Test the main package import
        result = indicate.hindi2english("हिंदी")
        self.assertIsInstance(result, str)
        self.assertEqual(result.lower(), "hindi")

    def test_transliterate_module_import(self):
        """Test that transliterate module import works."""
        result = transliterate.hindi2english("गौरव")
        self.assertIsInstance(result, str)
        self.assertEqual(result.lower(), "gaurav")

    def test_hindi2english_class_direct(self):
        """Test direct class usage."""
        result = HindiToEnglish.transliterate("राजशेखर")
        self.assertIsInstance(result, str)
        self.assertEqual(result.lower(), "rajshekar")

    def test_hindi2english_singleton(self):
        """Test that HindiToEnglish is a singleton."""
        instance1 = HindiToEnglish()
        instance2 = HindiToEnglish()
        self.assertIs(instance1, instance2)

    def test_model_path_methods(self):
        """Test that model path methods work."""
        model_path = HindiToEnglish.get_model_path()
        self.assertIsInstance(model_path, str)
        self.assertTrue(
            model_path.endswith("saved_weights")
            or model_path.endswith("saved_weights/")
        )

        input_vocab = HindiToEnglish.get_input_vocab()
        self.assertIsInstance(input_vocab, str)
        self.assertTrue(input_vocab.endswith("hindi_tokens.json"))

        target_vocab = HindiToEnglish.get_target_vocab()
        self.assertIsInstance(target_vocab, str)
        self.assertTrue(target_vocab.endswith("english_tokens.json"))

    def test_multiple_translations(self):
        """Test multiple translations in sequence."""
        # v2 (Aksharantar-scaled) model outputs.
        test_cases = [
            ("हिंदी", "hindi"),
            ("गौरव", "gaurav"),
            ("राजशेखर", "rajshekar"),
            ("चिंतालपति", "chintalpati"),
        ]

        for hindi, expected in test_cases:
            with self.subTest(hindi=hindi):
                result = transliterate.hindi2english(hindi)
                self.assertEqual(result.lower(), expected)

    def test_consistent_results(self):
        """Test that same input gives consistent results."""
        input_text = "हिंदी"
        result1 = indicate.hindi2english(input_text)
        result2 = transliterate.hindi2english(input_text)
        result3 = HindiToEnglish.transliterate(input_text)

        # All three ways should give the same result
        self.assertEqual(result1, result2)
        self.assertEqual(result2, result3)

    def test_api_return_types(self):
        """Test that all API functions return strings."""
        input_text = "हिंदी"

        result1 = indicate.hindi2english(input_text)
        result2 = transliterate.hindi2english(input_text)
        result3 = HindiToEnglish.transliterate(input_text)

        self.assertIsInstance(result1, str)
        self.assertIsInstance(result2, str)
        self.assertIsInstance(result3, str)


if __name__ == "__main__":
    unittest.main()
