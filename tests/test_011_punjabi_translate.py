#!/usr/bin/env python

"""
Test Punjabi (Gurmukhi) to English transliteration.
"""

import unittest

from indicate.punjabi2english import PunjabiToEnglish


class TestPunjabiToEnglish(unittest.TestCase):
    def test_punjabi_to_english(self):
        # Characterizes the shipped v2 Punjabi model's deterministic beam-search
        # output on common Gurmukhi names.
        test_inputs = ["ਰਵਿ ਸ਼ਰਮਾ", "ਸਿੰਘ", "ਕੌਰ", "ਗੁਰਪ੍ਰੀਤ"]
        test_outputs = ["ravi sharma", "singh", "kaur", "gurpreet"]
        for punjabi, english in zip(test_inputs, test_outputs, strict=False):
            self.assertEqual(PunjabiToEnglish.transliterate(punjabi), english)


if __name__ == "__main__":
    unittest.main()
