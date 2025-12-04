"""
Transliteration module with backward compatibility.

This module maintains the original API while delegating to the new Click-based CLI.
"""
from .hindi2english import HindiToEnglish
from .cli import main

# Keep the function available for API usage
hindi2english = HindiToEnglish.transliterate

# Export the main function for entry point
__all__ = ['hindi2english', 'main']