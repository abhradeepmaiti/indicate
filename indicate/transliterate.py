"""
Transliteration module with backward compatibility.

This module maintains the original API while delegating to the new Click-based CLI.
"""

from __future__ import annotations

from .cli import main
from .hindi2english import HindiToEnglish

# Keep the function available for API usage
hindi2english = HindiToEnglish.transliterate

# Export the main function for entry point
__all__ = ["hindi2english", "main"]
