"""
Transliteration module with backward compatibility.

This module maintains the original API while delegating to the new Click-based CLI.
"""

from __future__ import annotations

from .cli import main
from .hindi2english import HindiToEnglish
from .punjabi2english import PunjabiToEnglish

# Keep the functions available for API usage
hindi2english = HindiToEnglish.transliterate
punjabi2english = PunjabiToEnglish.transliterate

# Export the main function for entry point
__all__ = ["hindi2english", "punjabi2english", "main"]
