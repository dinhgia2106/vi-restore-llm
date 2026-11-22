"""vi-restore-llm: small LLMs for Vietnamese diacritic restoration and typo correction."""

import sys

__version__ = "0.1.0"


def utf8_stdio() -> None:
    """Make CLI output safe for Vietnamese on Windows consoles (cp1252 by default)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
