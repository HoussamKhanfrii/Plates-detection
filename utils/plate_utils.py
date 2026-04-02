"""
Shared utility functions for plate text normalisation and visualisation.
"""
import re


# ─── Text cleanup ─────────────────────────────────────────────────────────────
_OCR_SUBSTITUTIONS = {
    # Common misreadings in digit positions
    "O": "0",
    "I": "1",
    "S": "5",
    "B": "8",
    "Z": "2",
    "G": "6",
}

_PLATE_PATTERN = re.compile(r"[^A-Z0-9\- ]")


def normalize_plate(text: str, apply_digit_substitution: bool = False) -> str:
    """
    Normalise a raw OCR plate string:
    - Uppercase
    - Strip whitespace
    - Remove invalid characters
    - Optionally apply digit substitution heuristic
    """
    text = text.upper().strip()
    text = _PLATE_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()

    if apply_digit_substitution:
        # Only substitute in positions that look like digit fields
        # (simple heuristic: replace known letter-digit confusions)
        text = "".join(_OCR_SUBSTITUTIONS.get(c, c) for c in text)

    return text


def is_valid_plate(text: str, min_len: int = 3, max_len: int = 12) -> bool:
    """
    Basic sanity check for a plate string.
    Rejects empty strings and implausibly short/long results.
    """
    cleaned = text.strip()
    return min_len <= len(cleaned) <= max_len


def format_confidence(conf: float) -> str:
    """Format confidence as a percentage string."""
    return f"{conf * 100:.1f}%"
