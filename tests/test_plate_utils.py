"""
Unit tests for utils/plate_utils.py
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.plate_utils import normalize_plate, is_valid_plate, format_confidence


# ─── normalize_plate ──────────────────────────────────────────────────────────

def test_normalize_plate_basic():
    assert normalize_plate("ab 123 cd!") == "AB 123 CD"


def test_normalize_plate_empty():
    assert normalize_plate("") == ""


def test_normalize_plate_lowercase():
    assert normalize_plate("abc123") == "ABC123"


def test_normalize_plate_strips_whitespace():
    assert normalize_plate("  AB123  ") == "AB123"


def test_normalize_plate_collapses_spaces():
    assert normalize_plate("AB  123") == "AB 123"


def test_normalize_plate_removes_special_chars():
    # Special characters such as ! @ # $ should be stripped
    assert normalize_plate("AB@123#CD") == "AB123CD"


def test_normalize_plate_hyphen_preserved():
    # Hyphens are valid plate characters and must be kept
    assert normalize_plate("AB-123") == "AB-123"


def test_normalize_plate_digit_substitution_off_by_default():
    # Without digit substitution, O is kept as O
    result = normalize_plate("OO123")
    assert result == "OO123"


def test_normalize_plate_digit_substitution_on():
    # With substitution: O→0, I→1, S→5, B→8, Z→2, G→6
    result = normalize_plate("OISBZG", apply_digit_substitution=True)
    assert result == "015826"


def test_normalize_plate_digit_substitution_mixed():
    result = normalize_plate("AB O12", apply_digit_substitution=True)
    assert "0" in result  # O should have become 0


# ─── is_valid_plate ───────────────────────────────────────────────────────────

def test_is_valid_plate_normal():
    assert is_valid_plate("AB123") is True


def test_is_valid_plate_too_short():
    assert is_valid_plate("X") is False
    assert is_valid_plate("AB") is False


def test_is_valid_plate_too_long():
    assert is_valid_plate("A" * 20) is False
    assert is_valid_plate("A" * 13) is False


def test_is_valid_plate_exact_min():
    # Exactly min_len (3) characters → valid
    assert is_valid_plate("ABC") is True


def test_is_valid_plate_exact_max():
    # Exactly max_len (12) characters → valid
    assert is_valid_plate("A" * 12) is True


def test_is_valid_plate_one_over_max():
    assert is_valid_plate("A" * 13) is False


def test_is_valid_plate_with_spaces():
    # "AB 123" has 6 chars including the space; strip counts 6 → valid
    assert is_valid_plate("AB 123") is True


def test_is_valid_plate_whitespace_only():
    # All whitespace → stripped length 0, below min
    assert is_valid_plate("   ") is False


# ─── format_confidence ────────────────────────────────────────────────────────

def test_format_confidence_mid():
    assert format_confidence(0.856) == "85.6%"


def test_format_confidence_zero():
    assert format_confidence(0.0) == "0.0%"


def test_format_confidence_one():
    assert format_confidence(1.0) == "100.0%"


def test_format_confidence_rounding():
    # 0.12345 → 12.3%
    assert format_confidence(0.12345) == "12.3%"
