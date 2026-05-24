"""Command-line parsing helpers shared by runnable scripts."""

from __future__ import annotations

import argparse
from typing import Optional


def str_to_bool(value) -> bool:
    """Parse common string representations of booleans."""
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_bool_arg(value) -> bool:
    """Alias for argparse options that historically used parse_bool_arg."""
    return str_to_bool(value)


def parse_position_arg(value):
    """Parse a position argument as an int or one of the supported symbolic values."""
    if value is None or isinstance(value, int):
        return value
    text = str(value).strip()
    lowered = text.lower()
    if lowered in {"none", "null", ""}:
        return None
    if lowered in {"all", "all_no_extra", "extra", "consistent", "last"}:
        return lowered
    return int(text)


def parse_optional_int(value) -> Optional[int]:
    """Parse an optional integer argument."""
    parsed = parse_position_arg(value)
    if parsed is None or isinstance(parsed, int):
        return parsed
    raise argparse.ArgumentTypeError(f"Expected an integer or None, got: {value}")


def parse_int_list(value):
    """Parse comma-separated integers; 'none' returns None."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    lowered = str(value).strip().lower()
    if lowered in {"none", "null", ""}:
        return None
    return [int(part.strip()) for part in str(value).split(",") if part.strip()]
