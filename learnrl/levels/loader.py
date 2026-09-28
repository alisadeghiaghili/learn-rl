"""Level loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from learnrl.levels.catalog import (
    SEQUENCE_INFO,
    all_levels,
    get_level,
    iter_with_defaults,
    levels_in_sequence,
    register_custom,
    sequences,
)
from learnrl.levels.schema import Level

__all__ = [
    "Level",
    "SEQUENCE_INFO",
    "all_levels",
    "get_level",
    "iter_with_defaults",
    "level_from_json",
    "levels_in_sequence",
    "register_custom",
    "sequences",
]


def level_from_json(text: str) -> Level:
    """Parse a level definition from JSON text.

    Args:
        text: JSON object matching :meth:`Level.to_dict`.

    Returns:
        The parsed level (not yet registered).

    Raises:
        ValueError: If the JSON is invalid or missing required fields.

    Examples:
        >>> lv = level_from_json('{"id": "x.y", "name": "X", "env_spec": {"type": "gridworld"}}')
        >>> lv.id
        'x.y'
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid level JSON: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ValueError("level JSON must be an object")
    return Level.from_dict(data)


def load_level_file(path: str | Path) -> Level:
    """Load and parse a level from a file path.

    Args:
        path: Path to a ``.json`` level file.

    Returns:
        The parsed level.

    Raises:
        ValueError: If the file cannot be read or parsed.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read level file: {exc}") from exc
    return level_from_json(text)


def level_to_session_fields(level: Level) -> dict[str, Any]:
    """Extract env/agent specs for session construction.

    Args:
        level: Source level.

    Returns:
        Dict with ``env_spec`` and ``agent_spec``.
    """
    return {"env_spec": dict(level.env_spec), "agent_spec": dict(level.agent_spec)}
