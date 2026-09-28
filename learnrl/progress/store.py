"""Persistent progress store (solved levels) under the user home directory."""

from __future__ import annotations

import json
import os
from pathlib import Path


def progress_path() -> Path:
    """Return the on-disk progress file path.

    Returns:
        Path to ``learnrl_progress.json`` in :func:`default_progress_dir`.
    """
    return default_progress_dir() / "learnrl_progress.json"


def default_progress_dir() -> Path:
    """Return the directory used for user progress.

    Honors ``LEARNRL_HOME`` when set, else ``~/.learnrl``.

    Returns:
        The directory (not necessarily created yet).
    """
    override = os.environ.get("LEARNRL_HOME")
    if override:
        return Path(override)
    return Path.home() / ".learnrl"


def load_solved() -> set[str]:
    """Load the set of solved level ids from disk.

    Returns:
        Solved ids; empty set when missing or corrupt.
    """
    path = progress_path()
    if not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    if isinstance(data, dict):
        ids = data.get("solved", [])
    else:
        ids = data
    return {str(x) for x in ids} if isinstance(ids, list) else set()


def save_solved(ids: set[str] | frozenset[str]) -> Path:
    """Persist solved level ids.

    Args:
        ids: Level identifiers to record.

    Returns:
        Path written.
    """
    path = progress_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"solved": sorted(ids)}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def mark_solved(level_id: str) -> None:
    """Record one level id as solved on disk.

    Args:
        level_id: Level identifier such as ``intro.step``.
    """
    ids = load_solved()
    ids.add(level_id)
    save_solved(ids)
