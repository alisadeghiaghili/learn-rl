"""Undo/redo history of world snapshots."""

from __future__ import annotations

from typing import Any, Callable


class History:
    """Linear undo stack of immutable snapshots.

    Each successful mutating command pushes a snapshot of the world *before*
    the change. ``undo`` pops and restores. ``redo`` is intentionally omitted
    (matching LearnGitBranching's mental model of a single-step undo).

    Args:
        max_depth: Maximum snapshots retained (oldest dropped first).
    """

    def __init__(self, *, max_depth: int = 50) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be >= 1")
        self._max_depth = max_depth
        self._stack: list[Any] = []

    def push(self, snapshot: Any) -> None:
        """Record a pre-change snapshot.

        Args:
            snapshot: Opaque state captured by the session.
        """
        self._stack.append(snapshot)
        if len(self._stack) > self._max_depth:
            self._stack.pop(0)

    def undo(self, restore: Callable[[Any], None]) -> bool:
        """Restore the most recent snapshot.

        Args:
            restore: Callback that applies a snapshot to the live session.

        Returns:
            ``True`` if a snapshot was restored, ``False`` if the stack is empty.
        """
        if not self._stack:
            return False
        restore(self._stack.pop())
        return True

    def clear(self) -> None:
        """Drop all snapshots."""
        self._stack.clear()

    def __len__(self) -> int:
        return len(self._stack)
