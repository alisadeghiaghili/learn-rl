"""ANSI color helpers (optional, degrades to plain text)."""

from __future__ import annotations

import os
import sys


def color_enabled() -> bool:
    """Whether ANSI colors should be emitted.

    Returns:
        ``True`` when stdout is a TTY and ``NO_COLOR`` is unset.
    """
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def paint(text: str, code: str, *, enabled: bool | None = None) -> str:
    """Wrap ``text`` in an ANSI SGR code.

    Args:
        text: Content to colorize.
        code: SGR parameters, e.g. ``"36"`` or ``"1;32"``.
        enabled: Force on/off; default uses :func:`color_enabled`.

    Returns:
        Colorized string, or the original text when color is disabled.
    """
    if enabled is False or (enabled is None and not color_enabled()):
        return text
    return f"\x1b[{code}m{text}\x1b[0m"


# Common styles
DIM = "2"
BOLD = "1"
RED = "31"
GREEN = "32"
YELLOW = "33"
BLUE = "34"
MAGENTA = "35"
CYAN = "36"
GRAY = "90"
