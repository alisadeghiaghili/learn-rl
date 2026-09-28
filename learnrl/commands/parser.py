"""Command-line parsing for the LearnRL sandbox REPL."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True, slots=True)
class ParsedCommand:
    """One parsed user command line.

    Attributes:
        name: Canonical command name (lowercase).
        args: Positional arguments (strings).
        options: ``key=value`` options with normalized keys.
        raw: Original line as typed.
    """

    name: str
    args: tuple[str, ...] = ()
    options: dict[str, str] = field(default_factory=dict)
    raw: str = ""


_TOKEN_RE = re.compile(r'"([^"]*)"|\'([^\']*)\'|(\S+)')
_KV_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)=(.+)$")

_ALIASES = {
    "map": "show",
    "viz": "show",
    "ls": "levels",
    "h": "help",
    "?": "help",
    "exit": "quit",
    "e": "evaluate",
    "r": "reset",
    "u": "undo",
}


def parse_line(line: str) -> ParsedCommand | None:
    """Parse a single REPL line.

    Args:
        line: Raw user input.

    Returns:
        A :class:`ParsedCommand`, or ``None`` for blank/comment lines.

    Examples:
        >>> parse_line("train q 50 alpha=0.2").name
        'train'
        >>> parse_line("set alpha=0.1 epsilon=0.2").options
        {'alpha': '0.1', 'epsilon': '0.2'}
        >>> parse_line("# comment") is None
        True
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    tokens: list[str] = []
    for m in _TOKEN_RE.finditer(stripped):
        tokens.append(m.group(1) or m.group(2) or m.group(3))
    if not tokens:
        return None

    name = tokens[0].lower()
    name = _ALIASES.get(name, name)
    args: list[str] = []
    options: dict[str, str] = {}
    for tok in tokens[1:]:
        kv = _KV_RE.match(tok)
        if kv:
            options[kv.group(1).lower()] = kv.group(2)
        else:
            args.append(tok)

    # Shorthand: set a=0.1 g=0.9 e=0.2
    if name == "set":
        options.setdefault("alpha", options.pop("a", options.get("alpha", "")))
        if options.get("alpha") == "":
            options.pop("alpha", None)
        if "g" in options and "gamma" not in options:
            options["gamma"] = options.pop("g")
        if "e" in options and "epsilon" not in options:
            options["epsilon"] = options.pop("e")
        options = {k: v for k, v in options.items() if v != ""}

    return ParsedCommand(name=name, args=tuple(args), options=options, raw=stripped)


def parse_program(text: str) -> list[ParsedCommand]:
    """Parse a multi-command program (``;`` or newline separated).

    Args:
        text: Source text with one or more commands.

    Returns:
        Non-empty parsed commands in order.

    Examples:
        >>> [c.name for c in parse_program("show; step 2")]
        ['show', 'step']
    """
    lines: list[str] = []
    for chunk in text.replace(";", "\n").splitlines():
        lines.append(chunk)
    out: list[ParsedCommand] = []
    for line in lines:
        cmd = parse_line(line)
        if cmd is not None:
            out.append(cmd)
    return out


def option_float(options: dict[str, str], key: str, default: float | None = None) -> float | None:
    """Read a float option.

    Args:
        options: Parsed options.
        key: Option name.
        default: Returned when the option is absent or unparsable.

    Returns:
        The float value or ``default``.
    """
    if key not in options:
        return default
    try:
        return float(options[key])
    except ValueError:
        return default


def option_int(options: dict[str, str], key: str, default: int | None = None) -> int | None:
    """Read an int option.

    Args:
        options: Parsed options.
        key: Option name.
        default: Returned when the option is absent or unparsable.

    Returns:
        The int value or ``default``.
    """
    if key not in options:
        return default
    try:
        return int(float(options[key]))
    except ValueError:
        return default


def first_int(args: Sequence[str], default: int) -> int:
    """Return the first argument as int, or ``default``.

    Args:
        args: Positional arguments.
        default: Fallback value.

    Returns:
        Parsed integer.
    """
    if not args:
        return default
    try:
        return int(float(args[0]))
    except ValueError:
        return default
