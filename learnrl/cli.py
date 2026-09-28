"""LearnRL command-line entrypoint and REPL."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from learnrl.commands import dispatch, parse_line, parse_program
from learnrl.commands.handlers import check_and_record
from learnrl.engine.session import Session, default_session
from learnrl.viz import render_map, render_status
from learnrl.viz.color import BOLD, CYAN, paint

BANNER = """\
LearnRL — interactive reinforcement learning sandbox
Type `help` for commands, `levels` for the curriculum, `show` to see the world.
Inspired by LearnGitBranching.
"""


def run_repl(session: Session, *, intro: str = BANNER) -> int:
    """Run the interactive sandbox loop.

    Args:
        session: World to manipulate.
        intro: Banner printed on entry.

    Returns:
        Process exit code (0 on clean quit).
    """
    print(intro)
    print(render_map(session))
    while True:
        try:
            line = input(paint("learnrl> ", BOLD + ";" + CYAN))
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return 0
        cmd = parse_line(line)
        if cmd is None:
            continue
        result = dispatch(session, cmd)
        if result.output:
            print(result.output)
        if result.redraw:
            print(render_map(session))
        if session.level is not None and cmd.name in {
            "step",
            "train",
            "evaluate",
            "reset",
            "undo",
            "set",
        }:
            if check_and_record(session):
                pass
        if result.quit:
            return 0


def run_program(session: Session, program: str) -> int:
    """Execute a non-interactive command program.

    Used for demos, tests, and permalinks (LGB ``?command=`` analogue).

    Args:
        session: World to manipulate.
        program: Newline- or ``;``-separated commands.

    Returns:
        Exit code (1 if any command failed).
    """
    failed = 0
    for cmd in parse_program(program):
        result = dispatch(session, cmd)
        if result.output:
            print(result.output)
        if result.redraw:
            print(render_map(session))
        if not result.ok:
            failed += 1
        if session.level is not None and cmd.name in {
            "step",
            "train",
            "evaluate",
            "reset",
            "undo",
            "set",
        }:
            check_and_record(session)
        if result.quit:
            break
    print(render_status(session))
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    p = argparse.ArgumentParser(
        prog="learnrl",
        description="Interactive reinforcement learning tutorial and sandbox.",
    )
    p.add_argument(
        "-c",
        "--command",
        help="Run a command program (semicolon/newline separated) instead of the REPL.",
    )
    p.add_argument(
        "-l",
        "--level",
        help="Load a level id on startup (e.g. intro.step).",
    )
    p.add_argument(
        "-e",
        "--env",
        default="gridworld",
        choices=("gridworld", "cliff", "bandit"),
        help="Sandbox environment when no level is loaded.",
    )
    p.add_argument("--seed", type=int, default=0, help="Default RNG seed.")
    p.add_argument("--version", action="store_true", help="Print version and exit.")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.version:
        from learnrl import __version__

        print(__version__)
        return 0

    session = default_session(env_name=args.env, seed=args.seed)

    if args.level:
        from learnrl.commands.handlers import load_level_into_session
        from learnrl.levels import get_level

        lv = get_level(args.level)
        if lv is None:
            print(f"unknown level: {args.level!r}", file=sys.stderr)
            return 2
        result = load_level_into_session(session, lv)
        if result.output:
            print(result.output)

    if args.command:
        return run_program(session, args.command)
    return run_repl(session)


if __name__ == "__main__":
    raise SystemExit(main())
