"""Tests for the command parser."""

from __future__ import annotations

from learnrl.commands.parser import first_int, option_float, parse_line, parse_program


def test_parse_simple() -> None:
    cmd = parse_line("train q 50")
    assert cmd is not None
    assert cmd.name == "train"
    assert cmd.args == ("q", "50")


def test_parse_options() -> None:
    cmd = parse_line("set alpha=0.1 epsilon=0.2")
    assert cmd is not None
    assert cmd.options == {"alpha": "0.1", "epsilon": "0.2"}


def test_parse_set_shorthand() -> None:
    cmd = parse_line("set a=0.3 g=0.5 e=0.1")
    assert cmd is not None
    assert cmd.options["alpha"] == "0.3"
    assert cmd.options["gamma"] == "0.5"
    assert cmd.options["epsilon"] == "0.1"


def test_parse_quoted() -> None:
    cmd = parse_line('build level "hello world"')
    assert cmd is not None
    assert cmd.args[0] == "level"
    assert cmd.args[1] == "hello world"


def test_parse_blank_and_comment() -> None:
    assert parse_line("") is None
    assert parse_line("   ") is None
    assert parse_line("# hi") is None


def test_parse_aliases() -> None:
    assert parse_line("map").name == "show"
    assert parse_line("exit").name == "quit"
    assert parse_line("?").name == "help"


def test_parse_program() -> None:
    cmds = parse_program("show; step 2\n# skip\nvalues")
    assert [c.name for c in cmds] == ["show", "step", "values"]
    assert cmds[1].args == ("2",)


def test_helpers() -> None:
    assert first_int(["7"], 1) == 7
    assert first_int([], 3) == 3
    assert option_float({"a": "0.25"}, "a") == 0.25
    assert option_float({}, "a", 1.5) == 1.5
