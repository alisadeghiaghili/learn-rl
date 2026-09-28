"""Smoke tests for visualization renderers."""

from __future__ import annotations

from learnrl.commands import dispatch, parse_line
from learnrl.engine.session import default_session
from learnrl.viz import (
    render_curve,
    render_help,
    render_map,
    render_policy,
    render_q,
    render_status,
    render_values,
)


def test_renderers_do_not_crash() -> None:
    s = default_session(seed=0)
    assert "agent" in render_map(s).lower() or "A" in render_map(s)
    assert "V(s)" in render_values(s)
    assert "π" in render_policy(s) or "policy" in render_policy(s)
    assert "Q" in render_q(s) or "q" in render_q(s).lower()
    assert "learning curve" in render_curve(s)
    assert "env" in render_status(s)
    assert "commands" in render_help()


def test_train_then_render() -> None:
    s = default_session(seed=1)
    cmd = parse_line("train q 15")
    assert cmd is not None
    dispatch(s, cmd)
    out = render_q(s)
    assert "a" in out.lower()
    out = render_curve(s)
    assert "mean=" in out
