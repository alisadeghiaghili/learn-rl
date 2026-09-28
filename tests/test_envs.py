"""Tests for environment dynamics."""

from __future__ import annotations

import pytest

from learnrl.envs import Bandit, CliffWalk, GridWorld, env_from_spec, gridworld_from_spec
from learnrl.envs.gridworld import DOWN, LEFT, RIGHT, UP


def test_gridworld_basic_step() -> None:
    env = GridWorld(grid=["S.G"], seed=0)
    s = env.reset(seed=0)
    assert s == (0, 0)
    result = env.step(RIGHT)
    assert result.state == (0, 1)
    assert result.reward == -1.0
    assert not result.done
    result = env.step(RIGHT)
    assert result.state == (0, 2)
    assert result.done
    assert result.reward == 10.0


def test_gridworld_wall_bump() -> None:
    env = GridWorld(grid=["S#G"], rewards={"wall": -2.0}, seed=0)
    env.reset(seed=0)
    result = env.step(RIGHT)
    assert result.state == (0, 0)
    assert result.reward == -2.0
    assert not result.done


def test_gridworld_invalid_action() -> None:
    env = GridWorld()
    env.reset()
    with pytest.raises(ValueError, match="invalid action"):
        env.step("X")


def test_gridworld_step_after_done() -> None:
    env = GridWorld(grid=["SG"], seed=0)
    env.reset(seed=0)
    env.step(RIGHT)
    with pytest.raises(RuntimeError, match="already done"):
        env.step(LEFT)


def test_transition_probabilities_sum_to_one() -> None:
    env = GridWorld(grid=["S.", ".G"], slip_prob=0.2, seed=0)
    for s in env.states:
        if env.is_terminal(s):
            continue
        for a in env.actions:
            dist = env.transition(s, a)
            assert abs(sum(p for p, *_ in dist) - 1.0) < 1e-9


def test_cliff_defaults() -> None:
    env = CliffWalk()
    assert len(env.states) == 27
    assert env.is_terminal((2, 0))


def test_bandit_pull() -> None:
    env = Bandit(means=(0.0, 1.0), noise_std=0.0, seed=1)
    env.reset(seed=1)
    result = env.step(1)
    assert result.done
    assert result.reward == pytest.approx(1.0)
    assert env.optimal_action() == 1


def test_env_from_spec_roundtrip() -> None:
    env = gridworld_from_spec({"type": "gridworld", "grid": ["S.G"], "seed": 3})
    spec = env.spec()
    env2 = env_from_spec(spec)
    assert isinstance(env2, GridWorld)
    assert env2.grid == env.grid


def test_env_from_spec_unknown() -> None:
    with pytest.raises(ValueError, match="unknown env type"):
        env_from_spec({"type": "nope"})
