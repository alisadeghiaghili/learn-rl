"""Tests for agents: bandit learning and DP correctness."""

from __future__ import annotations

import pytest

from learnrl.agents import (
    BanditAgent,
    PolicyIteration,
    QAgent,
    RandomAgent,
    ValueIteration,
    agent_from_spec,
)
from learnrl.envs import GridWorld
from learnrl.envs.gridworld import RIGHT


def test_q_agent_learn_moves_toward_target() -> None:
    agent = QAgent(["a", "b"], alpha=0.5, gamma=0.0, epsilon=0.0, mode="q", seed=0)
    before = agent.values("s")["a"]
    agent.learn(("s", "a", 1.0, "s2", True))
    after = agent.values("s")["a"]
    assert after == pytest.approx(before + 0.5 * (1.0 - before))


def test_value_iteration_on_corridor() -> None:
    env = GridWorld(grid=["S.G"], gamma=0.9, rewards={"step": -1, "goal": 10})
    agent = ValueIteration(gamma=0.9)
    agent.plan(env, iterations=50)
    v0 = agent.state_value((0, 0))
    # Optimal: one step of -1 then goal 10 from (0,1): V(0,1)= -1? wait
    # From (0,1): right -> 10, so Q = 10, V(0,1)=10 if no step cost on terminal
    # From (0,0): right -> -1 + 0.9*V(0,1)
    v1 = agent.state_value((0, 1))
    assert v1 is not None and v0 is not None
    assert v1 > v0
    assert v1 == pytest.approx(10.0, abs=1e-3)
    assert v0 == pytest.approx(-1 + 0.9 * 10, abs=1e-3)
    assert agent.policy((0, 0)) == RIGHT


def test_policy_iteration_stable() -> None:
    env = GridWorld(grid=["S#.", "...", "..G"], gamma=0.9, rewards={"step": -1, "goal": 10})
    agent = PolicyIteration(gamma=0.9)
    metrics = agent.plan(env, iterations=10)
    assert metrics["stable"] == 1.0
    assert agent.policy((0, 0)) in {"D", "U", "L", "R"}


def test_bandit_sample_average() -> None:
    agent = BanditAgent(2, epsilon=0.0, alpha=None, optimistic=0.0, seed=0)
    agent.learn(("s", 0, 1.0, "s", True))
    agent.learn(("s", 0, 0.0, "s", True))
    assert agent.values(0)[0] == pytest.approx(0.5)
    assert agent.counts() == [2, 0]


def test_random_agent_deterministic_with_seed() -> None:
    a = RandomAgent(["U", "D", "L", "R"], seed=42)
    b = RandomAgent(["U", "D", "L", "R"], seed=42)
    assert [a.act(0) for _ in range(5)] == [b.act(0) for _ in range(5)]


def test_agent_from_spec_names() -> None:
    assert agent_from_spec({"type": "q-learning"}, actions=["U"]).name == "q-learning"
    assert agent_from_spec({"type": "sarsa"}, actions=["U"]).name == "sarsa"
    assert agent_from_spec({"type": "double-q"}, actions=["U"]).name == "double-q"
    with pytest.raises(ValueError):
        agent_from_spec({"type": "nope"}, actions=["U"])


def test_q_snapshot_restore() -> None:
    agent = QAgent(["U", "D"], alpha=0.5, gamma=0.5, epsilon=0.1, seed=1)
    agent.learn(("s", "U", 1.0, "s2", False))
    snap = agent.snapshot()
    agent.learn(("s", "U", 1.0, "s2", False))
    agent.restore(snap)
    assert agent.values("s")["U"] == pytest.approx(0.5)
