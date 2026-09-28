"""Tests for Monte Carlo, TD prediction, and n-step SARSA."""

from __future__ import annotations

import pytest

from learnrl.agents import MonteCarloPrediction, NStepSARSA, TDPrediction, agent_from_spec
from learnrl.commands import dispatch, parse_line
from learnrl.engine.session import Session
from learnrl.envs import env_from_spec


def test_mc_prediction_averages_returns() -> None:
    agent = MonteCarloPrediction(["U", "D"], seed=0)
    # Single-step episode: reward 1 → V(s)=1
    agent.reset_episode()
    agent.learn(("s", "U", 1.0, "s2", True))
    assert agent.state_value("s") == pytest.approx(1.0)
    # Second episode: reward 3 → V(s)=(1+3)/2=2
    agent.reset_episode()
    agent.learn(("s", "U", 3.0, "s2", True))
    assert agent.state_value("s") == pytest.approx(2.0)


def test_td_prediction_bootstraps() -> None:
    agent = TDPrediction(["U"], alpha=1.0, gamma=0.5, seed=0)
    agent.learn(("s", "U", 1.0, "s2", False))  # V(s2)=0 → target=1
    assert agent.state_value("s") == pytest.approx(1.0)
    agent.learn(("s2", "U", 0.0, "s3", True))  # terminal, target=0 → V(s2)=0
    agent.learn(("s", "U", 0.0, "s2", False))  # target = 0 + 0.5*0 = 0
    assert agent.state_value("s") == pytest.approx(0.0)


def test_nstep_sarsa_n1_matches_sarsa_shape() -> None:
    agent = NStepSARSA(["U", "D"], n=1, alpha=0.5, gamma=0.0, epsilon=0.0, seed=0)
    agent.reset_episode()
    agent.learn(("s", "U", 1.0, "s2", True))
    # γ=0 → target is just r=1; α=0.5 from 0 → 0.5
    assert agent.values("s")["U"] == pytest.approx(0.5)


def test_nstep_uses_bootstrap() -> None:
    agent = NStepSARSA(["U", "D", "R"], n=2, alpha=1.0, gamma=1.0, epsilon=0.0, seed=0)
    agent._q[("s3", "R")] = 10.0  # noqa: SLF001  — A_{t+n} bootstrap value
    agent.reset_episode()
    agent.learn(("s", "U", 1.0, "s2", False))
    agent.learn(("s2", "D", 1.0, "s3", False))
    # n-step return: r1 + γ r2 + γ² Q(s3, A3) = 1 + 1 + 10 = 12
    metrics = agent.learn(("s3", "R", 0.0, "s4", False))
    assert metrics.get("td_error") == pytest.approx(12.0)
    assert agent.values("s")["U"] == pytest.approx(12.0)


def test_agent_from_spec_prediction() -> None:
    assert agent_from_spec({"type": "mc"}, actions=["U"]).name == "mc-prediction"
    assert agent_from_spec({"type": "td"}, actions=["U"]).name == "td-prediction"
    assert agent_from_spec({"type": "n-step-sarsa", "n": 4}, actions=["U"]).name == "4-step-sarsa"


def test_train_mc_on_corridor() -> None:
    env = env_from_spec(
        {"type": "gridworld", "grid": ["S.G"], "gamma": 1.0, "rewards": {"step": -1, "goal": 10}}
    )
    agent = agent_from_spec({"type": "mc", "seed": 3}, actions=list(env.actions))
    s = Session(env, agent)
    s.train(algorithm=None, iterations=200)
    v0 = agent.state_value((0, 0))
    # Under random policy on 3-cell corridor, V is finite and usually positive-ish
    assert v0 is not None


def test_level_catalog_has_prediction_and_cliff_sarsa() -> None:
    from learnrl.levels import all_levels

    ids = {lv.id for lv in all_levels()}
    assert "prediction.mc" in ids
    assert "prediction.td" in ids
    assert "prediction.nstep" in ids
    assert "advanced.cliff_sarsa" in ids


def test_dispatch_train_mc() -> None:
    env = env_from_spec({"type": "gridworld", "grid": ["S.G"], "gamma": 1.0})
    agent = agent_from_spec({"type": "random"}, actions=list(env.actions))
    s = Session(env, agent)
    result = dispatch(s, parse_line("train mc 30"))
    assert result.ok
    result = dispatch(s, parse_line("values"))
    assert "V(s)" in result.output
