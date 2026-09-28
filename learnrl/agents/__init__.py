"""Agent factory and public re-exports."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from learnrl.agents.base import Agent, PlanningAgent
from learnrl.agents.dp import PolicyIteration, ValueIteration
from learnrl.agents.prediction import MonteCarloPrediction, NStepSARSA, TDPrediction
from learnrl.agents.tabular import BanditAgent, QAgent, RandomAgent
from learnrl.types import Action

__all__ = [
    "Agent",
    "BanditAgent",
    "MonteCarloPrediction",
    "NStepSARSA",
    "PlanningAgent",
    "PolicyIteration",
    "QAgent",
    "RandomAgent",
    "TDPrediction",
    "ValueIteration",
    "agent_from_spec",
]

_Q_MODES = {
    "q": "q",
    "q-learning": "q",
    "sarsa": "sarsa",
    "expected": "expected",
    "expected-sarsa": "expected",
    "double-q": "q",
    "double-q-learning": "q",
}


def agent_from_spec(
    spec: Mapping[str, Any],
    *,
    actions: Sequence[Action],
) -> Agent:
    """Instantiate an agent from a serializable spec.

    Args:
        spec: Mapping with ``type`` and algorithm-specific options.
        actions: Action space of the active environment.

    Returns:
        A concrete :class:`~learnrl.agents.base.Agent`.

    Raises:
        ValueError: If ``type`` is unknown.

    Examples:
        >>> agent = agent_from_spec({"type": "q-learning"}, actions=("U", "D"))
        >>> agent.name
        'q-learning'
    """
    kind = str(spec.get("type", "random")).lower()
    if kind in ("random", "uniform"):
        return RandomAgent(actions, seed=int(spec.get("seed", 0)), name=str(spec.get("name", "random")))
    if kind in _Q_MODES:
        mode = _Q_MODES[kind]
        double_q = "double" in kind
        return QAgent(
            actions,
            alpha=float(spec.get("alpha", 0.1)),
            gamma=float(spec.get("gamma", 0.9)),
            epsilon=float(spec.get("epsilon", 0.1)),
            mode=mode,
            double_q=double_q,
            seed=int(spec.get("seed", 0)),
            name=spec.get("name"),
        )
    if kind in ("bandit", "bandit-epsilon", "epsilon-greedy"):
        n = int(spec.get("n_arms", len(actions) or 1))
        return BanditAgent(
            n,
            epsilon=float(spec.get("epsilon", 0.1)),
            alpha=spec.get("alpha"),
            optimistic=float(spec.get("optimistic", 0.0)),
            seed=int(spec.get("seed", 0)),
            name=str(spec.get("name", "bandit-epsilon")),
        )
    if kind in ("value-iteration", "vi", "dp-vi"):
        return ValueIteration(
            gamma=spec.get("gamma"),
            theta=float(spec.get("theta", 1e-4)),
            name=str(spec.get("name", "value-iteration")),
        )
    if kind in ("policy-iteration", "pi", "dp-pi"):
        return PolicyIteration(
            gamma=spec.get("gamma"),
            eval_iterations=int(spec.get("eval_iterations", 50)),
            theta=float(spec.get("theta", 1e-4)),
            name=str(spec.get("name", "policy-iteration")),
        )
    if kind in ("mc", "mc-prediction", "monte-carlo"):
        return MonteCarloPrediction(
            actions,
            seed=int(spec.get("seed", 0)),
            name=str(spec.get("name", "mc-prediction")),
        )
    if kind in ("td", "td-prediction", "td0"):
        return TDPrediction(
            actions,
            alpha=float(spec.get("alpha", 0.1)),
            gamma=float(spec.get("gamma", 1.0)),
            seed=int(spec.get("seed", 0)),
            name=str(spec.get("name", "td-prediction")),
        )
    if kind in ("n-step-sarsa", "nstep-sarsa", "n-step"):
        return NStepSARSA(
            actions,
            n=int(spec.get("n", 3)),
            alpha=float(spec.get("alpha", 0.1)),
            gamma=float(spec.get("gamma", 0.9)),
            epsilon=float(spec.get("epsilon", 0.1)),
            seed=int(spec.get("seed", 0)),
            name=spec.get("name"),
        )
    raise ValueError(f"unknown agent type {kind!r}")
