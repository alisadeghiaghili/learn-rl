"""Environment factory and public re-exports."""

from __future__ import annotations

from typing import Any, Mapping

from learnrl.envs.bandit import Bandit, bandit_from_spec
from learnrl.envs.base import Env, FiniteMDP, reward_fn_from_spec
from learnrl.envs.cliff import CliffWalk, cliff_from_spec
from learnrl.envs.gridworld import GridWorld, gridworld_from_spec

__all__ = [
    "Bandit",
    "CliffWalk",
    "Env",
    "FiniteMDP",
    "GridWorld",
    "bandit_from_spec",
    "cliff_from_spec",
    "env_from_spec",
    "gridworld_from_spec",
    "reward_fn_from_spec",
]

_FACTORIES = {
    "bandit": bandit_from_spec,
    "cliff": cliff_from_spec,
    "cliffwalk": cliff_from_spec,
    "gridworld": gridworld_from_spec,
}


def env_from_spec(spec: Mapping[str, Any]) -> Env:
    """Instantiate an environment from a serializable spec.

    Args:
        spec: Must include ``type`` in {``bandit``, ``cliff``, ``gridworld``}.
            Remaining keys are passed to the matching factory.

    Returns:
        A concrete :class:`~learnrl.envs.base.Env`.

    Raises:
        ValueError: If ``type`` is missing or unknown.

    Examples:
        >>> env = env_from_spec({"type": "gridworld", "grid": ["S.G"]})
        >>> env.name
        'gridworld'
    """
    kind = str(spec.get("type", "")).lower()
    factory = _FACTORIES.get(kind)
    if factory is None:
        known = ", ".join(sorted(_FACTORIES))
        raise ValueError(f"unknown env type {kind!r}; expected one of: {known}")
    return factory(spec)
