"""Cliff-walking gridworld (Sutton & Barto Figure 6.4)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from learnrl.envs.gridworld import ACTIONS, Action, GridWorld
from learnrl.types import State

DEFAULT_CLIFF = (
    "S.......G",
    ".........",
    "PPPPPPPPP",
)


class CliffWalk(GridWorld):
    """Cliff-walk MDP: step along the bottom edge of the map and fall.

    The default map is 3x9 with start top-left, goal top-right, and a row of
    pits along the bottom. Stepping into a pit gives a large negative reward
    and terminates. Stochastic slips are disabled so algorithms can be
    compared cleanly.

    Args:
        grid: Map rows (see :class:`~learnrl.envs.gridworld.GridWorld`).
        rewards: Named reward overrides.
        gamma: Discount factor.
        seed: RNG seed.
        name: Display name.
    """

    def __init__(
        self,
        grid: Sequence[str] = DEFAULT_CLIFF,
        *,
        rewards: Mapping[str, float] | None = None,
        gamma: float = 1.0,
        seed: int = 0,
        name: str = "cliff",
    ) -> None:
        defaults = {"step": -1.0, "goal": -1.0, "pit": -100.0, "wall": -1.0}
        merged = {**defaults, **(rewards or {})}
        super().__init__(
            grid=grid,
            rewards=merged,
            gamma=gamma,
            slip_prob=0.0,
            seed=seed,
            name=name,
        )


def cliff_from_spec(spec: Mapping[str, Any]) -> CliffWalk:
    """Build a :class:`CliffWalk` from a serializable env spec.

    Args:
        spec: Mapping with optional ``grid``, ``rewards``, ``gamma``,
            ``seed``, ``name``.

    Returns:
        A configured cliff-walk environment.

    Examples:
        >>> env = cliff_from_spec({})
        >>> len(env.states)
        27
    """
    return CliffWalk(
        grid=tuple(spec.get("grid", DEFAULT_CLIFF)),
        rewards=spec.get("rewards"),
        gamma=float(spec.get("gamma", 1.0)),
        seed=int(spec.get("seed", 0)),
        name=str(spec.get("name", "cliff")),
    )


__all__ = ["CliffWalk", "DEFAULT_CLIFF", "cliff_from_spec", "ACTIONS", "Action", "GridWorld", "State"]
