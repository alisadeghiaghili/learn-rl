"""Agent interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from learnrl.types import Action, Done, Reward, State, Transition


class Agent(ABC):
    """Policy that can optionally learn from transitions.

    Args:
        name: Short identifier shown in status output.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def act(self, state: State, *, explore: bool = True) -> Action:
        """Choose an action for ``state``.

        Args:
            state: Current state.
            explore: Whether exploration noise/epsilon is active.

        Returns:
            The selected action.
        """

    def learn(self, transition: Transition) -> dict[str, float]:
        """Update the agent from one transition.

        Args:
            transition: ``(s, a, r, s', done)`` experience.

        Returns:
            Diagnostics such as ``{"td_error": ...}``. Default is empty.
        """
        return {}

    def learn_episode(self, transitions: Sequence[Transition]) -> dict[str, float]:
        """Update from a full episode. Default applies :meth:`learn` per step.

        Args:
            transitions: Ordered experience from one episode.

        Returns:
            Aggregated diagnostics (last-step values win).
        """
        metrics: dict[str, float] = {}
        for t in transitions:
            metrics.update(self.learn(t))
        return metrics

    def reset_episode(self) -> None:
        """Clear any per-episode state (e.g. eligibility traces)."""

    def values(self, state: State) -> Mapping[Action, float]:
        """Return action-value estimates for ``state`` if available.

        Args:
            state: State to query.

        Returns:
            Mapping action → value. Empty if the agent has no Q-table.
        """
        return {}

    def state_value(self, state: State) -> float | None:
        """Return V(s) if defined (greedy over Q or a stored V table).

        Args:
            state: State to query.

        Returns:
            The value estimate, or ``None`` if unavailable.
        """
        q = self.values(state)
        if not q:
            return None
        return max(q.values())

    def policy(self, state: State) -> Action | None:
        """Return the greedy action for ``state`` if a policy exists.

        Args:
            state: State to query.

        Returns:
            Greedy action or ``None``.
        """
        q = self.values(state)
        if not q:
            return None
        return max(q.items(), key=lambda kv: kv[1])[0]

    def snapshot(self) -> dict[str, Any]:
        """Serializable-ish snapshot used by undo. Must be JSON-friendly.

        Returns:
            A deep-enough copy of learnable parameters and RNG state.
        """
        return {"name": self._name}

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        """Restore parameters from a :meth:`snapshot`.

        Args:
            snapshot: Previously captured snapshot.

        Raises:
            ValueError: If the snapshot does not match this agent type.
        """
        if snapshot.get("name") != self._name:
            raise ValueError("snapshot agent name mismatch")

    def spec(self) -> dict[str, Any]:
        """Serializable agent description for level/export files.

        Returns:
            JSON-friendly config.
        """
        return {"type": type(self).__name__, "name": self._name}


class PlanningAgent(Agent, ABC):
    """Agent that can run batch planning updates on a finite MDP."""

    @abstractmethod
    def plan(self, env: Any, *, iterations: int = 1) -> dict[str, float]:
        """Run planning iterations (e.g. value iteration sweeps).

        Args:
            env: A :class:`~learnrl.envs.base.FiniteMDP`.
            iterations: Number of sweeps/iterations.

        Returns:
            Diagnostics such as ``{"delta": ...}``.
        """
