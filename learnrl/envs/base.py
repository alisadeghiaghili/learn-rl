"""Environment interfaces and shared helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

from learnrl.types import Action, Done, Reward, State, StepResult


class Env(ABC):
    """Minimal environment interface used by agents, viz, and levels.

    Implementations must be deterministic given an explicit ``seed`` so that
    levels can reproduce trajectories for tests and goal checks.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in status output and level specs."""

    @property
    @abstractmethod
    def states(self) -> Sequence[State]:
        """All states in the state space (finite MDPs only)."""

    @property
    @abstractmethod
    def actions(self) -> Sequence[Action]:
        """All actions available to the agent."""

    @abstractmethod
    def reset(self, *, seed: int | None = None) -> State:
        """Start a new episode.

        Args:
            seed: Optional RNG seed for this episode. Omit to continue the
                environment's internal stream.

        Returns:
            The initial observation/state.
        """

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        """Apply ``action`` and advance the environment one step.

        Args:
            action: Action chosen by the agent.

        Returns:
            A :class:`~learnrl.types.StepResult` with the next state.

        Raises:
            ValueError: If ``action`` is not in :attr:`actions`.
            RuntimeError: If the episode is already done.
        """

    @abstractmethod
    def is_terminal(self, state: State) -> bool:
        """Return whether ``state`` ends an episode."""

    def spec(self) -> dict[str, Any]:
        """Serializable description of this environment for level files.

        Returns:
            A JSON-friendly dict. Subclasses should extend the base mapping.
        """
        return {"type": type(self).__name__, "name": self.name}

    def state_label(self, state: State) -> str:
        """Human-readable label used in visualizations.

        Args:
            state: State to label.

        Returns:
            A short string such as ``"(1, 2)"``.
        """
        return str(state)

    def action_label(self, action: Action) -> str:
        """Human-readable action glyph/label.

        Args:
            action: Action to label.

        Returns:
            A short string such as ``"↑"``.
        """
        return str(action)


class FiniteMDP(Env, ABC):
    """Finite MDP with an explicit transition model.

    Subclasses expose ``transition(s, a) -> list[(prob, reward, s', done)]``
    so dynamic-programming agents can plan without sampling.
    """

    @abstractmethod
    def transition(
        self, state: State, action: Action
    ) -> Sequence[tuple[float, Reward, State, Done]]:
        """Return the distribution P(s', r, done | s, a).

        Args:
            state: Current state.
            action: Action taken.

        Returns:
            Sequence of ``(probability, reward, next_state, done)`` tuples
            whose probabilities sum to approximately 1.0.

        Raises:
            ValueError: If ``state`` or ``action`` is invalid.
        """

    def sample_transition(
        self,
        state: State,
        action: Action,
        rng: Any,
    ) -> tuple[Reward, State, Done]:
        """Sample one outcome from :meth:`transition`.

        Args:
            state: Current state.
            action: Action taken.
            rng: Object exposing ``random()`` in [0, 1).

        Returns:
            ``(reward, next_state, done)``.
        """
        dist = self.transition(state, action)
        x = rng.random()
        acc = 0.0
        for prob, reward, next_state, done in dist:
            acc += prob
            if x <= acc:
                return reward, next_state, done
        # Numerical edge: return the last outcome.
        _, reward, next_state, done = dist[-1]
        return reward, next_state, done


def reward_fn_from_spec(spec: Mapping[str, Any]) -> dict[str, float]:
    """Build a named reward table from a level/env spec fragment.

    Args:
        spec: Mapping of reward names to numeric values. Missing keys fall
            back to sensible gridworld defaults.

    Returns:
        A dict with keys ``step``, ``goal``, ``pit``, and ``wall``.

    Examples:
        >>> reward_fn_from_spec({"step": -1, "goal": 10})
        {'step': -1.0, 'goal': 10.0, 'pit': -10.0, 'wall': -1.0}
    """
    defaults = {"step": -1.0, "goal": 10.0, "pit": -10.0, "wall": -1.0}
    merged = {**defaults, **{k: float(v) for k, v in spec.items()}}
    return {
        "step": merged["step"],
        "goal": merged["goal"],
        "pit": merged["pit"],
        "wall": merged["wall"],
    }
