"""Shared type aliases and small value objects for LearnRL."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Mapping, Sequence

State = Hashable
Action = Hashable
Reward = float
Done = bool
Probability = float

Transition = tuple[State, Action, Reward, State, Done]
"""A single environment transition ``(s, a, r, s', done)``."""


@dataclass(frozen=True, slots=True)
class StepResult:
    """Outcome of one environment step.

    Args:
        state: The state observed after the transition.
        reward: Immediate reward signal.
        done: Whether the episode terminated.
        info: Optional diagnostics from the environment.
        prev_state: State before the transition.
        action: Action applied.
    """

    state: State
    reward: Reward
    done: Done
    info: Mapping[str, Any] = field(default_factory=dict)
    prev_state: State | None = None
    action: Action | None = None


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Aggregated evaluation of a policy over one or more episodes.

    Args:
        episodes: Number of episodes run.
        mean_return: Average undiscounted return per episode.
        min_return: Worst episode return.
        max_return: Best episode return.
        mean_length: Average episode length in steps.
        returns: Full list of episode returns.
    """

    episodes: int
    mean_return: float
    min_return: float
    max_return: float
    mean_length: float
    returns: Sequence[float]


@dataclass(frozen=True, slots=True)
class GoalCheck:
    """Result of comparing session state against a level goal.

    Args:
        satisfied: Whether the goal is fully met.
        details: Human-readable per-criterion messages.
    """

    satisfied: bool
    details: tuple[str, ...] = ()
