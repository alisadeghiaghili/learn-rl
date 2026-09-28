"""Tabular TD control and prediction agents."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from learnrl.agents.base import Agent
from learnrl.types import Action, State, Transition


class RandomAgent(Agent):
    """Uniform random policy over a fixed action set.

    Args:
        actions: Action space.
        seed: RNG seed for reproducible action choices.
        name: Display name.
    """

    def __init__(
        self,
        actions: Sequence[Action],
        *,
        seed: int = 0,
        name: str = "random",
    ) -> None:
        super().__init__(name)
        if not actions:
            raise ValueError("actions must be non-empty")
        self._actions = tuple(actions)
        self._seed = int(seed)
        self._rng = _Lcg(self._seed)

    def act(self, state: State, *, explore: bool = True) -> Action:
        return self._actions[self._rng.randrange(len(self._actions))]

    def snapshot(self) -> dict[str, Any]:
        return {"name": self.name, "type": "random", "seed": self._seed, "rng": self._rng.state}

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        if snapshot.get("type") != "random":
            raise ValueError("snapshot is not a RandomAgent")
        self._rng = _Lcg(0)
        self._rng.state = int(snapshot.get("rng", 1))

    def spec(self) -> dict[str, Any]:
        return {"type": "random", "name": self.name, "seed": self._seed}


class QAgent(Agent):
    """Epsilon-greedy tabular TD agent (SARSA / Q-learning / Expected SARSA).

    Args:
        actions: Action space.
        alpha: Step size.
        gamma: Discount factor.
        epsilon: Exploration rate for epsilon-greedy action selection.
        mode: One of ``q`` (Q-learning), ``sarsa``, ``expected``.
        double_q: If True, use Double Q-learning style decoupled estimates.
        name: Display name.

    Raises:
        ValueError: If ``mode`` is unknown or hyperparameters are out of range.
    """

    MODES = frozenset({"q", "sarsa", "expected"})

    def __init__(
        self,
        actions: Sequence[Action],
        *,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        mode: str = "q",
        double_q: bool = False,
        seed: int = 0,
        name: str | None = None,
    ) -> None:
        mode = mode.lower()
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {sorted(self.MODES)}")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError("gamma must be in [0, 1]")
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError("epsilon must be in [0, 1]")
        default_name = {
            "q": "double-q" if double_q else "q-learning",
            "sarsa": "sarsa",
            "expected": "expected-sarsa",
        }[mode]
        super().__init__(name or default_name)
        self._actions = tuple(actions)
        self._alpha = float(alpha)
        self._gamma = float(gamma)
        self._epsilon = float(epsilon)
        self._mode = mode
        self._double_q = bool(double_q)
        self._seed = int(seed)
        self._rng = _Lcg(self._seed)
        self._q: dict[tuple[State, Action], float] = {}
        self._q2: dict[tuple[State, Action], float] = {}
        self._next_action: Action | None = None

    @property
    def alpha(self) -> float:
        return self._alpha

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @property
    def mode(self) -> str:
        return self._mode

    def set_hyperparams(
        self,
        *,
        alpha: float | None = None,
        gamma: float | None = None,
        epsilon: float | None = None,
    ) -> None:
        """Update learning hyperparameters in place.

        Args:
            alpha: New step size in (0, 1], or None to keep.
            gamma: New discount in [0, 1], or None to keep.
            epsilon: New exploration rate in [0, 1], or None to keep.

        Raises:
            ValueError: If a provided value is out of range.
        """
        if alpha is not None:
            if not (0.0 < alpha <= 1.0):
                raise ValueError("alpha must be in (0, 1]")
            self._alpha = float(alpha)
        if gamma is not None:
            if not (0.0 <= gamma <= 1.0):
                raise ValueError("gamma must be in [0, 1]")
            self._gamma = float(gamma)
        if epsilon is not None:
            if not (0.0 <= epsilon <= 1.0):
                raise ValueError("epsilon must be in [0, 1]")
            self._epsilon = float(epsilon)

    def act(self, state: State, *, explore: bool = True) -> Action:
        if explore and self._rng.random() < self._epsilon:
            return self._actions[self._rng.randrange(len(self._actions))]
        return self._greedy(state)

    def learn(self, transition: Transition) -> dict[str, float]:
        s, a, r, s2, done = transition
        q = self._qsa(s, a)
        if done:
            target = r
        elif self._mode == "q":
            target = r + self._gamma * (self._double_max(s2) if self._double_q else self._max(s2))
        elif self._mode == "sarsa":
            a2 = self._next_action if self._next_action is not None else self._greedy(s2)
            target = r + self._gamma * self._qsa(s2, a2)
        else:  # expected
            target = r + self._gamma * self._expected(s2)
        td = target - q
        self._set_q(s, a, q + self._alpha * td)
        return {"td_error": td, "q": self._qsa(s, a)}

    def set_next_action(self, action: Action) -> None:
        """Inject the SARSA next action (on-policy bootstrap).

        Args:
            action: Action the agent will take from ``s'``.
        """
        self._next_action = action

    def reset_episode(self) -> None:
        self._next_action = None

    def values(self, state: State) -> Mapping[Action, float]:
        return {a: self._qsa(state, a) for a in self._actions}

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "q",
            "mode": self._mode,
            "double_q": self._double_q,
            "alpha": self._alpha,
            "gamma": self._gamma,
            "epsilon": self._epsilon,
            "seed": self._seed,
            "rng": self._rng.state,
            "q": {f"{s!r}|{a!r}": v for (s, a), v in self._q.items()},
            "q2": {f"{s!r}|{a!r}": v for (s, a), v in self._q2.items()},
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        if snapshot.get("type") != "q":
            raise ValueError("snapshot is not a QAgent")
        self._mode = str(snapshot["mode"])
        self._double_q = bool(snapshot["double_q"])
        self._alpha = float(snapshot["alpha"])
        self._gamma = float(snapshot["gamma"])
        self._epsilon = float(snapshot["epsilon"])
        self._rng = _Lcg(0)
        self._rng.state = int(snapshot.get("rng", 1))
        self._q = {_parse_key(k): float(v) for k, v in snapshot.get("q", {}).items()}
        self._q2 = {_parse_key(k): float(v) for k, v in snapshot.get("q2", {}).items()}

    def spec(self) -> dict[str, Any]:
        return {
            "type": "q",
            "name": self.name,
            "mode": self._mode,
            "double_q": self._double_q,
            "alpha": self._alpha,
            "gamma": self._gamma,
            "epsilon": self._epsilon,
            "seed": self._seed,
        }

    def q_table(self) -> dict[tuple[State, Action], float]:
        """Return a copy of the primary Q-table.

        Returns:
            Mapping ``(state, action) -> value``.
        """
        return dict(self._q)

    # -- internals -----------------------------------------------------

    def _greedy(self, state: State) -> Action:
        best_a = self._actions[0]
        best_v = self._qsa(state, best_a)
        for a in self._actions[1:]:
            v = self._qsa(state, a)
            if v > best_v + 1e-12:
                best_a, best_v = a, v
        return best_a

    def _max(self, state: State) -> float:
        return max(self._qsa(state, a) for a in self._actions)

    def _double_max(self, state: State) -> float:
        a_star = self._greedy(state)
        # Decouple selection (Q) and evaluation (Q2).
        return self._q2.get((state, a_star), 0.0)

    def _expected(self, state: State) -> float:
        # Epsilon-greedy expectation.
        n = len(self._actions)
        greedy = self._greedy(state)
        exp = 0.0
        for a in self._actions:
            p = self._epsilon / n + (1.0 - self._epsilon if a == greedy else 0.0)
            exp += p * self._qsa(state, a)
        return exp

    def _qsa(self, state: State, action: Action) -> float:
        return self._q.get((state, action), 0.0)

    def _set_q(self, state: State, action: Action, value: float) -> None:
        self._q[(state, action)] = value
        if self._double_q:
            self._q2[(state, action)] = value  # keep in sync for teaching clarity


class BanditAgent(Agent):
    """Epsilon-greedy sample-average (or constant-α) bandit algorithm.

    Args:
        n_arms: Number of arms.
        epsilon: Exploration rate.
        alpha: Constant step size; None means 1/n sample averages.
        optimistic: Initial Q values (optimistic initialization).
        seed: RNG seed.
        name: Display name.
    """

    def __init__(
        self,
        n_arms: int,
        *,
        epsilon: float = 0.1,
        alpha: float | None = None,
        optimistic: float = 0.0,
        seed: int = 0,
        name: str = "bandit-epsilon",
    ) -> None:
        super().__init__(name)
        if n_arms <= 0:
            raise ValueError("n_arms must be positive")
        self._n = int(n_arms)
        self._epsilon = float(epsilon)
        self._alpha = alpha
        self._optimistic = float(optimistic)
        self._seed = int(seed)
        self._rng = _Lcg(self._seed)
        self._q = [self._optimistic] * self._n
        self._counts = [0] * self._n

    def act(self, state: State, *, explore: bool = True) -> Action:
        if explore and self._rng.random() < self._epsilon:
            return self._rng.randrange(self._n)
        return self._greedy()

    def learn(self, transition: Transition) -> dict[str, float]:
        _, a, r, _, _ = transition
        if not isinstance(a, int) or not (0 <= a < self._n):
            return {}
        self._counts[a] += 1
        if self._alpha is None:
            step = 1.0 / self._counts[a]
        else:
            step = self._alpha
        self._q[a] += step * (float(r) - self._q[a])
        return {"q": self._q[a], "count": float(self._counts[a])}

    def values(self, state: State) -> Mapping[Action, float]:
        return {i: self._q[i] for i in range(self._n)}

    def counts(self) -> list[int]:
        return list(self._counts)

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "bandit",
            "n": self._n,
            "epsilon": self._epsilon,
            "alpha": self._alpha,
            "optimistic": self._optimistic,
            "seed": self._seed,
            "rng": self._rng.state,
            "q": list(self._q),
            "counts": list(self._counts),
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        if snapshot.get("type") != "bandit":
            raise ValueError("snapshot is not a BanditAgent")
        self._n = int(snapshot["n"])
        self._epsilon = float(snapshot["epsilon"])
        self._alpha = snapshot["alpha"]
        self._optimistic = float(snapshot["optimistic"])
        self._rng = _Lcg(0)
        self._rng.state = int(snapshot.get("rng", 1))
        self._q = [float(x) for x in snapshot["q"]]
        self._counts = [int(x) for x in snapshot["counts"]]

    def spec(self) -> dict[str, Any]:
        return {
            "type": "bandit-epsilon",
            "name": self.name,
            "n_arms": self._n,
            "epsilon": self._epsilon,
            "alpha": self._alpha,
            "optimistic": self._optimistic,
            "seed": self._seed,
        }

    def _greedy(self) -> Action:
        best = 0
        for i in range(1, self._n):
            if self._q[i] > self._q[best]:
                best = i
        return best


def _parse_key(key: str) -> tuple[Any, Any]:
    # Snapshot keys are "state_repr|action_repr".
    state_s, action_s = key.split("|", 1)
    return _literal(state_s), _literal(action_s)


def _literal(text: str) -> Any:
    import ast

    return ast.literal_eval(text)


class _Lcg:
    """Numerical Recipes LCG. Use high bits only — low bits have short periods."""

    __slots__ = ("state",)

    def __init__(self, seed: int) -> None:
        self.state = int(seed) & 0xFFFFFFFF or 1

    def random(self) -> float:
        return self.randrange(1 << 24) / float(1 << 24)

    def randrange(self, n: int) -> int:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        if n <= 0:
            return 0
        return (self.state >> 8) % n
