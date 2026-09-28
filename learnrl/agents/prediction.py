"""Monte Carlo and TD value-prediction agents (state-value estimation)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from learnrl.agents.base import Agent
from learnrl.types import Action, State, Transition


class PredictionAgent(Agent):
    """Base class for agents that estimate V(s) under a fixed behavior policy.

    These agents are for the *prediction* problem: estimate the value of the
    policy they follow. They do not optimize the policy (see QAgent for control).

    Args:
        actions: Action space of the environment.
        name: Display name.
    """

    def __init__(self, actions: Sequence[Action], *, name: str) -> None:
        super().__init__(name)
        self._actions = tuple(actions)
        self._v: dict[State, float] = {}
        self._counts: dict[State, int] = {}

    def act(self, state: State, *, explore: bool = True) -> Action:
        """Pick an action under the fixed behavior policy.

        The default behavior is uniform random. Subclasses may override.
        """
        import random as _random

        return self._actions[_random.randrange(len(self._actions))]

    def state_value(self, state: State) -> float | None:
        return self._v.get(state, 0.0 if self._v else None)

    def v_table(self) -> dict[State, float]:
        return dict(self._v)

    def values(self, state: State) -> Mapping[Action, float]:
        v = self._v.get(state, 0.0)
        n = max(len(self._actions), 1)
        return {a: v for a in self._actions}


class MonteCarloPrediction(PredictionAgent):
    """Every-visit Monte Carlo prediction of V^π.

    After each complete episode, every visited state's value is updated toward
    the empirical mean return-to-go:

        V(s) ← V(s) + (1/N(s)) [G_t − V(s)]

    Args:
        actions: Action space (behavior policy is uniform random unless seeded).
        seed: RNG seed for action selection.
        name: Display name.
    """

    def __init__(
        self,
        actions: Sequence[Action],
        *,
        seed: int = 0,
        name: str = "mc-prediction",
    ) -> None:
        super().__init__(actions, name=name)
        self._seed = int(seed)
        self._rng = _Lcg(self._seed)
        self._episode: list[Transition] = []

    def act(self, state: State, *, explore: bool = True) -> Action:
        return self._actions[self._rng.randrange(len(self._actions))]

    def reset_episode(self) -> None:
        self._episode = []

    def learn(self, transition: Transition) -> dict[str, float]:
        self._episode.append(transition)
        if transition[4]:  # done
            return self._update_from_episode()
        return {}

    def learn_episode(self, transitions: Sequence[Transition]) -> dict[str, float]:
        self._episode = list(transitions)
        return self._update_from_episode()

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "mc-prediction",
            "seed": self._seed,
            "rng": self._rng.state,
            "v": {repr(k): v for k, v in self._v.items()},
            "counts": {repr(k): v for k, v in self._counts.items()},
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        if snapshot.get("type") != "mc-prediction":
            raise ValueError("snapshot is not MonteCarloPrediction")
        self._rng = _Lcg(0)
        self._rng.state = int(snapshot.get("rng", 1))
        self._v = {_lit(k): float(v) for k, v in snapshot.get("v", {}).items()}
        self._counts = {_lit(k): int(v) for k, v in snapshot.get("counts", {}).items()}

    def spec(self) -> dict[str, Any]:
        return {"type": "mc-prediction", "name": self.name, "seed": self._seed}

    def _update_from_episode(self) -> dict[str, float]:
        transitions = self._episode
        if not transitions:
            return {}
        # Compute return-to-go backwards (γ=1 for MC prediction by default).
        gamma = 1.0
        g = 0.0
        last_state: State | None = None
        for s, _a, r, s2, done in reversed(transitions):
            del s2, done
            g = r + gamma * g
            if s is None:
                continue
            self._counts[s] = self._counts.get(s, 0) + 1
            n = self._counts[s]
            old = self._v.get(s, 0.0)
            self._v[s] = old + (g - old) / n
            last_state = s
        self._episode = []
        return {"v": self._v.get(last_state, 0.0) if last_state is not None else 0.0, "episodes": 1.0}


class TDPrediction(PredictionAgent):
    """TD(0) prediction of V^π (one-step temporal difference).

    Update after every step:

        V(s) ← V(s) + α [r + γ V(s') − V(s)]

    Args:
        actions: Action space.
        alpha: Step size.
        gamma: Discount factor.
        seed: RNG seed for the behavior policy.
        name: Display name.
    """

    def __init__(
        self,
        actions: Sequence[Action],
        *,
        alpha: float = 0.1,
        gamma: float = 1.0,
        seed: int = 0,
        name: str = "td-prediction",
    ) -> None:
        super().__init__(actions, name=name)
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        self._alpha = float(alpha)
        self._gamma = float(gamma)
        self._seed = int(seed)
        self._rng = _Lcg(self._seed)

    def act(self, state: State, *, explore: bool = True) -> Action:
        return self._actions[self._rng.randrange(len(self._actions))]

    def learn(self, transition: Transition) -> dict[str, float]:
        s, _a, r, s2, done = transition
        if s is None:
            return {}
        old = self._v.get(s, 0.0)
        boot = 0.0 if done else self._v.get(s2, 0.0)
        target = r + self._gamma * boot
        td = target - old
        self._v[s] = old + self._alpha * td
        self._counts[s] = self._counts.get(s, 0) + 1
        return {"td_error": td, "v": self._v[s]}

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "td-prediction",
            "alpha": self._alpha,
            "gamma": self._gamma,
            "seed": self._seed,
            "rng": self._rng.state,
            "v": {repr(k): v for k, v in self._v.items()},
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        if snapshot.get("type") != "td-prediction":
            raise ValueError("snapshot is not TDPrediction")
        self._alpha = float(snapshot["alpha"])
        self._gamma = float(snapshot["gamma"])
        self._rng = _Lcg(0)
        self._rng.state = int(snapshot.get("rng", 1))
        self._v = {_lit(k): float(v) for k, v in snapshot.get("v", {}).items()}

    def spec(self) -> dict[str, Any]:
        return {
            "type": "td-prediction",
            "name": self.name,
            "alpha": self._alpha,
            "gamma": self._gamma,
            "seed": self._seed,
        }


class NStepSARSA(Agent):
    """n-step SARSA control agent (Sutton & Barto §7.2).

    Buffers the last n transitions and updates Q(s_t, a_t) toward the n-step
    return:

        G_{t:t+n} = r_{t+1} + γ r_{t+2} + ... + γ^{n-1} r_{t+n} + γ^n Q(s_{t+n}, a_{t+n})

    Args:
        actions: Action space.
        n: Bootstrapping horizon (1 is ordinary SARSA).
        alpha: Step size.
        gamma: Discount factor.
        epsilon: ε-greedy exploration rate.
        seed: RNG seed.
        name: Display name.
    """

    def __init__(
        self,
        actions: Sequence[Action],
        *,
        n: int = 3,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        seed: int = 0,
        name: str | None = None,
    ) -> None:
        if n < 1:
            raise ValueError("n must be >= 1")
        super().__init__(name or f"{n}-step-sarsa")
        self._actions = tuple(actions)
        self._n = int(n)
        self._alpha = float(alpha)
        self._gamma = float(gamma)
        self._epsilon = float(epsilon)
        self._seed = int(seed)
        self._rng = _Lcg(self._seed)
        self._q: dict[tuple[State, Action], float] = {}
        self._buf: list[Transition] = []

    def act(self, state: State, *, explore: bool = True) -> Action:
        if explore and self._rng.random() < self._epsilon:
            return self._actions[self._rng.randrange(len(self._actions))]
        return self._greedy(state)

    def reset_episode(self) -> None:
        self._buf = []

    def learn(self, transition: Transition) -> dict[str, float]:
        self._buf.append(transition)
        metrics: dict[str, float] = {}
        # True n-step SARSA needs A_{t+n} (action at S_{t+n}), so we wait for
        # the next transition that starts at S_{t+n}. On terminal, flush.
        if transition[4]:
            while self._buf:
                metrics = self._update(flush=True) or metrics
            return metrics
        if len(self._buf) > self._n:
            metrics = self._update(flush=False)
        return metrics

    def values(self, state: State) -> Mapping[Action, float]:
        return {a: self._q.get((state, a), 0.0) for a in self._actions}

    def q_table(self) -> dict[tuple[State, Action], float]:
        return dict(self._q)

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "n-step-sarsa",
            "n": self._n,
            "alpha": self._alpha,
            "gamma": self._gamma,
            "epsilon": self._epsilon,
            "seed": self._seed,
            "rng": self._rng.state,
            "q": {f"{s!r}|{a!r}": v for (s, a), v in self._q.items()},
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        if snapshot.get("type") != "n-step-sarsa":
            raise ValueError("snapshot is not NStepSARSA")
        self._n = int(snapshot["n"])
        self._alpha = float(snapshot["alpha"])
        self._gamma = float(snapshot["gamma"])
        self._epsilon = float(snapshot["epsilon"])
        self._rng = _Lcg(0)
        self._rng.state = int(snapshot.get("rng", 1))
        self._q = {_qkey(k): float(v) for k, v in snapshot.get("q", {}).items()}

    def spec(self) -> dict[str, Any]:
        return {
            "type": "n-step-sarsa",
            "name": self.name,
            "n": self._n,
            "alpha": self._alpha,
            "gamma": self._gamma,
            "epsilon": self._epsilon,
            "seed": self._seed,
        }

    def _greedy(self, state: State) -> Action:
        best_a = self._actions[0]
        best_v = self._q.get((state, best_a), 0.0)
        for a in self._actions[1:]:
            v = self._q.get((state, a), 0.0)
            if v > best_v + 1e-9:
                best_a, best_v = a, v
        return best_a

    def _update(self, flush: bool) -> dict[str, float]:
        """Update Q(s0, a0) with an n-step return and pop the oldest transition.

        Args:
            flush: If True, there is no valid bootstrap (episode ended).

        Returns:
            Diagnostics with ``td_error``.
        """
        if not self._buf:
            return {}
        # First n transitions contribute rewards; transition[n] supplies A_{t+n}.
        window = self._buf[: self._n]
        s0, a0, _, _, _ = window[0]
        g = 0.0
        saw_terminal = False
        for i, (_s, _a, r, _s2, done) in enumerate(window):
            g += (self._gamma**i) * r
            if done:
                saw_terminal = True
                break
        boot = 0.0
        if not flush and not saw_terminal and len(self._buf) > self._n:
            # S_{t+n} is next state of window[-1]; A_{t+n} is action of buf[n].
            nxt = self._buf[self._n]
            s_n, a_n = nxt[0], nxt[1]
            boot = (self._gamma**self._n) * self._q.get((s_n, a_n), 0.0)
        elif not flush and not saw_terminal and len(self._buf) == self._n:
            # No A_{t+n} yet — keep buffering unless caller said flush.
            return {}
        target = g + boot
        q = self._q.get((s0, a0), 0.0)
        td = target - q
        self._q[(s0, a0)] = q + self._alpha * td
        self._buf.pop(0)
        return {"td_error": td, "q": self._q[(s0, a0)]}


def _lit(text: str) -> Any:
    import ast

    return ast.literal_eval(text)


def _qkey(key: str) -> tuple[Any, Any]:
    state_s, action_s = key.split("|", 1)
    return _lit(state_s), _lit(action_s)


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
