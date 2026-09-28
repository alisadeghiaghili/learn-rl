"""Dynamic-programming agents (policy / value iteration)."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from learnrl.agents.base import PlanningAgent
from learnrl.envs.base import FiniteMDP
from learnrl.types import Action, State, Transition


class ValueIteration(PlanningAgent):
    """Synchronous value iteration with an extracted greedy policy.

    Args:
        gamma: Discount factor (overrides env.gamma if provided).
        theta: Convergence threshold on Bellman residual.
        name: Display name.
    """

    def __init__(
        self,
        *,
        gamma: float | None = None,
        theta: float = 1e-4,
        name: str = "value-iteration",
    ) -> None:
        super().__init__(name)
        self._gamma_override = gamma
        self._theta = float(theta)
        self._v: dict[State, float] = {}
        self._q: dict[tuple[State, Action], float] = {}
        self._actions: tuple[Action, ...] = ()
        self._gamma = 0.9

    def plan(self, env: FiniteMDP, *, iterations: int = 1) -> dict[str, float]:
        self._actions = tuple(env.actions)
        self._gamma = float(self._gamma_override if self._gamma_override is not None else getattr(env, "gamma", 0.9))
        if not self._v:
            self._v = {s: 0.0 for s in env.states}
        max_delta = 0.0
        for _ in range(max(1, iterations)):
            max_delta = self._sweep(env)
            if max_delta < self._theta:
                break
        self._refresh_q(env)
        return {"delta": max_delta, "gamma": self._gamma}

    def act(self, state: State, *, explore: bool = True) -> Action:
        if state in self._q or self._q:
            return max(self._actions, key=lambda a: self._q.get((state, a), 0.0))
        return self._actions[0] if self._actions else 0

    def values(self, state: State) -> Mapping[Action, float]:
        return {a: self._q.get((state, a), 0.0) for a in self._actions}

    def state_value(self, state: State) -> float | None:
        return self._v.get(state)

    def v_table(self) -> dict[State, float]:
        return dict(self._v)

    def q_table(self) -> dict[tuple[State, Action], float]:
        return dict(self._q)

    def learn(self, transition: Transition) -> dict[str, float]:
        return {}

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "value-iteration",
            "gamma": self._gamma,
            "theta": self._theta,
            "v": {repr(k): v for k, v in self._v.items()},
            "q": {f"{s!r}|{a!r}": v for (s, a), v in self._q.items()},
            "actions": list(self._actions),
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        if snapshot.get("type") != "value-iteration":
            raise ValueError("snapshot is not ValueIteration")
        self._gamma = float(snapshot["gamma"])
        self._v = {_lit(k): float(v) for k, v in snapshot.get("v", {}).items()}
        self._q = {_qkey(k): float(v) for k, v in snapshot.get("q", {}).items()}
        self._actions = tuple(snapshot.get("actions", ()))

    def spec(self) -> dict[str, Any]:
        return {
            "type": "value-iteration",
            "name": self.name,
            "gamma": self._gamma_override,
            "theta": self._theta,
        }

    def _sweep(self, env: FiniteMDP) -> float:
        max_delta = 0.0
        new_v: dict[State, float] = {}
        for s in env.states:
            if env.is_terminal(s):
                new_v[s] = self._v.get(s, 0.0)
                continue
            best = -math.inf
            for a in env.actions:
                q = 0.0
                for p, r, s2, done in env.transition(s, a):
                    boot = 0.0 if done else self._v.get(s2, 0.0)
                    q += p * (r + self._gamma * boot)
                best = max(best, q)
            new_v[s] = best if best > -math.inf else 0.0
            max_delta = max(max_delta, abs(new_v[s] - self._v.get(s, 0.0)))
        self._v = new_v
        return max_delta

    def _refresh_q(self, env: FiniteMDP) -> None:
        self._q = {}
        for s in env.states:
            for a in env.actions:
                q = 0.0
                for p, r, s2, done in env.transition(s, a):
                    boot = 0.0 if done else self._v.get(s2, 0.0)
                    q += p * (r + self._gamma * boot)
                self._q[(s, a)] = q


class PolicyIteration(PlanningAgent):
    """Policy evaluation + improvement.

    Args:
        gamma: Discount factor override.
        eval_iterations: Exact evaluation sweeps per improvement.
        theta: Evaluation convergence threshold.
        name: Display name.
    """

    def __init__(
        self,
        *,
        gamma: float | None = None,
        eval_iterations: int = 50,
        theta: float = 1e-4,
        name: str = "policy-iteration",
    ) -> None:
        super().__init__(name)
        self._gamma_override = gamma
        self._eval_iterations = int(eval_iterations)
        self._theta = float(theta)
        self._v: dict[State, float] = {}
        self._policy: dict[State, Action] = {}
        self._actions: tuple[Action, ...] = ()
        self._gamma = 0.9

    def plan(self, env: FiniteMDP, *, iterations: int = 1) -> dict[str, float]:
        self._actions = tuple(env.actions)
        self._gamma = float(
            self._gamma_override if self._gamma_override is not None else getattr(env, "gamma", 0.9)
        )
        if not self._policy:
            self._policy = {s: env.actions[0] for s in env.states}
            self._v = {s: 0.0 for s in env.states}
        stable = False
        steps = 0
        for _ in range(max(1, iterations)):
            self._evaluate(env)
            stable = self._improve(env)
            steps += 1
            if stable:
                break
        return {"stable": 1.0 if stable else 0.0, "iterations": float(steps), "gamma": self._gamma}

    def act(self, state: State, *, explore: bool = True) -> Action:
        return self._policy.get(state, self._actions[0] if self._actions else 0)

    def values(self, state: State) -> Mapping[Action, float]:
        # Q implied by greedy policy value; store one-step backup for viz.
        return {self._policy.get(state, self._actions[0]): self._v.get(state, 0.0)}

    def state_value(self, state: State) -> float | None:
        return self._v.get(state)

    def v_table(self) -> dict[State, float]:
        return dict(self._v)

    def policy_table(self) -> dict[State, Action]:
        return dict(self._policy)

    def snapshot(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": "policy-iteration",
            "gamma": self._gamma,
            "v": {repr(k): v for k, v in self._v.items()},
            "policy": {repr(k): v for k, v in self._policy.items()},
            "actions": list(self._actions),
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        if snapshot.get("type") != "policy-iteration":
            raise ValueError("snapshot is not PolicyIteration")
        self._gamma = float(snapshot["gamma"])
        self._v = {_lit(k): float(v) for k, v in snapshot.get("v", {}).items()}
        self._policy = {_lit(k): v for k, v in snapshot.get("policy", {}).items()}
        self._actions = tuple(snapshot.get("actions", ()))

    def spec(self) -> dict[str, Any]:
        return {
            "type": "policy-iteration",
            "name": self.name,
            "gamma": self._gamma_override,
            "eval_iterations": self._eval_iterations,
            "theta": self._theta,
        }

    def _evaluate(self, env: FiniteMDP) -> None:
        for _ in range(self._eval_iterations):
            max_delta = 0.0
            new_v: dict[State, float] = {}
            for s in env.states:
                if env.is_terminal(s):
                    new_v[s] = self._v.get(s, 0.0)
                    continue
                a = self._policy.get(s, self._actions[0])
                val = 0.0
                for p, r, s2, done in env.transition(s, a):
                    boot = 0.0 if done else self._v.get(s2, 0.0)
                    val += p * (r + self._gamma * boot)
                new_v[s] = val
                max_delta = max(max_delta, abs(val - self._v.get(s, 0.0)))
            self._v = new_v
            if max_delta < self._theta:
                break

    def _improve(self, env: FiniteMDP) -> bool:
        stable = True
        for s in env.states:
            if env.is_terminal(s):
                continue
            old = self._policy.get(s)
            best_a = self._actions[0]
            best_q = -math.inf
            for a in self._actions:
                q = 0.0
                for p, r, s2, done in env.transition(s, a):
                    boot = 0.0 if done else self._v.get(s2, 0.0)
                    q += p * (r + self._gamma * boot)
                if q > best_q + 1e-12:
                    best_q, best_a = q, a
            self._policy[s] = best_a
            if old != best_a:
                stable = False
        return stable


def _lit(text: str) -> Any:
    import ast

    return ast.literal_eval(text)


def _qkey(key: str) -> tuple[Any, Any]:
    state_s, action_s = key.split("|", 1)
    return _lit(state_s), _lit(action_s)
