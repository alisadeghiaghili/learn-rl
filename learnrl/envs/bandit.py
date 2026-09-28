"""Multi-armed bandit environments."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from learnrl.envs.base import Env
from learnrl.types import Action, State, StepResult


class Bandit(Env):
    """Stationary Gaussian multi-armed bandit.

    Each arm has a fixed mean reward. Selecting arm ``k`` yields a reward
    sampled from Normal(mean_k, noise_std). The "state" is a single dummy
    value ``0`` — bandits are stateless one-step problems.

    Args:
        means: True mean reward of each arm.
        noise_std: Standard deviation of the reward noise.
        seed: Default RNG seed.
        name: Display name.
    """

    def __init__(
        self,
        means: Sequence[float] = (0.1, -0.2, 0.4, 0.0, 0.5),
        *,
        noise_std: float = 1.0,
        seed: int = 0,
        name: str = "bandit",
    ) -> None:
        if not means:
            raise ValueError("means must be non-empty")
        self._means = tuple(float(m) for m in means)
        self._noise_std = float(noise_std)
        self._seed = int(seed)
        self._name = name
        self._rng = _Lcg(self._seed)
        self._state: State = 0
        self._done = True  # each pull is a 1-step episode unless stepped as a run

    @property
    def name(self) -> str:
        return self._name

    @property
    def states(self) -> Sequence[State]:
        return (0,)

    @property
    def actions(self) -> Sequence[Action]:
        return tuple(range(len(self._means)))

    @property
    def means(self) -> tuple[float, ...]:
        return self._means

    def reset(self, *, seed: int | None = None) -> State:
        if seed is not None:
            self._rng = _Lcg(int(seed))
        self._state = 0
        self._done = False
        return self._state

    def step(self, action: Action) -> StepResult:
        if not isinstance(action, int) or not (0 <= action < len(self._means)):
            raise ValueError(f"invalid action: {action!r}")
        if self._done:
            raise RuntimeError("call reset() before step()")
        reward = self._means[action] + self._noise_std * self._gauss()
        # Bandit episodes are single-step by convention.
        self._done = True
        return StepResult(
            state=self._state,
            reward=float(reward),
            done=True,
            info={"arm": action, "true_mean": self._means[action]},
            prev_state=self._state,
            action=action,
        )

    def is_terminal(self, state: State) -> bool:
        return self._done and state == self._state

    def optimal_action(self) -> Action:
        """Return the arm with the highest true mean."""
        return max(range(len(self._means)), key=lambda i: self._means[i])

    def action_label(self, action: Action) -> str:
        return f"a{action}"

    def state_label(self, state: State) -> str:
        return "s"

    def spec(self) -> dict[str, Any]:
        return {
            "type": "bandit",
            "name": self._name,
            "means": list(self._means),
            "noise_std": self._noise_std,
            "seed": self._seed,
        }

    def _gauss(self) -> float:
        # Box–Muller with the LCG; good enough for teaching demos.
        u1 = max(self._rng.random(), 1e-12)
        u2 = self._rng.random()
        import math

        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def bandit_from_spec(spec: Mapping[str, Any]) -> Bandit:
    """Build a :class:`Bandit` from a serializable env spec.

    Args:
        spec: Mapping with optional ``means``, ``noise_std``, ``seed``, ``name``.

    Returns:
        A configured bandit environment.

    Examples:
        >>> env = bandit_from_spec({"means": [0.0, 1.0]})
        >>> env.optimal_action()
        1
    """
    return Bandit(
        means=tuple(spec.get("means", (0.1, -0.2, 0.4, 0.0, 0.5))),
        noise_std=float(spec.get("noise_std", 1.0)),
        seed=int(spec.get("seed", 0)),
        name=str(spec.get("name", "bandit")),
    )


class _Lcg:
    """LCG using high bits only — low bits have short periods."""

    __slots__ = ("_s",)

    def __init__(self, seed: int) -> None:
        self._s = int(seed) & 0xFFFFFFFF or 1

    def random(self) -> float:
        return self.randrange(1 << 24) / float(1 << 24)

    def randrange(self, n: int) -> int:
        self._s = (1664525 * self._s + 1013904223) & 0xFFFFFFFF
        if n <= 0:
            return 0
        return (self._s >> 8) % n
