"""Gridworld environments with optional stochastic slips and pits."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from learnrl.envs.base import FiniteMDP, reward_fn_from_spec
from learnrl.types import Action, Done, Reward, State

# Actions as movement deltas. Order is stable for Q-tables and policies.
UP: Action = "U"
DOWN: Action = "D"
LEFT: Action = "L"
RIGHT: Action = "R"
ACTIONS: tuple[Action, ...] = (UP, DOWN, LEFT, RIGHT)

DELTAS: dict[Action, tuple[int, int]] = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}

ACTION_GLYPHS: dict[Action, str] = {
    UP: "↑",
    DOWN: "↓",
    LEFT: "←",
    RIGHT: "→",
}

DEFAULT_GRID = (
    "S..",
    ".#.",
    "..G",
)


class GridWorld(FiniteMDP):
    """Classic gridworld MDP.

    Map legend:
        ``S`` start, ``G`` goal (terminal), ``P`` pit (terminal), ``#`` wall,
        ``.`` empty. Exactly one ``S`` and at least one goal or pit should
        appear in a well-formed level map.

    Args:
        grid: Row strings describing the map.
        rewards: Named reward overrides (see :func:`reward_fn_from_spec`).
        gamma: Discount factor stored for planners (not applied in raw returns).
        slip_prob: Probability of sliding perpendicular to the intended action.
        seed: Default RNG seed for episode resets.
        name: Optional display name.
    """

    def __init__(
        self,
        grid: Sequence[str] = DEFAULT_GRID,
        *,
        rewards: Mapping[str, float] | None = None,
        gamma: float = 0.9,
        slip_prob: float = 0.0,
        seed: int = 0,
        name: str = "gridworld",
    ) -> None:
        if not grid or not grid[0]:
            raise ValueError("grid must be a non-empty sequence of non-empty rows")
        width = len(grid[0])
        if any(len(row) != width for row in grid):
            raise ValueError("all grid rows must have the same width")

        self._grid: tuple[str, ...] = tuple(grid)
        self._rewards = reward_fn_from_spec(rewards or {})
        self._gamma = float(gamma)
        self._slip_prob = float(slip_prob)
        self._seed = int(seed)
        self._name = name
        self._rng = _Lcg(self._seed)
        self._state: State | None = None

        self._walls: set[tuple[int, int]] = set()
        self._starts: list[tuple[int, int]] = []
        self._goals: set[tuple[int, int]] = set()
        self._pits: set[tuple[int, int]] = set()
        self._free: list[tuple[int, int]] = []

        for r, row in enumerate(self._grid):
            for c, ch in enumerate(row):
                cell = (r, c)
                if ch == "#":
                    self._walls.add(cell)
                else:
                    self._free.append(cell)
                if ch == "S":
                    self._starts.append(cell)
                elif ch == "G":
                    self._goals.add(cell)
                elif ch == "P":
                    self._pits.add(cell)

        if not self._starts:
            self._starts = [self._free[0]] if self._free else [(0, 0)]
        if not self._free:
            raise ValueError("grid has no traversable cells")

    # -- Env interface -------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def states(self) -> Sequence[State]:
        return list(self._free)

    @property
    def actions(self) -> Sequence[Action]:
        return ACTIONS

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def grid(self) -> tuple[str, ...]:
        return self._grid

    def reset(self, *, seed: int | None = None) -> State:
        if seed is not None:
            self._rng = _Lcg(int(seed))
        start = self._starts[self._rng.randrange(len(self._starts))]
        self._state = start
        return start

    def step(self, action: Action) -> Any:
        if action not in DELTAS:
            raise ValueError(f"invalid action: {action!r}")
        if self._state is None:
            raise RuntimeError("call reset() before step()")
        if self.is_terminal(self._state):
            raise RuntimeError("episode already done; call reset()")

        prev = self._state
        if self._slip_prob > 0.0 and self._rng.random() < self._slip_prob:
            perp = [a for a in ACTIONS if a != action and _is_perpendicular(a, action)]
            action = perp[self._rng.randrange(len(perp))] if perp else action

        nxt, hit_wall = self._apply_movement(prev, action)
        self._state = nxt
        reward, done = self._reward_for(nxt, hit_wall)
        return _step(prev, action, reward, nxt, done)

    def is_terminal(self, state: State) -> bool:
        cell = (state[0], state[1])
        return cell in self._goals or cell in self._pits

    def transition(
        self, state: State, action: Action
    ) -> Sequence[tuple[float, Reward, State, Done]]:
        if action not in DELTAS:
            raise ValueError(f"invalid action: {action!r}")
        cell = (state[0], state[1])
        if cell not in self._free:
            raise ValueError(f"unknown state: {state!r}")
        if self.is_terminal(state):
            return [(1.0, 0.0, state, True)]

        outcomes: list[tuple[float, Reward, State, Done]] = []
        if self._slip_prob > 0.0:
            intended_p = 1.0 - self._slip_prob
            perp = [a for a in ACTIONS if a != action and _is_perpendicular(a, action)]
            slip_each = self._slip_prob / max(len(perp), 1)
            candidates: list[tuple[float, Action]] = [(intended_p, action)]
            if perp:
                candidates.extend((slip_each, a) for a in perp)
        else:
            candidates = [(1.0, action)]

        for prob, act in candidates:
            if prob <= 0.0:
                continue
            nxt, hit_wall = self._apply_movement(cell, act)
            reward, done = self._reward_for(nxt, hit_wall)
            outcomes.append((prob, reward, nxt, done))
        return outcomes

    def state_label(self, state: State) -> str:
        return f"({state[0]},{state[1]})"

    def action_label(self, action: Action) -> str:
        return ACTION_GLYPHS.get(action, str(action))

    def cell_kind(self, cell: tuple[int, int]) -> str:
        """Classify a map cell for visualizations.

        Args:
            cell: Row/column coordinate.

        Returns:
            One of ``wall``, ``start``, ``goal``, ``pit``, ``empty``.
        """
        if cell in self._walls:
            return "wall"
        if cell in self._goals:
            return "goal"
        if cell in self._pits:
            return "pit"
        if cell in self._starts:
            return "start"
        return "empty"

    def start_states(self) -> Sequence[State]:
        return list(self._starts)

    def goal_states(self) -> Sequence[State]:
        return sorted(self._goals)

    def pit_states(self) -> Sequence[State]:
        return sorted(self._pits)

    def spec(self) -> dict[str, Any]:
        return {
            "type": "gridworld",
            "name": self._name,
            "grid": list(self._grid),
            "gamma": self._gamma,
            "slip_prob": self._slip_prob,
            "rewards": dict(self._rewards),
            "seed": self._seed,
        }

    # -- internals -----------------------------------------------------

    def _apply_movement(
        self, cell: tuple[int, int], action: Action
    ) -> tuple[tuple[int, int], bool]:
        dr, dc = DELTAS[action]
        nr, nc = cell[0] + dr, cell[1] + dc
        nrows, ncols = len(self._grid), len(self._grid[0])
        if not (0 <= nr < nrows and 0 <= nc < ncols) or (nr, nc) in self._walls:
            return cell, True
        return (nr, nc), False

    def _reward_for(
        self, cell: tuple[int, int], hit_wall: bool
    ) -> tuple[Reward, Done]:
        if hit_wall:
            return self._rewards["wall"], False
        if cell in self._goals:
            return self._rewards["goal"], True
        if cell in self._pits:
            return self._rewards["pit"], True
        return self._rewards["step"], False


def gridworld_from_spec(spec: Mapping[str, Any]) -> GridWorld:
    """Construct a :class:`GridWorld` from a serializable env spec.

    Args:
        spec: Mapping with optional ``grid``, ``rewards``, ``gamma``,
            ``slip_prob``, ``seed``, and ``name``.

    Returns:
        A configured gridworld.

    Examples:
        >>> env = gridworld_from_spec({"grid": ["S.G"]})
        >>> env.gamma
        0.9
    """
    return GridWorld(
        grid=tuple(spec.get("grid", DEFAULT_GRID)),
        rewards=spec.get("rewards"),
        gamma=float(spec.get("gamma", 0.9)),
        slip_prob=float(spec.get("slip_prob", 0.0)),
        seed=int(spec.get("seed", 0)),
        name=str(spec.get("name", "gridworld")),
    )


class _Lcg:
    """Tiny deterministic RNG so environments need no external deps.

    Uses high bits of the LCG state — low bits have short periods.
    """

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


def _is_perpendicular(a: Action, b: Action) -> bool:
    da, db = DELTAS[a], DELTAS[b]
    return da[0] * db[0] + da[1] * db[1] == 0


def _step(prev: State, action: Action, reward: Reward, nxt: State, done: Done) -> Any:
    from learnrl.types import StepResult

    return StepResult(
        state=nxt,
        reward=reward,
        done=done,
        info={},
        prev_state=prev,
        action=action,
    )
