"""Level schema: goals, specs, and checking against a session."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from learnrl.types import GoalCheck


@dataclass(frozen=True, slots=True)
class Level:
    """A tutorial/challenge definition.

    Attributes:
        id: Stable identifier such as ``intro.step``.
        name: Short display title.
        description: Multi-line lesson text shown on load.
        hint: Optional progressive hint.
        env_spec: Environment factory kwargs (``type`` required).
        agent_spec: Starting agent factory kwargs.
        goal: Goal criteria mapping (see :func:`check_goal`).
        par: Command-golf par count.
        sequence: Owning sequence id (``intro``, ``bandits``, ...).
        order: Display order within the sequence.
    """

    id: str
    name: str
    description: str
    hint: str = ""
    env_spec: Mapping[str, Any] = field(default_factory=dict)
    agent_spec: Mapping[str, Any] = field(default_factory=lambda: {"type": "random"})
    goal: Mapping[str, Any] = field(default_factory=dict)
    par: int = 5
    sequence: str = "intro"
    order: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize this level to a JSON-friendly dict.

        Returns:
            A dict suitable for ``json.dumps`` and ``import level``.
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "hint": self.hint,
            "env_spec": dict(self.env_spec),
            "agent_spec": dict(self.agent_spec),
            "goal": dict(self.goal),
            "par": self.par,
            "sequence": self.sequence,
            "order": self.order,
        }

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize to a JSON string.

        Args:
            indent: ``json.dumps`` indent.

        Returns:
            Pretty-printed JSON.
        """
        return json.dumps(self.to_dict(), indent=indent)

    @staticmethod
    def from_dict(data: Mapping[str, Any]) -> Level:
        """Build a level from a mapping.

        Args:
            data: Fields matching :meth:`to_dict`.

        Returns:
            A :class:`Level`.

        Raises:
            ValueError: If required fields are missing.
        """
        if "id" not in data or "name" not in data:
            raise ValueError("level requires 'id' and 'name'")
        return Level(
            id=str(data["id"]),
            name=str(data["name"]),
            description=str(data.get("description", "")),
            hint=str(data.get("hint", "")),
            env_spec=dict(data.get("env_spec", {"type": "gridworld"})),
            agent_spec=dict(data.get("agent_spec", {"type": "random"})),
            goal=dict(data.get("goal", {})),
            par=int(data.get("par", 5)),
            sequence=str(data.get("sequence", "intro")),
            order=int(data.get("order", 0)),
        )

    def check_goal(self, session: Any) -> GoalCheck:
        """Compare the live session against this level's goal criteria.

        Supported goal keys (all optional; every provided key must pass):

        * ``min_commands`` / ``max_commands`` — bound on golf count
        * ``max_commands`` is also implied by ``par`` when ``use_par`` is true
        * ``min_steps`` / ``max_steps`` — episode step bounds after last action
        * ``min_return`` / ``max_return`` — current episode return bounds
        * ``episode_done`` — require the episode to be finished
        * ``visited_states`` — states that must appear in the episode history
        * ``value_close`` — ``{state: target, tol}`` V(s) match
        * ``policy_match`` — ``{state: action}`` greedy policy match
        * ``min_eval_return`` — mean return from `evaluate` (uses last_eval or runs one)

        Args:
            session: Live :class:`~learnrl.engine.session.Session`.

        Returns:
            A :class:`~learnrl.types.GoalCheck` with per-criterion notes.
        """
        goal = dict(self.goal or {})
        details: list[str] = []
        ok = True

        def check(cond: bool, msg: str) -> None:
            nonlocal ok
            details.append(("✓ " if cond else "✗ ") + msg)
            if not cond:
                ok = False

        max_cmds = goal.get("max_commands")
        if max_cmds is None and goal.get("use_par", True) and "max_commands" not in goal:
            # Par is informational golf; only enforce when explicitly requested.
            max_cmds = None
        if max_cmds is not None:
            check(session.commands_used <= int(max_cmds), f"commands {session.commands_used} ≤ {max_cmds}")
        min_cmds = goal.get("min_commands")
        if min_cmds is not None:
            check(session.commands_used >= int(min_cmds), f"commands {session.commands_used} ≥ {min_cmds}")

        if "episode_done" in goal:
            want = bool(goal["episode_done"])
            check(session.episode.done is want, f"episode_done == {want}")

        if "min_steps" in goal:
            check(session.episode.steps >= int(goal["min_steps"]), f"steps ≥ {goal['min_steps']}")
        if "max_steps" in goal:
            check(session.episode.steps <= int(goal["max_steps"]), f"steps ≤ {goal['max_steps']}")

        if "min_return" in goal:
            check(session.episode.reward >= float(goal["min_return"]), f"return ≥ {goal['min_return']}")
        if "max_return" in goal:
            check(session.episode.reward <= float(goal["max_return"]), f"return ≤ {goal['max_return']}")

        visited = goal.get("visited_states") or []
        if visited:
            seen: list[Any] = []
            for t in session.episode.history:
                if t[0] is not None:
                    seen.append(t[0])
                seen.append(t[3])
            # Also include current state
            if session.state is not None:
                seen.append(session.state)
            for need in visited:
                check(_state_in(need, seen), f"visited {need}")

        value_close = goal.get("value_close") or {}
        if value_close:
            tol = float(value_close.get("tol", 0.15))
            targets = {k: v for k, v in value_close.items() if k != "tol"}
            for state_key, target in targets.items():
                state = _coerce_state(state_key)
                v = session.agent.state_value(state)
                if v is None:
                    check(False, f"V({state_key}) available")
                else:
                    check(abs(v - float(target)) <= tol, f"V({state_key}) ≈ {target} (got {v:.3f}, tol {tol})")

        policy_match = goal.get("policy_match") or {}
        for state_key, action in policy_match.items():
            state = _coerce_state(state_key)
            got = session.agent.policy(state)
            check(got == action or str(got) == str(action), f"π({state_key}) == {action} (got {got})")

        if "min_eval_return" in goal:
            ev = session.last_eval
            stale = getattr(session, "_eval_gen", None) != getattr(session, "_world_gen", 0)
            if ev is None or stale:
                ev = session.evaluate(episodes=int(goal.get("eval_episodes", 20)))
            check(
                ev.mean_return >= float(goal["min_eval_return"]),
                f"mean eval return ≥ {goal['min_eval_return']} (got {ev.mean_return:.2f})",
            )

        if not details:
            details.append("✓ sandbox mode (no criteria)")
            ok = True

        return GoalCheck(ok, tuple(details))


def _coerce_state(key: Any) -> Any:
    """Coerce a goal-map key into a state object.

    Args:
        key: ``"(r, c)"``, ``[r, c]``, ``r`` (bandit arm state), or raw state.

    Returns:
        A hashable state.
    """
    if isinstance(key, (list, tuple)) and len(key) == 2 and all(isinstance(x, int) for x in key):
        return (key[0], key[1])
    if isinstance(key, str):
        text = key.strip()
        if text.startswith("(") and text.endswith(")"):
            import ast

            return ast.literal_eval(text)
        if text.startswith("[") and text.endswith("]"):
            import ast

            val = ast.literal_eval(text)
            return tuple(val) if isinstance(val, list) else val
        try:
            return int(text)
        except ValueError:
            return text
    return key


def _state_in(need: Any, seen: Sequence[Any]) -> bool:
    target = _coerce_state(need)
    for s in seen:
        if s == target or str(s) == str(target):
            return True
        if isinstance(s, tuple) and isinstance(target, tuple) and s == target:
            return True
    return False
