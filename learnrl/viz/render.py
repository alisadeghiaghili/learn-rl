"""Terminal renderers for maps, values, policies, Q-tables, and curves."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from learnrl.agents.base import Agent
from learnrl.agents.tabular import BanditAgent, QAgent
from learnrl.envs.bandit import Bandit
from learnrl.envs.gridworld import ACTION_GLYPHS, GridWorld
from learnrl.viz.color import BOLD, CYAN, DIM, GRAY, GREEN, MAGENTA, RED, YELLOW, paint


def render_map(session: Any) -> str:
    """Render the environment map with the agent and legend.

    Args:
        session: A :class:`~learnrl.engine.session.Session`.

    Returns:
        Multi-line ASCII map.
    """
    env = session.env
    if isinstance(env, Bandit):
        return render_bandit(session)
    if not isinstance(env, GridWorld):
        return f"Environment `{env.name}` has no grid map. Try `values` or `policy`."

    grid = env.grid
    state = session.state
    rows: list[str] = []
    for r, row in enumerate(grid):
        cells: list[str] = []
        for c, ch in enumerate(row):
            cell = (r, c)
            kind = env.cell_kind(cell)
            if state is not None and (state[0], state[1]) == cell and kind not in ("wall",):
                glyph = "A"
                cells.append(paint(glyph, BOLD + ";" + YELLOW))
            elif kind == "wall":
                cells.append(paint("█", GRAY))
            elif kind == "goal":
                cells.append(paint("G", GREEN))
            elif kind == "pit":
                cells.append(paint("P", RED))
            elif kind == "start":
                cells.append(paint("S", CYAN))
            else:
                cells.append("·")
        rows.append(" ".join(cells))

    legend = f"{paint('A', BOLD + ';' + YELLOW)} agent  {paint('S', CYAN)} start  {paint('G', GREEN)} goal  {paint('P', RED)} pit  {paint('█', GRAY)} wall"
    header = f"map · {env.name} · episode step {session.episode.steps}  return {session.episode.reward:.2f}"
    if session.episode.done:
        header += paint("  [done]", BOLD + ";" + GREEN if session.episode.reward > 0 else BOLD + ";" + RED)
    return "\n".join([paint(header, DIM), *rows, paint(legend, GRAY)])


def render_values(session: Any) -> str:
    """Render state values V(s) as a numeric grid.

    Args:
        session: Active session.

    Returns:
        Multi-line value heatmap (plain numbers) or Q fallback.
    """
    env = session.env
    agent = session.agent
    if isinstance(env, Bandit):
        return render_q(session)
    if not isinstance(env, GridWorld):
        return "No tabular values for this environment. Try `q`."

    vmap: dict[tuple[int, int], float] = {}
    for s in env.states:
        cell = (s[0], s[1])
        v = agent.state_value(s)
        vmap[cell] = float(v) if v is not None else 0.0

    if isinstance(agent, QAgent) and agent.q_table():
        qmap = agent.q_table()
        for (s, a), q in qmap.items():
            cell = (s[0], s[1])
            # show max_a Q as V when Q-table exists
            cur = vmap.get(cell, float("-inf"))
            vmap[cell] = max(cur, q)

    rows = _numeric_grid(env, vmap, fmt="{:+.2f}")
    header = f"V(s) · agent={agent.name}"
    return "\n".join([paint(header, DIM), *rows])


def render_policy(session: Any) -> str:
    """Render the greedy policy as arrows on the map.

    Args:
        session: Active session.

    Returns:
        Multi-line arrow map.
    """
    env = session.env
    agent = session.agent
    if isinstance(env, Bandit):
        best = None
        try:
            best = agent.policy(0)
        except Exception:
            best = None
        return f"bandit policy: pick arm {best}" if best is not None else "bandit policy: (no Q yet)"

    if not isinstance(env, GridWorld):
        return "No spatial policy for this environment. Try `q`."

    def cell_glyph(r: int, c: int) -> str:
        cell = (r, c)
        kind = env.cell_kind(cell)
        if kind == "wall":
            return paint("█", GRAY)
        if kind == "goal":
            return paint("G", GREEN)
        if kind == "pit":
            return paint("P", RED)
        action = agent.policy(cell)
        if action is None:
            return paint("·", DIM)
        glyph = ACTION_GLYPHS.get(action, str(action))
        return paint(glyph, CYAN)

    rows: list[str] = []
    for r, row in enumerate(env.grid):
        rows.append(" ".join(cell_glyph(r, c) for c in range(len(row))))

    header = f"π(s) greedy · agent={agent.name}"
    legend = paint("arrows = greedy action  ·  G goal  ·  P pit  ·  █ wall", GRAY)
    return "\n".join([paint(header, DIM), *rows, legend])


def render_q(session: Any) -> str:
    """Render action values.

    For bandits: one bar/value per arm. For gridworlds: per-state best-action
    table with all four Q values compactly.

    Args:
        session: Active session.

    Returns:
        Text block of Q values.
    """
    env = session.env
    agent = session.agent
    if isinstance(env, Bandit) and isinstance(agent, BanditAgent):
        q = agent.values(0)
        counts = agent.counts()
        lines = [paint(f"Q(a) estimates · agent={agent.name}", DIM)]
        best = max(q.values()) if q else 0.0
        for a in env.actions:
            val = q.get(a, 0.0)
            bar = _bar(val, best)
            star = " ← optimal" if a == env.optimal_action() else ""
            lines.append(f"  a{a}  {val:+7.3f}  {bar}  n={counts[int(a)]}{star}")
        return "\n".join(lines)

    qmap: Mapping[Any, float] = {}
    if isinstance(agent, QAgent):
        qmap = agent.q_table()
    if not qmap:
        return "No Q-table yet. Try `train q 50` or `train sarsa 50`."

    lines = [paint(f"Q(s,a) · agent={agent.name}", DIM)]
    # Group by state
    by_state: dict[Any, dict[Any, float]] = {}
    for (s, a), v in qmap.items():
        by_state.setdefault(s, {})[a] = float(v)

    for s in sorted(by_state, key=lambda x: (x[0], x[1]) if isinstance(x, tuple) else (str(x), 0)):
        acts = by_state[s]
        best_a = max(acts, key=lambda a: acts[a])
        parts = []
        for a in sorted(acts, key=str):
            mark = "*" if a == best_a else " "
            parts.append(f"{env.action_label(a)}{mark}{acts[a]:+.2f}")
        state_l = env.state_label(s)
        agent_mark = "A" if session.state == s else " "
        lines.append(f"  {agent_mark}{state_l:>8}  " + "  ".join(parts))
    lines.append(paint("  format: <action><*best><value>", GRAY))
    return "\n".join(lines)


def render_curve(session: Any, *, width: int = 40) -> str:
    """Render a sparkline of training returns.

    Args:
        session: Active session.
        width: Maximum sparkline width in characters.

    Returns:
        One-line sparkline plus min/max/mean summary.
    """
    data = session.learning_curve
    if not data:
        return "learning curve: empty — run `train q 20` first."
    pts = data[-width:]
    lo, hi = min(pts), max(pts)
    span = (hi - lo) or 1.0
    blocks = "▁▂▃▄▅▆▇█"
    chars = []
    for v in pts:
        idx = int((v - lo) / span * (len(blocks) - 1))
        chars.append(blocks[idx])
    mean = sum(pts) / len(pts)
    summary = f"n={len(data)}  mean={mean:.2f}  min={lo:.2f}  max={hi:.2f}"
    return "\n".join([paint("learning curve (episode return)", DIM), "  " + "".join(chars), paint("  " + summary, GRAY)])


def render_status(session: Any) -> str:
    """Compact status line used after many commands.

    Args:
        session: Active session.

    Returns:
        Multi-line status block.
    """
    env = session.env
    agent = session.agent
    lines = [
        paint("status", BOLD),
        f"  env     : {env.name}  ({type(env).__name__})",
        f"  agent   : {agent.name}",
        f"  state   : {env.state_label(session.state) if session.state is not None else '-'}",
        f"  episode : steps={session.episode.steps}  return={session.episode.reward:.2f}  done={session.episode.done}",
        f"  golf    : {session.commands_used} command(s)",
    ]
    if session.level is not None:
        lines.append(f"  level   : {session.level.id} — {session.level.name}")
        goal = session.check_goal()
        mark = paint("OK", GREEN) if goal.satisfied else paint("…", YELLOW)
        lines.append(f"  goal    : {mark}")
    if session.last_eval is not None:
        ev = session.last_eval
        lines.append(f"  last eval: mean_return={ev.mean_return:.2f} over {ev.episodes} eps")
    return "\n".join(lines)


def render_bandit(session: Any) -> str:
    """Map view specialized for bandits (true means hidden).

    Args:
        session: Active session.

    Returns:
        Text block listing estimated values.
    """
    return render_q(session)


def render_help() -> str:
    """Return the command cheat sheet.

    Returns:
        Formatted help text.
    """
    rows = [
        ("help", "show this help"),
        ("levels", "list level sequences"),
        ("level <id>", "load a level (e.g. level intro.step)"),
        ("hint", "hint for the current level"),
        ("goal", "show goal criteria"),
        ("show | map", "visualize the environment"),
        ("values", "show V(s)"),
        ("policy", "show greedy policy π(s)"),
        ("q", "show action values Q(s,a)"),
        ("curve", "show learning curve sparkline"),
        ("status", "compact world status"),
        ("step [n]", "take n environment steps (default 1)"),
        ("reset", "reset episode (same level)"),
        ("undo", "undo last command"),
        ("evaluate [n]", "evaluate policy over n episodes (default 20)"),
        ("train [algo] [n]", "train (algo: q, sarsa, expected, vi, pi, mc, td, n-step-sarsa, bandit)"),
        ("set a=0.1 g=0.9 e=0.2", "set alpha/gamma/epsilon"),
        ("golf", "command-golf stats for the level"),
        ("build level", "export current world as a level JSON"),
        ("export level", "alias of build level"),
        ("import level", "load a level from JSON (stdin or file)"),
        ("clear", "clear the screen"),
        ("quit | exit", "leave LearnRL"),
    ]
    width = max(len(c) for c, _ in rows)
    lines = [paint("LearnRL commands", BOLD)]
    for cmd, desc in rows:
        lines.append(f"  {cmd:<{width}}  {desc}")
    lines.append("")
    lines.append(paint("Tip: levels are solved by meeting the goal — type `goal` after loading one.", DIM))
    return "\n".join(lines)


def _numeric_grid(
    env: GridWorld,
    values: Mapping[tuple[int, int], float],
    *,
    fmt: str = "{:+.2f}",
) -> list[str]:
    rows: list[str] = []
    width = max((len(fmt.format(v)) for v in values.values()), default=6)
    for r, row in enumerate(env.grid):
        cells: list[str] = []
        for c, _ch in enumerate(row):
            cell = (r, c)
            kind = env.cell_kind(cell)
            if kind == "wall":
                cells.append(" " * width)
            elif kind == "goal":
                cells.append(paint(fmt.format(values.get(cell, 0.0)).center(width), GREEN))
            elif kind == "pit":
                cells.append(paint(fmt.format(values.get(cell, 0.0)).center(width), RED))
            else:
                val = values.get(cell, 0.0)
                text = fmt.format(val).center(width)
                cells.append(text)
        rows.append(" ".join(cells))
    return rows


def _bar(value: float, best: float) -> str:
    if best <= 0:
        return ""
    n = max(0, int(round(value / best * 10))) if best else 0
    n = min(n, 10)
    return "█" * n
