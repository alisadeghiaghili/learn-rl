"""Command handlers — the sandbox surface of LearnRL."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from learnrl.agents import agent_from_spec
from learnrl.commands.parser import ParsedCommand, first_int
from learnrl.engine.session import Session
from learnrl.envs import env_from_spec
from learnrl.levels import (
    SEQUENCE_INFO,
    all_levels,
    get_level,
    level_from_json,
    levels_in_sequence,
    register_custom,
    sequences,
)
from learnrl.levels.schema import Level
from learnrl.viz import (
    render_curve,
    render_help,
    render_map,
    render_policy,
    render_q,
    render_status,
    render_values,
)
from learnrl.viz.color import BOLD, GREEN, RED, YELLOW, paint


@dataclass
class CommandResult:
    """What the REPL should print / whether to continue.

    Attributes:
        output: Text to print (may be multi-line).
        redraw: Whether the default map view should also be drawn.
        quit: Set true to exit the REPL.
        ok: False marks a user/level failure (still not a crash).
    """

    output: str = ""
    redraw: bool = False
    quit: bool = False
    ok: bool = True


Handler = Callable[[Session, ParsedCommand], CommandResult]

_REGISTRY: dict[str, Handler] = {}


def register(name: str) -> Callable[[Handler], Handler]:
    """Decorator registering a command handler.

    Args:
        name: Canonical command name.

    Returns:
        Decorator that stores the handler and returns it unchanged.
    """

    def deco(fn: Handler) -> Handler:
        _REGISTRY[name] = fn
        return fn

    return deco


def dispatch(session: Session, cmd: ParsedCommand) -> CommandResult:
    """Run one command against the session.

    Args:
        session: Live world.
        cmd: Parsed command.

    Returns:
        A :class:`CommandResult`.

    Raises:
        KeyError: If the command name is unknown (callers should catch).
    """
    handler = _REGISTRY.get(cmd.name)
    if handler is None:
        return CommandResult(
            output=f"unknown command: {cmd.name!r} — type `help`",
            ok=False,
        )
    result = handler(session, cmd)
    # Golf: count mutating commands (LGB counts essentially every command;
    # we count world-mutating ones plus explicit level interactions).
    if cmd.name in _GOLF_COMMANDS:
        session.commands_used += 1
    return result


def known_commands() -> list[str]:
    """Return sorted registered command names.

    Returns:
        Command names for `help` and tests.
    """
    return sorted(_REGISTRY)


_GOLF_COMMANDS = frozenset(
    {
        "step",
        "reset",
        "undo",
        "evaluate",
        "train",
        "set",
    }
)


@register("help")
def _help(session: Session, cmd: ParsedCommand) -> CommandResult:
    return CommandResult(render_help())


@register("show")
def _show(session: Session, cmd: ParsedCommand) -> CommandResult:
    return CommandResult(render_map(session))


@register("values")
def _values(session: Session, cmd: ParsedCommand) -> CommandResult:
    return CommandResult(render_values(session))


@register("policy")
def _policy(session: Session, cmd: ParsedCommand) -> CommandResult:
    return CommandResult(render_policy(session))


@register("q")
def _q(session: Session, cmd: ParsedCommand) -> CommandResult:
    return CommandResult(render_q(session))


@register("curve")
def _curve(session: Session, cmd: ParsedCommand) -> CommandResult:
    return CommandResult(render_curve(session))


@register("status")
def _status(session: Session, cmd: ParsedCommand) -> CommandResult:
    return CommandResult(render_status(session))


@register("step")
def _step(session: Session, cmd: ParsedCommand) -> CommandResult:
    n = first_int(cmd.args, 1)
    if n < 1:
        return CommandResult("step count must be ≥ 1", ok=False)
    try:
        results = session.step_many(n)
    except RuntimeError as exc:
        return CommandResult(str(exc), ok=False)
    if not results:
        return CommandResult("episode already done — `reset` to start again", ok=False)
    last = results[-1]
    lines = [f"step {len(results)}×  action={session.env.action_label(last.action) if last.action is not None else '?'}  r={last.reward:+.2f}  state={session.env.state_label(last.state)}"]
    if last.done:
        verdict = paint("episode done", BOLD + ";" + (GREEN if session.episode.reward > 0 else RED))
        lines.append(f"{verdict}  total return={session.episode.reward:.2f}  steps={session.episode.steps}")
        if session.level is not None:
            goal = session.check_goal()
            lines.extend(goal.details)
            if goal.satisfied:
                lines.append(paint("LEVEL GOAL MET — type `levels` to pick the next one", BOLD + ";" + GREEN))
    return CommandResult("\n".join(lines), redraw=True)


@register("reset")
def _reset(session: Session, cmd: ParsedCommand) -> CommandResult:
    if session.level is not None and cmd.args and cmd.args[0] == "all":
        session.reset_all()
        return CommandResult("Full reset — level restored, golf cleared.", redraw=True)
    state = session.reset_episode()
    return CommandResult(
        f"Episode reset → {session.env.state_label(state)}",
        redraw=True,
    )


@register("undo")
def _undo(session: Session, cmd: ParsedCommand) -> CommandResult:
    # undo itself is not golfed in our counter (already only mutators)
    if session.commands_used > 0:
        session.commands_used -= 0  # keep count; LGB still counts the undone? Keep simple: no-op
    if session.undo():
        return CommandResult("Undid last command.", redraw=True)
    return CommandResult("Nothing to undo.", ok=False)


@register("evaluate")
def _evaluate(session: Session, cmd: ParsedCommand) -> CommandResult:
    n = first_int(cmd.args, 20)
    if n < 1:
        return CommandResult("episode count must be ≥ 1", ok=False)
    ev = session.evaluate(episodes=n)
    text = (
        f"eval · episodes={ev.episodes}  mean={ev.mean_return:.2f}  "
        f"min={ev.min_return:.2f}  max={ev.max_return:.2f}  mean_len={ev.mean_length:.1f}"
    )
    extra = ""
    if session.level is not None:
        goal = session.check_goal()
        extra = "\n" + "\n".join(goal.details)
        if goal.satisfied:
            extra += "\n" + paint("LEVEL GOAL MET", BOLD + ";" + GREEN)
    return CommandResult(text + extra)


@register("train")
def _train(session: Session, cmd: ParsedCommand) -> CommandResult:
    algo = None
    iters = 50
    if cmd.args:
        # train [algo] [n]  OR  train [n]
        if cmd.args[0].isdigit() or _is_number(cmd.args[0]):
            iters = first_int(cmd.args, 50)
        else:
            algo = cmd.args[0]
            if len(cmd.args) > 1:
                iters = first_int(cmd.args[1:], 50)
    if "n" in cmd.options or "iterations" in cmd.options:
        iters = int(float(cmd.options.get("n", cmd.options.get("iterations", iters))))
    try:
        metrics = session.train(algorithm=algo, iterations=iters)
    except (RuntimeError, ValueError) as exc:
        return CommandResult(str(exc), ok=False)
    parts = [f"train · agent={session.agent.name}"]
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.3f}")
        else:
            parts.append(f"{k}={v}")
    body = "  ".join(parts)
    extra = ""
    if session.level is not None:
        goal = session.check_goal()
        extra = "\n" + "\n".join(goal.details)
        if goal.satisfied:
            extra += "\n" + paint("LEVEL GOAL MET", BOLD + ";" + GREEN)
    return CommandResult(body + extra, redraw=True)


@register("set")
def _set(session: Session, cmd: ParsedCommand) -> CommandResult:
    if not cmd.options:
        return CommandResult("usage: set alpha=0.1 gamma=0.9 epsilon=0.2", ok=False)
    kwargs = {}
    for key in ("alpha", "gamma", "epsilon"):
        if key in cmd.options:
            kwargs[key] = float(cmd.options[key])
    try:
        applied = session.set_hyperparams(**kwargs)
    except (RuntimeError, ValueError) as exc:
        return CommandResult(str(exc), ok=False)
    pretty = "  ".join(f"{k}={v}" for k, v in applied.items())
    return CommandResult(f"hyperparams · {pretty}")


@register("levels")
def _levels(session: Session, cmd: ParsedCommand) -> CommandResult:
    lines = [paint("Level sequences", BOLD)]
    for seq in sequences():
        info = SEQUENCE_INFO[seq]
        lines.append("")
        lines.append(paint(f"{seq} — {info['display_name']}", BOLD + ";" + YELLOW))
        lines.append(paint(f"  {info['about']}", "2"))
        for lv in levels_in_sequence(seq):
            mark = "✓" if _is_solved(lv.id) else "·"
            cur = " ← current" if session.level and session.level.id == lv.id else ""
            lines.append(f"  {mark} {lv.id:<22} {lv.name}  (par {lv.par}){cur}")
    lines.append("")
    lines.append(paint("Load with: level <id>", "2"))
    return CommandResult("\n".join(lines))


@register("level")
def _level(session: Session, cmd: ParsedCommand) -> CommandResult:
    if not cmd.args:
        return CommandResult("usage: level <id>", ok=False)
    level_id = cmd.args[0]
    lv = get_level(level_id)
    if lv is None:
        return CommandResult(f"unknown level: {level_id!r} — type `levels`", ok=False)
    return load_level_into_session(session, lv)


@register("hint")
def _hint(session: Session, cmd: ParsedCommand) -> CommandResult:
    if session.level is None:
        return CommandResult("No level loaded. Type `levels`.", ok=False)
    text = session.level.hint or "No hint for this level."
    return CommandResult(paint(text, YELLOW))


@register("goal")
def _goal(session: Session, cmd: ParsedCommand) -> CommandResult:
    if session.level is None:
        return CommandResult("Sandbox mode — no goal. Type `levels` to pick one.", ok=False)
    lv = session.level
    check = lv.check_goal(session)
    header = paint(f"Goal · {lv.id} — {lv.name}", BOLD)
    body = "\n".join(check.details)
    par = f"par {lv.par}  ·  your golf {session.commands_used}"
    return CommandResult(f"{header}\n{body}\n{paint(par, '2')}")


@register("golf")
def _golf(session: Session, cmd: ParsedCommand) -> CommandResult:
    if session.level is None:
        return CommandResult(f"Sandbox golf: {session.commands_used} mutating command(s).")
    lv = session.level
    diff = session.commands_used - lv.par
    tone = GREEN if diff <= 0 else RED
    note = "at or under par" if diff <= 0 else f"{diff} over par"
    return CommandResult(
        f"golf · {session.commands_used} commands  ·  par {lv.par}  ·  {paint(note, tone)}"
    )


@register("build")
def _build(session: Session, cmd: ParsedCommand) -> CommandResult:
    if cmd.args and cmd.args[0] == "level":
        lv = _level_from_session(session)
        register_custom(lv)
        return CommandResult(
            "Exported current world as level JSON (also registered in-memory).\n"
            "Paste this into a file or gist:\n\n" + lv.to_json()
        )
    return CommandResult("usage: build level", ok=False)


@register("export")
def _export(session: Session, cmd: ParsedCommand) -> CommandResult:
    return _build(session, ParsedCommand(name="build", args=("level",), options={}, raw="build level"))


@register("import")
def _import(session: Session, cmd: ParsedCommand) -> CommandResult:
    if cmd.args and cmd.args[0] != "level":
        return CommandResult("usage: import level [file.json]", ok=False)
    if cmd.args and cmd.args[0] == "level" and len(cmd.args) > 1:
        path = Path(cmd.args[1])
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            return CommandResult(f"cannot read {path}: {exc}", ok=False)
    else:
        # Accept JSON from the next program segment via options.payload
        text = cmd.options.get("json", "")
        if not text:
            return CommandResult(
                "Paste level JSON via:  import level  json={...}\n"
                "or:  import level path/to/level.json",
                ok=False,
            )
    try:
        lv = level_from_json(text)
    except ValueError as exc:
        return CommandResult(str(exc), ok=False)
    register_custom(lv)
    return load_level_into_session(session, lv)


@register("clear")
def _clear(session: Session, cmd: ParsedCommand) -> CommandResult:
    return CommandResult("\033[2J\033[H", redraw=True)


@register("quit")
def _quit(session: Session, cmd: ParsedCommand) -> CommandResult:
    return CommandResult("Goodbye.", quit=True)


def load_level_into_session(session: Session, lv: Level) -> CommandResult:
    """Install a level into the session (env, agent, episode, golf).

    Args:
        session: Live world to reconfigure.
        level: Level definition to load.

    Returns:
        Banner result with the level description.
    """
    env = env_from_spec(lv.env_spec)
    agent = agent_from_spec(lv.agent_spec, actions=list(env.actions))
    session.env = env
    session.agent = agent
    session.level = lv
    session.episode = session.episode.__class__()
    session.commands_used = 0
    session.last_eval = None
    session.learning_curve = []
    session.history.clear()
    session._state = env.reset(seed=env.spec().get("seed", 0))  # noqa: SLF001
    session.agent.reset_episode()
    _mark_current(lv.id)
    banner = [
        paint(f"Level {lv.id} — {lv.name}", BOLD + ";" + GREEN),
        "",
        lv.description,
        "",
        paint(f"par {lv.par}  ·  `hint`  ·  `goal`  ·  `help`", "2"),
    ]
    return CommandResult("\n".join(banner), redraw=True)


def _level_from_session(session: Session) -> Level:
    base_id = session.level.id if session.level else "custom.world"
    lv = Level(
        id=f"{base_id}.custom",
        name=f"Custom from {base_id}",
        description="Built with `build level` from the live sandbox world.",
        hint="You wrote this one.",
        env_spec=session.env.spec(),
        agent_spec=session.agent.spec(),
        goal={"episode_done": True} if session.episode.done else {},
        par=max(session.commands_used, 1),
        sequence="custom",
        order=0,
    )
    return lv


# --- progress (solved marks) -------------------------------------------

_SOLVED: set[str] = set()
_CURRENT: str | None = None


def _is_solved(level_id: str) -> bool:
    if level_id in _SOLVED:
        return True
    try:
        from learnrl.progress import load_solved

        return level_id in load_solved()
    except Exception:
        return False


def _mark_current(level_id: str) -> None:
    global _CURRENT
    _CURRENT = level_id


def mark_solved(level_id: str) -> None:
    """Record a level as solved (memory + disk).

    Args:
        level_id: Level identifier.
    """
    _SOLVED.add(level_id)
    try:
        from learnrl.progress import mark_solved as persist

        persist(level_id)
    except Exception:
        pass


def solved_ids() -> frozenset[str]:
    """Return the set of solved level ids in this process.

    Returns:
        Frozen set of ids.
    """
    return frozenset(_SOLVED)


def check_and_record(session: Session) -> bool:
    """Run the goal check and mark the level solved when it passes.

    Args:
        session: Live session with an optional level.

    Returns:
        ``True`` if the goal is satisfied.
    """
    goal = session.check_goal()
    if goal.satisfied and session.level is not None:
        mark_solved(session.level.id)
        # Check after any mutating command too.
    return goal.satisfied


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False
