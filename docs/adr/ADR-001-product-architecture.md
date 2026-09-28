# ADR-001: Product architecture

## Status

Accepted

## Context

LearnGitBranching teaches git through a live visualization sandbox and a level
curriculum. We want the same product shape for reinforcement learning, in
Python, with zero runtime dependencies so the tool is trivial to install.

Constraints:

- RL concepts (MDP, V, π, Q) are invisible without a visual surface.
- Levels need reproducible environments (tests, goal checks, golf par).
- Command golf and undo are core to the LGB feel.
- The user (Ali Sadeghi Aghili) requires clean architecture, full type hints,
  docstrings, tests alongside code, and ADRs.

## Decision

Build a **local Python package** with a **REPL sandbox** (not a web app):

1. **Session** owns env + agent + episode + golf + undo history.
2. **Commands** are the only mutation surface (parser → handlers → session).
3. **Viz** renders ASCII/ANSI views of the session (map, V, π, Q, curve).
4. **Levels** are declarative data (env/agent/goal/par) checked against session.
5. **stdlib only** at runtime; `pytest`/`ruff` for development.

Web UI is explicitly deferred. The session/viz/command split keeps a future
Textual or browser front-end possible without rewriting the engine.

## Consequences

- Install is `pip install -e .` with no native builds.
- Visualization is terminal-grade, not Canvas-grade like LGB's SVG tree.
- Level sharing is file/gist JSON (same as LGB's gist level builder).
- Undo is snapshot-based (deepcopy of env + agent snapshot dict), sufficient
  for teaching-scale state.
