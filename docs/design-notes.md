# Design notes — LearnRL

## Product thesis

LearnGitBranching is not a git manual with pictures. It is a **mechanical
sympathy trainer**: every command moves a structure you can see, so internal
models form faster than documentation allows. RL needs the same treatment.

## Mapping

| Invisible RL object        | Visible stand-in                       |
| -------------------------- | -------------------------------------- |
| State space                | Grid cells / single bandit state       |
| Action                     | Arrow glyph or arm index               |
| Transition dynamics        | Agent position after `step`            |
| Reward                     | Per-step number + episode return       |
| Policy π                   | Arrow field on the map                 |
| Value V / Q                | Numeric grid / per-state action table  |
| Learning progress          | Sparkline of episode returns           |
| Level “goal tree”          | Metric predicates on the session       |

## Interaction loop

```
banner → map
   ↓
command → handler mutates Session → output (+ optional map redraw)
   ↓
goal check → "LEVEL GOAL MET" when predicates pass
```

Undo snapshots the world before each mutating command. Reset restores the
level’s initial env/agent and clears golf (`reset all` in level context).

## What we deliberately did not build (v0.1)

- Browser canvas / React tree (LGB’s signature). Terminal first.
- Function approximation, DQN, policy gradients. Curriculum stops at tabular
  control + DP + Double Q.
- Persistence of solved levels across processes (in-memory marks only).
- Multi-agent / POMDPs.

## Extension seams

- `viz.render_*` can be swapped for Textual widgets without touching `engine`.
- `Level.check_goal` is the only goal language — add keys, not new frameworks.
- `env_from_spec` / `agent_from_spec` registries accept new types in one place.
