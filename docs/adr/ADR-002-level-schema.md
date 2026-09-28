# ADR-002: Level schema and goal language

## Status

Accepted

## Context

LGB levels compare a live commit tree to a goal tree. RL has no single tree
to match. Goals must express learning outcomes: reach a state, finish an
episode, match a value/policy, or hit a return threshold.

## Decision

A level is JSON with:

- `env_spec` — factory kwargs (`type` in `gridworld` | `cliff` | `bandit`)
- `agent_spec` — factory kwargs (`type` in `random` | `q-learning` | `sarsa` |
  `expected` | `double-q` | `value-iteration` | `policy-iteration` | `bandit-epsilon`)
- `goal` — declarative criteria (AND semantics)
- `par` — command-golf par (informational unless `max_commands` is set)

Goal keys (all optional):

| Key              | Meaning                                      |
| ---------------- | -------------------------------------------- |
| `min/max_commands` | Golf bounds                                |
| `min/max_steps`  | Episode length bounds                        |
| `min/max_return` | Undiscounted return bounds                   |
| `episode_done`   | Require termination                          |
| `visited_states` | States that must appear in the trajectory    |
| `value_close`    | `{"(0,0)": 8.0, "tol": 1.5}` V(s) match      |
| `policy_match`   | `{"(0,0)": "D"}` greedy action match         |
| `min_eval_return`| Mean return from `evaluate`                  |

State keys accept `"(r, c)"`, `[r, c]`, or bare integers (bandit state 0).

## Consequences

- New pedagogical goals are a new key in `Level.check_goal`, not a new engine.
- Levels are shareable as single JSON objects (`build level` / `import level`).
- Par is visible for golf; hard constraints use `max_commands` explicitly so
  learners are not failed for a missing `max_commands` field.
