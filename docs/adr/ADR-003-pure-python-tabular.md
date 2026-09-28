# ADR-003: Pure-Python tabular stack

## Status

Accepted

## Context

Teaching RL needs correct algorithms more than scalable ones. NumPy would
speed up DP but adds install friction and obscures the Bellman updates behind
vectorization. The audience is learners reading the code as well as the app.

## Decision

Implement environments, agents, and DP as **explicit Python loops** over finite
state/action spaces:

- `FiniteMDP.transition(s, a) → [(p, r, s', done)]` is the single model API.
- Value iteration and policy iteration call that API directly.
- Tabular TD agents use dict Q-tables keyed by `(state, action)`.
- RNG is a tiny LCG so trajectories are reproducible without `random` seeding
  concerns across library versions.

Double Q-learning keeps a second table but syncs it for teaching clarity
(documented in the class docstring — this is not production Double Q).

## Consequences

- Code reads like the pseudocode in Sutton & Barto.
- No vectorized function approximation yet (DQN is out of scope).
- Performance is fine for teaching grids (≤ a few hundred states).
