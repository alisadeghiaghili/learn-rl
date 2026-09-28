# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-01-01

### Added

- Command sandbox REPL with `step`, `train`, `evaluate`, `set`, `undo`, `reset`.
- Live ASCII visualizers: map, V(s), greedy policy, Q(s,a), learning curve.
- Environments: GridWorld (optional slips), Bandit, CliffWalk.
- Agents: random, ε-greedy bandit, SARSA, Q-learning, Expected SARSA,
  Double Q-learning, value iteration, policy iteration.
- Level system with declarative goals and command-golf par.
- Curriculum: `intro`, `bandits`, `mdp`, `dp`, `td`, `advanced` (16 levels).
- Level builder (`build level`) and JSON import.
- Non-interactive `-c/--command` program mode for demos and tests.
