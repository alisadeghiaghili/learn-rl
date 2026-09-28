# LearnRL

Interactive reinforcement learning visualizer, sandbox, and tutorial —
inspired by [LearnGitBranching](https://github.com/pcottle/learnGitBranching).

Type commands. Watch the MDP, values, policy, and learning curve update live.
Solve levels. Play command golf.

```
learnrl> level intro.step
learnrl> show
learnrl> step 3
learnrl> train q 50
learnrl> values
learnrl> policy
```

## Why

LearnGitBranching works because git is otherwise invisible: you type commands
and a tree moves. RL is worse — MDPs, value functions, and policies are pure
abstraction. LearnRL gives them a body: a grid, arrows, numbers, a sparkline.

**Try the web sandbox:** https://alisadeghiaghili.github.io/learn-rl/

The product shape mirrors LGB on purpose:

| LGB                         | LearnRL                          |
| --------------------------- | -------------------------------- |
| Commit tree canvas          | Grid map / V / π / Q / curve     |
| `git commit` etc.           | `step`, `train`, `evaluate`, …   |
| `undo` / `reset`            | same                             |
| Levels with goal trees      | Levels with goal metrics         |
| Sequences (intro, remote…)  | Sequences (intro, bandits, td…)  |
| Git golf (command count)    | Command golf (par per level)     |
| `build level` / `import`    | same                             |
| Sandbox mode                | same                             |

## Install

```bash
cd learn-rl
python -m pip install -e ".[dev]"
learnrl
# or
python -m learnrl
```

No runtime dependencies. Python 3.10+.

## Commands

| Command                 | Meaning                                      |
| ----------------------- | -------------------------------------------- |
| `help`                  | Cheat sheet                                  |
| `levels`                | List sequences and challenges                |
| `level <id>`            | Load a level                                 |
| `hint` / `goal`         | Level hint / goal criteria                   |
| `show` (`map`)          | Environment map with the agent               |
| `values` / `policy` / `q` | V(s), greedy π(s), Q(s,a)                  |
| `curve`                 | Training-return sparkline                    |
| `status`                | Env / agent / episode / golf                 |
| `step [n]`              | Take n environment steps                     |
| `reset` / `undo`        | New episode / undo last mutating command     |
| `evaluate [n]`          | Roll out the greedy policy n times           |
| `train [algo] [n]`      | `q`, `sarsa`, `expected`, `vi`, `pi`, `bandit` |
| `set a=0.1 g=0.9 e=0.2` | Hyperparameters (α, γ, ε)                    |
| `golf`                  | Command count vs par                         |
| `build level`           | Export the live world as level JSON          |
| `import level [file]`   | Load custom level JSON                       |

Non-interactive mode (demos, tests, scripting):

```bash
python -m learnrl -c "level td.qlearning; train q 80; evaluate 15; goal"
python -m learnrl -l intro.step -c "step 3; goal"
```

## Curriculum

| Sequence   | What it teaches                                      |
| ---------- | ---------------------------------------------------- |
| `intro`    | States, actions, rewards, episodes, undo/reset       |
| `bandits`  | Explore vs exploit, ε-greedy, sample averages        |
| `mdp`      | Transitions, stochasticity, discounting              |
| `dp`       | Policy evaluation / improvement, value iteration     |
| `td`       | SARSA, Q-learning, Expected SARSA                    |
| `advanced` | Cliff walk, double Q-learning, command golf          |

List them in the app with `levels`. Load with `level intro.step`, etc.

## Architecture

```
learnrl/
  envs/        GridWorld, Bandit, CliffWalk (finite MDPs with transition models)
  agents/      Random, ε-greedy bandit, tabular TD (SARSA/Q/Expected/Double), VI, PI
  engine/      Session (world + episode + golf + undo History)
  commands/    Parser + handlers (the REPL surface)
  viz/         ASCII/ANSI map, values, policy, Q, curve
  levels/      Level schema, goal checks, catalog
```

Design decisions live in [`docs/adr/`](docs/adr/). See also
[`docs/design-notes.md`](docs/design-notes.md).

## Level builder

```
learnrl> level intro.goal
learnrl> train vi 30
learnrl> build level
```

The printed JSON is a complete level (env spec, agent spec, goal criteria, par).
Save it and load later with `import level mylevel.json`, or paste it into a PR.

## Development

```bash
python -m pytest
python -m ruff check learnrl tests
```

## License

MIT
