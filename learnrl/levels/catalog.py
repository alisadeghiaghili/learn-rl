"""Level catalog: sequences and all built-in challenges."""

from __future__ import annotations

from typing import Iterable, Sequence

from learnrl.levels.schema import Level

SEQUENCE_INFO: dict[str, dict[str, str]] = {
    "intro": {
        "display_name": "Introduction Sequence",
        "about": "States, actions, rewards, and episodes — the RL vocabulary.",
    },
    "bandits": {
        "display_name": "Multi-Armed Bandits",
        "about": "Explore vs exploit without state. Sample averages and ε-greedy.",
    },
    "mdp": {
        "display_name": "MDPs & Returns",
        "about": "Transitions, discounting, and why γ matters.",
    },
    "dp": {
        "display_name": "Dynamic Programming",
        "about": "Policy evaluation, improvement, and value iteration on known models.",
    },
    "prediction": {
        "display_name": "Prediction (MC & TD)",
        "about": "Estimate V^π before you control it. Monte Carlo vs TD(0) vs n-step.",
    },
    "td": {
        "display_name": "Temporal Difference",
        "about": "Learn from interaction: SARSA, Q-learning, Expected SARSA.",
    },
    "advanced": {
        "display_name": "Advanced Topics",
        "about": "Exploration pitfalls, double Q-learning, and cliff-walk trade-offs.",
    },
}

_LEVELS: list[Level] = []


def _register(level: Level) -> Level:
    _LEVELS.append(level)
    return level


def all_levels() -> Sequence[Level]:
    """Return every built-in level in catalog order.

    Returns:
        A sequence of :class:`~learnrl.levels.schema.Level`.
    """
    return tuple(_LEVELS)


def get_level(level_id: str) -> Level | None:
    """Look up a level by id.

    Args:
        level_id: Exact id such as ``intro.step``.

    Returns:
        The level or ``None``.
    """
    for lv in _LEVELS:
        if lv.id == level_id:
            return lv
    return None


def levels_in_sequence(sequence: str) -> Sequence[Level]:
    """Return levels belonging to one sequence, sorted by order.

    Args:
        sequence: Sequence key from :data:`SEQUENCE_INFO`.

    Returns:
        Matching levels.
    """
    return tuple(sorted((lv for lv in _LEVELS if lv.sequence == sequence), key=lambda lv: lv.order))


def sequences() -> Sequence[str]:
    """Return sequence keys in display order.

    Returns:
        Sequence identifiers.
    """
    return tuple(SEQUENCE_INFO.keys())


def register_custom(level: Level) -> Level:
    """Add a user-imported level to the in-memory catalog.

    Args:
        level: Level to register. Existing ids are replaced.

    Returns:
        The registered level.
    """
    global _LEVELS
    _LEVELS = [lv for lv in _LEVELS if lv.id != level.id]
    _LEVELS.append(level)
    return level


def iter_with_defaults(levels: Iterable[Level]) -> list[Level]:
    """Sort helper: sequence order then level order then id.

    Args:
        levels: Levels to sort.

    Returns:
        A new sorted list.
    """
    seq_index = {k: i for i, k in enumerate(SEQUENCE_INFO.keys())}
    return sorted(levels, key=lambda lv: (seq_index.get(lv.sequence, 99), lv.order, lv.id))


# ---------------------------------------------------------------------------
# Catalog content
# ---------------------------------------------------------------------------

# --- intro ---------------------------------------------------------------

_register(
    Level(
        id="intro.step",
        name="Take a Step",
        description=(
            "Welcome to LearnRL — the LearnGitBranching of reinforcement learning.\n"
            "\n"
            "You are the agent `A` on a tiny gridworld. The environment has states\n"
            "(cells), actions (U/D/L/R), and rewards. Your job right now is just to\n"
            "feel an episode: run `show`, then `step` three times.\n"
            "\n"
            "Commands cheat sheet: `help` · `show` · `step` · `status` · `goal`."
        ),
        hint="Type `step` three times. Watch the map and the episode return in `status`.",
        env_spec={"type": "gridworld", "grid": ["S..", "...", "..G"], "gamma": 0.9, "seed": 1},
        agent_spec={"type": "random", "seed": 1},
        goal={"min_steps": 3, "max_commands": 10},
        par=3,
        sequence="intro",
        order=1,
    )
)

_register(
    Level(
        id="intro.goal",
        name="Reach the Goal",
        description=(
            "Now do something useful. Navigate to the green `G`.\n"
            "\n"
            "`step` lets the agent choose (it starts random). You can also step\n"
            "manually once you know the geometry: `step` is enough for this level —\n"
            "keep stepping until the episode is done.\n"
            "\n"
            "The map: `S` start, `G` goal (terminal, positive reward), `·` empty."
        ),
        hint="Keep calling `step` until you land on G. Use `reset` + `undo` if stuck exploring poorly.",
        env_spec={"type": "gridworld", "grid": ["S.", ".G"], "gamma": 0.9, "seed": 2, "rewards": {"step": -1, "goal": 10}},
        agent_spec={"type": "random", "seed": 2},
        goal={"episode_done": True, "min_return": 0.0},
        par=6,
        sequence="intro",
        order=2,
    )
)

_register(
    Level(
        id="intro.rewards",
        name="Mind the Pit",
        description=(
            "Rewards are how we grade behavior.\n"
            "\n"
            "Here `P` is a pit: stepping in ends the episode with a large penalty.\n"
            "Reach `G` without falling in. Inspect the map with `show` and the\n"
            "episode return with `status` after each step."
        ),
        hint="Avoid the P cells. The safe route stays on the top/right side.",
        env_spec={
            "type": "gridworld",
            "grid": ["S.P", "..P", "..G"],
            "gamma": 0.9,
            "seed": 3,
            "rewards": {"step": -1, "goal": 5, "pit": -50},
        },
        agent_spec={"type": "random", "seed": 3},
        goal={"episode_done": True, "min_return": -5.0},
        par=8,
        sequence="intro",
        order=3,
    )
)

_register(
    Level(
        id="intro.sandbox",
        name="Sandbox Reset & Undo",
        description=(
            "LearnGitBranching has `undo` and `reset`. So do we.\n"
            "\n"
            "Prove you can rewind: take a step, then `undo`, then `reset`.\n"
            "Golf counts every command that mutates the world (step/train/…).\n"
            "Stay under par."
        ),
        hint="`step` then `undo` then `reset`. That's three commands.",
        env_spec={"type": "gridworld", "grid": ["S.G"], "gamma": 0.9, "seed": 4},
        agent_spec={"type": "random", "seed": 4},
        goal={"max_commands": 5},
        par=3,
        sequence="intro",
        order=4,
    )
)

# --- bandits -------------------------------------------------------------

_register(
    Level(
        id="bandits.pull",
        name="Pull an Arm",
        description=(
            "A multi-armed bandit has actions (arms) but no state.\n"
            "Each pull returns a noisy reward. Your estimate Q(a) starts at 0.\n"
            "\n"
            "Load this level, look at `q`, then `step` a few times to update Q.\n"
            "Goal: take at least 5 pulls so the estimates move."
        ),
        hint="`step 10` runs several pulls in one golf command… if the parser accepts a count.",
        env_spec={"type": "bandit", "means": [0.0, 0.5, -0.3], "noise_std": 0.5, "seed": 5},
        agent_spec={"type": "bandit-epsilon", "n_arms": 3, "epsilon": 0.2, "seed": 5},
        goal={"min_steps": 5},
        par=3,
        sequence="bandits",
        order=1,
    )
)

_register(
    Level(
        id="bandits.explore",
        name="Explore, Then Exploit",
        description=(
            "Pure greedy freezes on the first lucky arm. ε-greedy keeps sampling.\n"
            "\n"
            "Train the bandit agent for many pulls, then evaluate. The best arm\n"
            "has mean 0.5 — your eval return should approach it (noise still moves\n"
            "each pull).\n"
            "\n"
            "Try: `set e=0.2` · `train bandit 50` · `evaluate 30`."
        ),
        hint="Use `train bandit 50` then `evaluate 30`. Target mean return is about 0.5.",
        env_spec={"type": "bandit", "means": [0.0, 0.5, -0.3], "noise_std": 0.5, "seed": 6},
        agent_spec={"type": "bandit-epsilon", "n_arms": 3, "epsilon": 0.2, "seed": 6},
        goal={"min_eval_return": 0.2, "eval_episodes": 30},
        par=4,
        sequence="bandits",
        order=2,
    )
)

# --- prediction (MC & TD) -----------------------------------------------

_register(
    Level(
        id="prediction.mc",
        name="Monte Carlo Prediction",
        description=(
            "Before controlling a policy, estimate the value of one.\n"
            "\n"
            "Monte Carlo waits for a complete episode, computes return-to-go G_t,\n"
            "and averages: V(s) ← V(s) + (1/N)[G − V(s)].\n"
            "\n"
            "This tiny map has a known optimal V(0,0) ≈ 8 under greedy walks to G.\n"
            "Your job is not to control — it is to *estimate* V by rolling episodes.\n"
            "\n"
            "  train mc 100\n"
            "  values\n"
            "\n"
            "Notice how V is noisier than value-iteration's exact answer."
        ),
        hint="`train mc 150` then `values`. V(0,0) should land near 8 but will jitter.",
        env_spec={
            "type": "gridworld",
            "grid": ["S.G"],
            "gamma": 1.0,
            "seed": 20,
            "rewards": {"step": -1, "goal": 10},
        },
        agent_spec={"type": "mc-prediction", "seed": 20},
        goal={"value_close": {"(0, 0)": 8.0, "tol": 2.5}},
        par=3,
        sequence="prediction",
        order=1,
    )
)

_register(
    Level(
        id="prediction.td",
        name="TD(0) Prediction",
        description=(
            "TD bootstraps every step — no waiting for the episode to end:\n"
            "\n"
            "  V(s) ← V(s) + α [r + γ V(s') − V(s)]\n"
            "\n"
            "Lower variance than MC, higher bias early on. Run TD on the same map\n"
            "and compare the smoothness of `values` vs the MC level.\n"
            "\n"
            "  train td 100\n"
            "  values"
        ),
        hint="`set a=0.2` is not needed — alpha is in the agent. Just `train td 100` · `values`.",
        env_spec={
            "type": "gridworld",
            "grid": ["S.G"],
            "gamma": 1.0,
            "seed": 21,
            "rewards": {"step": -1, "goal": 10},
        },
        agent_spec={"type": "td-prediction", "alpha": 0.2, "gamma": 1.0, "seed": 21},
        goal={"value_close": {"(0, 0)": 8.0, "tol": 1.2}},
        par=3,
        sequence="prediction",
        order=2,
    )
)

_register(
    Level(
        id="prediction.nstep",
        name="n-Step Returns",
        description=(
            "n-step SARSA sits between TD(0) and MC:\n"
            "\n"
            "  G_{t:t+n} = r + γ r' + ... + γ^{n-1} r^{(n)} + γ^n Q(s^{(n)}, a^{(n)})\n"
            "\n"
            "n=1 is ordinary SARSA. Large n → more MC-like (less bias, more variance).\n"
            "\n"
            "Train 3-step SARSA and reach the goal with a solid eval return.\n"
            "Then try `train n-step-sarsa 80` after `set` — or reload with different n\n"
            "via `level prediction.nstep` and compare `curve`."
        ),
        hint="`train n-step-sarsa 80` · `evaluate 10`. Target mean return ≥ 5.",
        env_spec={
            "type": "gridworld",
            "grid": ["S..", ".#.", "..G"],
            "gamma": 0.9,
            "seed": 22,
            "rewards": {"step": -1, "goal": 10},
        },
        agent_spec={"type": "n-step-sarsa", "n": 3, "alpha": 0.2, "gamma": 0.9, "epsilon": 0.1, "seed": 22},
        goal={"min_eval_return": 4.0, "eval_episodes": 15},
        par=4,
        sequence="prediction",
        order=3,
    )
)

# --- mdp -----------------------------------------------------------------

_register(
    Level(
        id="mdp.discount",
        name="Discount the Future",
        description=(
            "Return G_t = r_{t+1} + γ r_{t+2} + γ² r_{t+3} + …\n"
            "\n"
            "γ close to 0 → myopic. γ close to 1 → far-sighted.\n"
            "This corridor gives +10 at the far goal and -1 every step.\n"
            "\n"
            "Set `gamma` via the env (already 0.9) and reach the goal in few steps.\n"
            "Fewer steps = higher discounted return."
        ),
        hint="A straight line along the top row is optimal. Keep stepping toward G.",
        env_spec={
            "type": "gridworld",
            "grid": ["S....G", "######"],
            "gamma": 0.9,
            "seed": 7,
            "rewards": {"step": -1, "goal": 10},
        },
        agent_spec={"type": "random", "seed": 7},
        goal={"episode_done": True, "max_steps": 8, "min_return": 0.0},
        par=8,
        sequence="mdp",
        order=1,
    )
)

_register(
    Level(
        id="mdp.stochastic",
        name="Slippery Floor",
        description=(
            "Real transitions are stochastic. This grid has slip_prob=0.2:\n"
            "with probability 0.2 you slide perpendicular to your intended move.\n"
            "\n"
            "Still reach G. You cannot fully control the outcome — only the policy.\n"
            "That's the entire RL premise."
        ),
        hint="`train q 40` then let the agent `step` to the goal. Or brute-force `step` until done.",
        env_spec={
            "type": "gridworld",
            "grid": ["S..", "..P", "..G"],
            "gamma": 0.9,
            "slip_prob": 0.2,
            "seed": 8,
            "rewards": {"step": -1, "goal": 10, "pit": -20},
        },
        agent_spec={"type": "random", "seed": 8},
        goal={"episode_done": True},
        par=12,
        sequence="mdp",
        order=2,
    )
)

# --- dp ------------------------------------------------------------------

_register(
    Level(
        id="dp.evaluate",
        name="Policy Evaluation",
        description=(
            "Given a model P(s'|s,a) and a fixed policy π, compute V^π by\n"
            "iterative policy evaluation — the Bellman expectation backup.\n"
            "\n"
            "Load value iteration instead and run it until V looks right:\n"
            "  `train vi 100`\n"
            "  `values`\n"
            "\n"
            "Goal: V(0,0) near the optimal value on this tiny map."
        ),
        hint="`train vi 50` then `values`. V at the start should be clearly positive.",
        env_spec={
            "type": "gridworld",
            "grid": ["S.G"],
            "gamma": 0.9,
            "seed": 9,
            "rewards": {"step": -1, "goal": 10},
        },
        agent_spec={"type": "value-iteration", "gamma": 0.9},
        goal={"value_close": {"(0, 0)": 7.0, "tol": 1.5}},
        par=3,
        sequence="dp",
        order=1,
    )
)

_register(
    Level(
        id="dp.improve",
        name="Policy Improvement",
        description=(
            "Greedy policy improvement: π'(s) = argmax_a Σ P(s'|s,a)[r + γ V(s')]\n"
            "\n"
            "After `train pi` or `train vi`, the policy arrows should point toward G.\n"
            "Check with `policy`."
        ),
        hint="`train vi 50` · `policy`. Every free cell's arrow should walk toward G.",
        env_spec={
            "type": "gridworld",
            "grid": ["S#.", "...", "..G"],
            "gamma": 0.9,
            "seed": 10,
            "rewards": {"step": -1, "goal": 10},
        },
        agent_spec={"type": "value-iteration", "gamma": 0.9},
        goal={
            "policy_match": {
                "(0, 0)": "D",
                "(1, 0)": "D",
                "(1, 1)": "R",
            }
        },
        par=4,
        sequence="dp",
        order=2,
    )
)

_register(
    Level(
        id="dp.vi_vs_pi",
        name="VI vs PI",
        description=(
            "Value iteration and policy iteration both solve known MDPs.\n"
            "\n"
            "Solve this map with `train pi 20` (policy iteration) and prove it works\n"
            "via `evaluate`. Same goal: a competent greedy policy."
        ),
        hint="`train pi 10` · `evaluate 10`. Mean return should be healthy (around +5 to +8).",
        env_spec={
            "type": "gridworld",
            "grid": ["S..P", ".#.P", "..PG"],
            "gamma": 0.9,
            "seed": 11,
            "rewards": {"step": -1, "goal": 10, "pit": -30},
        },
        agent_spec={"type": "policy-iteration", "gamma": 0.9},
        goal={"min_eval_return": 3.0, "eval_episodes": 15},
        par=4,
        sequence="dp",
        order=3,
    )
)

# --- td ------------------------------------------------------------------

_register(
    Level(
        id="td.sarsa",
        name="SARSA On-Policy",
        description=(
            "SARSA updates with the action actually taken next:\n"
            "  Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') − Q(s,a)]\n"
            "\n"
            "It learns the policy it follows (including ε-greedy exploration).\n"
            "\n"
            "Train SARSA on this grid and reach the goal with positive return."
        ),
        hint="`train sarsa 80` · `evaluate 10` or just `step` after training.",
        env_spec={
            "type": "gridworld",
            "grid": ["S.", ".G"],
            "gamma": 0.9,
            "seed": 12,
            "rewards": {"step": -1, "goal": 10},
        },
        agent_spec={"type": "sarsa", "alpha": 0.2, "gamma": 0.9, "epsilon": 0.1, "seed": 12},
        goal={"min_eval_return": 5.0, "eval_episodes": 15},
        par=4,
        sequence="td",
        order=1,
    )
)

_register(
    Level(
        id="td.qlearning",
        name="Q-Learning Off-Policy",
        description=(
            "Q-learning bootstraps with max_a Q(s',a) — it learns the optimal\n"
            "policy while following an exploratory one.\n"
            "\n"
            "  Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') − Q(s,a)]\n"
            "\n"
            "Beat the same map. Then `q` and compare to SARSA's estimates."
        ),
        hint="`train q 80` · `evaluate 10`.",
        env_spec={
            "type": "gridworld",
            "grid": ["S.", ".G"],
            "gamma": 0.9,
            "seed": 13,
            "rewards": {"step": -1, "goal": 10},
        },
        agent_spec={"type": "q-learning", "alpha": 0.2, "gamma": 0.9, "epsilon": 0.1, "seed": 13},
        goal={"min_eval_return": 5.0, "eval_episodes": 15},
        par=4,
        sequence="td",
        order=2,
    )
)

_register(
    Level(
        id="td.expected",
        name="Expected SARSA",
        description=(
            "Expected SARSA replaces the sampled next action with its expectation\n"
            "under the current policy. Lower variance than SARSA.\n"
            "\n"
            "Train it and inspect `curve` — returns should trend up."
        ),
        hint="`set a=0.3` · `train expected 60` · `curve` · `evaluate 10`.",
        env_spec={
            "type": "gridworld",
            "grid": ["S#G", "..."],
            "gamma": 0.9,
            "seed": 14,
            "rewards": {"step": -1, "goal": 10},
        },
        agent_spec={"type": "expected", "alpha": 0.2, "gamma": 0.9, "epsilon": 0.1, "seed": 14},
        goal={"min_eval_return": 4.0, "eval_episodes": 15},
        par=5,
        sequence="td",
        order=3,
    )
)

# --- advanced ------------------------------------------------------------

_register(
    Level(
        id="advanced.cliff",
        name="The Cliff Walk",
        description=(
            "Sutton & Barto's cliff: the safe path is long, the risky path is short.\n"
            "SARSA learns the safe path (it experiences falls). Q-learning learns\n"
            "the optimal risky path (it bootstraps over falls).\n"
            "\n"
            "Solve the cliff with Q-learning. Stay alive (no pit) on evaluation."
        ),
        hint="`train q 200` · `evaluate 20`. Check `policy` — arrows should walk the top then right.",
        env_spec={
            "type": "cliff",
            "grid": ["S.......G", ".........", "PPPPPPPPP"],
            "gamma": 1.0,
            "seed": 15,
        },
        agent_spec={"type": "q-learning", "alpha": 0.2, "gamma": 1.0, "epsilon": 0.1, "seed": 15},
        goal={"min_eval_return": -15.0, "eval_episodes": 20},
        par=5,
        sequence="advanced",
        order=1,
    )
)

_register(
    Level(
        id="advanced.cliff_sarsa",
        name="Cliff — SARSA Plays Safe",
        description=(
            "Same cliff as before. Now use SARSA.\n"
            "\n"
            "SARSA *experiences* the falls (it bootstraps with the action it will\n"
            "actually take, including exploration). It learns the safe path along\n"
            "the top — longer, but never falls.\n"
            "\n"
            "Q-learning (previous level) learns the risky optimal path.\n"
            "Compare `policy` after each.\n"
            "\n"
            "Goal: eval mean return ≥ −20 (safe path is about −16; falling is −100)."
        ),
        hint="`train sarsa 200` · `evaluate 15` · `policy`. Arrows should hug the top row.",
        env_spec={
            "type": "cliff",
            "grid": ["S.......G", ".........", "PPPPPPPPP"],
            "gamma": 1.0,
            "seed": 18,
        },
        agent_spec={"type": "sarsa", "alpha": 0.2, "gamma": 1.0, "epsilon": 0.1, "seed": 18},
        goal={"min_eval_return": -20.0, "eval_episodes": 15},
        par=5,
        sequence="advanced",
        order=2,
    )
)

_register(
    Level(
        id="advanced.double",
        name="Double Q-Learning",
        description=(
            "Maximization bias: max_a Q is an optimistic estimate.\n"
            "Double Q-learning decouples selection and evaluation.\n"
            "\n"
            "Train `double-q` on a noisy map and beat the evaluation bar."
        ),
        hint="`train double-q 100` · `evaluate 20`.",
        env_spec={
            "type": "gridworld",
            "grid": ["S.P.G", "....."],
            "gamma": 0.95,
            "seed": 16,
            "rewards": {"step": -0.1, "goal": 5, "pit": -10},
        },
        agent_spec={"type": "double-q", "alpha": 0.15, "gamma": 0.95, "epsilon": 0.15, "seed": 16},
        goal={"min_eval_return": 0.0, "eval_episodes": 20},
        par=5,
        sequence="advanced",
        order=3,
    )
)

_register(
    Level(
        id="advanced.golf",
        name="Command Golf",
        description=(
            "The LearnGitBranching spirit: solve the level in as few commands as\n"
            "possible. Par is 2.\n"
            "\n"
            "Reach the adjacent goal in one mutating command if you can figure out\n"
            "the economy. `status` shows your golf count."
        ),
        hint="One `step` with a lucky random move can finish a 1-step map… or use `evaluate` only if counted. Prefer `step`.",
        env_spec={"type": "gridworld", "grid": ["SG"], "gamma": 0.9, "seed": 17, "rewards": {"step": -1, "goal": 10}},
        agent_spec={"type": "random", "seed": 17},
        goal={"episode_done": True, "min_return": 5.0, "max_commands": 2},
        par=2,
        sequence="advanced",
        order=4,
    )
)
