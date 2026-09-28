"""World session: environment + agent + episode + level + golf counter."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from learnrl.agents import Agent, PlanningAgent, agent_from_spec
from learnrl.agents.tabular import QAgent
from learnrl.envs import Env, FiniteMDP, env_from_spec
from learnrl.engine.history import History
from learnrl.types import Action, EvalResult, GoalCheck, State, StepResult, Transition


@dataclass
class EpisodeStats:
    """Running totals for the current/last episode.

    Attributes:
        steps: Steps taken in the current episode.
        reward: Sum of rewards in the current episode.
        done: Whether the current episode has terminated.
        history: Transition log for the current episode.
    """

    steps: int = 0
    reward: float = 0.0
    done: bool = False
    history: list[Transition] = field(default_factory=list)


class Session:
    """Mutable sandbox world manipulated by the command layer.

    The session owns the environment, agent, episode bookkeeping, command
    golf counter, optional loaded level, and an undo history. Commands never
    mutate the environment directly — they go through methods here so undo
    can snapshot consistently.

    Args:
        env: Active environment.
        agent: Active agent/policy.
        level: Optional loaded level definition (see :mod:`learnrl.levels`).
    """

    def __init__(
        self,
        env: Env,
        agent: Agent,
        *,
        level: Any | None = None,
    ) -> None:
        self.env = env
        self.agent = agent
        self.level = level
        self.episode = EpisodeStats()
        self.commands_used = 0
        self.history = History()
        self.last_eval: EvalResult | None = None
        self.last_train_metrics: dict[str, float] = {}
        self.learning_curve: list[float] = []
        self.log: list[str] = []
        self._state: State | None = None
        self._world_gen = 0
        self._eval_gen: int | None = None
        self._state = self.env.reset(seed=getattr(env, "_seed", 0))

    # -- episode control -----------------------------------------------

    @property
    def state(self) -> State | None:
        """Current environment state, or ``None`` if not started."""
        return self._state

    def reset_episode(self, *, seed: int | None = None) -> State:
        """Start a fresh episode.

        Args:
            seed: Optional seed for reproducibility.

        Returns:
            The initial state.
        """
        self._push_undo()
        self.episode = EpisodeStats()
        self.agent.reset_episode()
        self._state = self.env.reset(seed=seed)
        return self._state

    def reset_all(self) -> None:
        """Hard reset: new episode, cleared agent tables if possible, new golf.

        Used by the ``reset`` command (like LearnGitBranching's ``reset``).
        """
        self._push_undo()
        self.episode = EpisodeStats()
        self.commands_used = 0
        self.last_eval = None
        self.last_train_metrics = {}
        self.learning_curve.clear()
        self.agent.reset_episode()
        # Re-create the agent from its own spec to drop learned tables.
        try:
            fresh = agent_from_spec(self.agent.spec(), actions=list(self.env.actions))
            self.agent = fresh
        except Exception:
            pass
        self._state = self.env.reset(seed=getattr(self.env, "_seed", 0))
        if self.level is not None:
            self.log.append("Level reset to its initial configuration.")

    def step(self, action: Action | None = None) -> StepResult:
        """Take one environment step.

        Args:
            action: Explicit action, or ``None`` to let the agent choose.

        Returns:
            The step result.

        Raises:
            RuntimeError: If the episode is already done.
        """
        self._push_undo()
        if self._state is None:
            self._state = self.env.reset()
        if self.episode.done:
            raise RuntimeError("episode finished — use `reset` or `step` after `reset`")
        act = action if action is not None else self.agent.act(self._state, explore=True)
        if isinstance(self.agent, QAgent) and action is None:
            # SARSA bootstrap needs the action actually taken.
            self.agent.set_next_action(act)
        result = self.env.step(act)
        transition: Transition = (result.prev_state, result.action, result.reward, result.state, result.done)
        self.episode.history.append(transition)
        self.episode.steps += 1
        self.episode.reward += result.reward
        self.episode.done = result.done
        metrics = self.agent.learn(transition)
        self.last_train_metrics = metrics
        self._state = result.state
        if result.done:
            self.agent.reset_episode()
        return result

    def step_many(self, n: int, action: Action | None = None) -> list[StepResult]:
        """Take up to ``n`` steps, stopping if the episode ends.

        Args:
            n: Maximum steps.
            action: Optional forced action (applied only on the first step
                when the agent would otherwise choose; subsequent steps use
                the agent). Prefer ``step`` for full manual control.

        Returns:
            List of step results actually performed.
        """
        results: list[StepResult] = []
        for i in range(max(0, n)):
            if self.episode.done:
                break
            results.append(self.step(action if i == 0 else None))
        return results

    def run_episode(self, *, explore: bool = True, max_steps: int = 1000) -> list[StepResult]:
        """Run one full episode under the current policy.

        Args:
            explore: Whether the agent may explore.
            max_steps: Safety cap on episode length.

        Returns:
            Step results in order.
        """
        if self.episode.done or self._state is None:
            self.reset_episode()
        results: list[StepResult] = []
        self.agent.reset_episode()
        for _ in range(max_steps):
            if self.episode.done or self._state is None:
                break
            # One undo entry per command, not per micro-step.
            act = self.agent.act(self._state, explore=explore)
            if isinstance(self.agent, QAgent):
                self.agent.set_next_action(act)
            result = self.env.step(act)
            transition: Transition = (
                result.prev_state,
                result.action,
                result.reward,
                result.state,
                result.done,
            )
            self.episode.history.append(transition)
            self.episode.steps += 1
            self.episode.reward += result.reward
            self.episode.done = result.done
            self.agent.learn(transition)
            self._state = result.state
            results.append(result)
            if result.done:
                break
        self.agent.reset_episode()
        return results

    def evaluate(self, *, episodes: int = 20, max_steps: int = 1000, explore: bool = False) -> EvalResult:
        """Evaluate the greedy (or exploratory) policy over many episodes.

        Args:
            episodes: Number of evaluation episodes.
            max_steps: Safety cap per episode.
            explore: Use exploration during evaluation.

        Returns:
            Aggregated :class:`~learnrl.types.EvalResult`. The sandbox episode
            state is restored afterwards.
        """
        self._push_undo()
        saved = self._snapshot()
        returns: list[float] = []
        lengths: list[int] = []
        for i in range(max(1, episodes)):
            state = self.env.reset(seed=10_000 + i)
            total = 0.0
            length = 0
            self.agent.reset_episode()
            for _ in range(max_steps):
                if self.env.is_terminal(state):
                    break
                act = self.agent.act(state, explore=explore)
                result = self.env.step(act)
                total += result.reward
                length += 1
                state = result.state
                if result.done:
                    break
            returns.append(total)
            lengths.append(length)
        self._restore(saved)
        mean_r = sum(returns) / len(returns)
        mean_l = sum(lengths) / len(lengths)
        self.last_eval = EvalResult(
            episodes=len(returns),
            mean_return=mean_r,
            min_return=min(returns),
            max_return=max(returns),
            mean_length=mean_l,
            returns=returns,
        )
        self._eval_gen = self._world_gen
        return self.last_eval

    def train(
        self,
        *,
        algorithm: str | None = None,
        iterations: int = 100,
        episodes: int | None = None,
    ) -> dict[str, float]:
        """Train the agent via planning (DP) or TD episodes.

        Args:
            algorithm: Optional switch — ``vi``, ``pi``, ``q``, ``sarsa``,
                ``expected``, ``bandit``. ``None`` keeps the current agent.
            iterations: Planning iterations or default episode count.
            episodes: Override episode count for TD methods.

        Returns:
            Diagnostics including mean return of the last batch.

        Raises:
            RuntimeError: If DP is requested on a non-FiniteMDP environment.
            ValueError: If ``algorithm`` is unknown.
        """
        self._push_undo()
        self.last_eval = None
        if algorithm:
            self._swap_agent(algorithm)

        if isinstance(self.agent, PlanningAgent):
            if not isinstance(self.env, FiniteMDP):
                raise RuntimeError("planning requires a finite MDP environment")
            metrics = self.agent.plan(self.env, iterations=iterations)
            self.last_train_metrics = metrics
            return metrics

        n_eps = episodes if episodes is not None else max(1, iterations)
        returns: list[float] = []
        for i in range(n_eps):
            self.reset_episode(seed=1_000 + i)
            results = self.run_episode(explore=True)
            returns.append(self.episode.reward)
            self.learning_curve.append(self.episode.reward)
            _ = results
        metrics = {
            "episodes": float(n_eps),
            "mean_return": sum(returns) / len(returns) if returns else 0.0,
            "last_return": returns[-1] if returns else 0.0,
        }
        self.last_train_metrics = metrics
        return metrics

    def set_hyperparams(self, **kwargs: float) -> dict[str, float]:
        """Set agent hyperparameters (alpha/gamma/epsilon) if supported.

        Args:
            **kwargs: ``alpha``, ``gamma``, and/or ``epsilon``.

        Returns:
            The values actually applied.

        Raises:
            RuntimeError: If the agent has no hyperparameters.
        """
        self._push_undo()
        if not isinstance(self.agent, QAgent):
            # Try bandit epsilon.
            from learnrl.agents.tabular import BanditAgent

            if isinstance(self.agent, BanditAgent) and "epsilon" in kwargs:
                self.agent._epsilon = float(kwargs["epsilon"])  # noqa: SLF001
                return {"epsilon": self.agent._epsilon}
            raise RuntimeError(f"agent {self.agent.name!r} has no tunable hyperparameters")
        self.agent.set_hyperparams(
            alpha=kwargs.get("alpha"),
            gamma=kwargs.get("gamma"),
            epsilon=kwargs.get("epsilon"),
        )
        return {
            "alpha": self.agent.alpha,
            "gamma": self.agent.gamma,
            "epsilon": self.agent.epsilon,
        }

    def check_goal(self) -> GoalCheck:
        """Evaluate the loaded level goal against the current world.

        Returns:
            A :class:`~learnrl.types.GoalCheck`. If no level is loaded, the
            goal is considered satisfied with a note.
        """
        if self.level is None:
            return GoalCheck(True, ("No level loaded — sandbox mode."))
        return self.level.check_goal(self)

    def undo(self) -> bool:
        """Undo the last mutating command.

        Returns:
            ``True`` if something was undone.
        """
        return self.history.undo(self._restore)

    def note(self, message: str) -> None:
        """Append a short message to the session log.

        Args:
            message: Text to record.
        """
        self.log.append(message)
        if len(self.log) > 200:
            self.log = self.log[-200:]

    # -- internals -----------------------------------------------------

    def _push_undo(self) -> None:
        self._world_gen += 1
        self.history.push(self._snapshot())

    def _snapshot(self) -> dict[str, Any]:
        return {
            "env": copy.deepcopy(self.env),
            "agent": self.agent,
            "agent_snap": self.agent.snapshot(),
            "episode": copy.deepcopy(self.episode),
            "commands_used": self.commands_used,
            "last_eval": self.last_eval,
            "last_train_metrics": dict(self.last_train_metrics),
            "learning_curve": list(self.learning_curve),
            "log": list(self.log),
            "state": self._state,
        }

    def _restore(self, snap: Mapping[str, Any]) -> None:
        self.env = snap["env"]
        self.agent = snap["agent"]
        try:
            self.agent.restore(snap["agent_snap"])
        except Exception:
            # Fallback: keep the deep-copied agent object as-is.
            pass
        self.episode = copy.deepcopy(snap["episode"])
        self.commands_used = int(snap["commands_used"])
        self.last_eval = snap["last_eval"]
        self.last_train_metrics = dict(snap["last_train_metrics"])
        self.learning_curve = list(snap["learning_curve"])
        self.log = list(snap["log"])
        self._state = snap["state"]

    def _swap_agent(self, algorithm: str) -> None:
        algo = algorithm.lower()
        actions = list(self.env.actions)
        if algo in ("vi", "value-iteration"):
            self.agent = agent_from_spec({"type": "value-iteration"}, actions=actions)
        elif algo in ("pi", "policy-iteration"):
            self.agent = agent_from_spec({"type": "policy-iteration"}, actions=actions)
        elif algo in ("q", "q-learning", "double-q"):
            self.agent = agent_from_spec({"type": algo}, actions=actions)
        elif algo in ("sarsa", "expected", "expected-sarsa"):
            self.agent = agent_from_spec({"type": algo}, actions=actions)
        elif algo in ("mc", "mc-prediction", "td", "td-prediction", "n-step-sarsa", "nstep"):
            self.agent = agent_from_spec({"type": algo}, actions=actions)
        elif algo in ("bandit", "bandit-epsilon"):
            self.agent = agent_from_spec({"type": "bandit-epsilon", "n_arms": len(actions)}, actions=actions)
        elif algo in ("random",):
            self.agent = agent_from_spec({"type": "random"}, actions=actions)
        else:
            raise ValueError(f"unknown algorithm {algorithm!r}")


def default_session(*, env_name: str = "gridworld", seed: int = 0) -> Session:
    """Create a sandbox session with a small default gridworld.

    Args:
        env_name: ``gridworld`` (default), ``cliff``, or ``bandit``.
        seed: Environment seed.

    Returns:
        A ready-to-use :class:`Session`.

    Examples:
        >>> s = default_session()
        >>> s.agent.name
        'random'
    """
    if env_name == "bandit":
        spec: dict[str, Any] = {"type": "bandit", "seed": seed}
    elif env_name == "cliff":
        spec = {"type": "cliff", "seed": seed}
    else:
        spec = {
            "type": "gridworld",
            "grid": ["S..", ".#.", "..G"],
            "seed": seed,
        }
    env = env_from_spec(spec)
    agent = agent_from_spec({"type": "random", "seed": seed}, actions=list(env.actions))
    return Session(env, agent)
