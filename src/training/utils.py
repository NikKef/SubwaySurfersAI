"""Utilities for training helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from src.agent import DQNAgent


def update_dqn_hyperparameters(
    model,
    *,
    learning_rate: float,
    gamma: float,
    exploration_fraction: float,
    exploration_final_eps: float,
) -> None:
    """Update learning rate and exploration schedule for a loaded DQN model.

    This is useful when resuming training with new hyper-parameters.
    The optimizer's learning rate is updated, the discount factor is
    refreshed and the epsilon-greedy schedule is reset so that exploration
    anneals according to the new parameters.
    """
    # Update learning rate for the optimizer and schedule
    model.learning_rate = learning_rate
    model.lr_schedule = lambda _: learning_rate
    for param_group in model.policy.optimizer.param_groups:
        param_group["lr"] = learning_rate

    # Update discount factor
    model.gamma = gamma

    # Preserve current exploration rate instead of resetting to 1.0
    current_eps = model.exploration_rate
    prev_start = model.exploration_initial_eps
    prev_end = model.exploration_final_eps
    prev_fraction = model.exploration_fraction

    # Estimate the progress remaining when training was interrupted
    if current_eps > prev_end:
        progress_remaining = 1 - prev_fraction * (
            (current_eps - prev_start) / (prev_end - prev_start)
        )
    else:
        # Annealing already finished
        progress_remaining = 1 - prev_fraction

    # Update target parameters
    model.exploration_initial_eps = current_eps
    model.exploration_final_eps = exploration_final_eps
    model.exploration_fraction = exploration_fraction

    progress_end = 1 - exploration_fraction
    if progress_remaining > progress_end and exploration_fraction > 0:
        remaining_progress = progress_remaining - progress_end

        def exploration_schedule(progress: float) -> float:
            if progress >= progress_remaining:
                return current_eps
            if progress <= progress_end:
                return exploration_final_eps
            slope = (exploration_final_eps - current_eps) / remaining_progress
            return current_eps + slope * (progress - progress_remaining)

        model.exploration_schedule = exploration_schedule
    else:
        # No annealing left; keep exploration rate constant
        model.exploration_schedule = lambda _: current_eps

    model.exploration_rate = current_eps


def load_or_create_dqn_agent(
    model_path: str | Path,
    env,
    **agent_kwargs,
) -> Tuple[DQNAgent, bool]:
    """Return a loaded DQNAgent or create a new one on failure.

    Parameters
    ----------
    model_path:
        Path to the saved model. The associated replay-buffer file will also be
        restored if present. If loading fails (e.g. due to mismatched
        observation spaces), a new agent is instantiated instead.
    env:
        Environment to attach to the agent.
    **agent_kwargs:
        Keyword arguments forwarded to :class:`~src.agent.DQNAgent` when a new
        agent must be created.

    Returns
    -------
    Tuple[DQNAgent, bool]
        The agent and a flag indicating whether the model was successfully
        loaded (``True``) or a new agent was created (``False``).
    """

    try:
        agent = DQNAgent.load(str(model_path), env)
        return agent, True
    except Exception:
        agent = DQNAgent(env, **agent_kwargs)
        return agent, False
