"""Utilities for training helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from stable_baselines3.common.utils import LinearSchedule

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

    # Reset exploration schedule
    model.exploration_initial_eps = 1.0
    model.exploration_final_eps = exploration_final_eps
    model.exploration_fraction = exploration_fraction
    model.exploration_schedule = LinearSchedule(
        model.exploration_initial_eps,
        model.exploration_final_eps,
        model.exploration_fraction,
    )
    model.exploration_rate = model.exploration_initial_eps


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
