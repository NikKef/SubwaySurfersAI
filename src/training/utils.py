"""Utilities for training helpers."""

from __future__ import annotations

from stable_baselines3.common.utils import LinearSchedule


def update_dqn_hyperparameters(
    model,
    *,
    learning_rate: float,
    exploration_fraction: float,
    exploration_final_eps: float,
) -> None:
    """Update learning rate and exploration schedule for a loaded DQN model.

    This is useful when resuming training with new hyper-parameters.
    The optimizer's learning rate is updated and the epsilon-greedy schedule
    is reset so that exploration anneals according to the new parameters.
    """
    # Update learning rate for the optimizer and schedule
    model.learning_rate = learning_rate
    model.lr_schedule = lambda _: learning_rate
    for param_group in model.policy.optimizer.param_groups:
        param_group["lr"] = learning_rate

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
