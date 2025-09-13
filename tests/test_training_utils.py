"""Tests for training utility functions."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pytest

from src.agent import DQNAgent
from src.training.utils import (
    update_dqn_hyperparameters,
    load_or_create_dqn_agent,
)


class DummyEnv(gym.Env):
    """Minimal environment returning zero observations."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(2)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action: int):
        obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        reward = 0.0
        terminated = False
        truncated = False
        info: dict = {}
        return obs, reward, terminated, truncated, info


class StackedDummyEnv(DummyEnv):
    """Dummy environment with a different observation shape."""

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 12), dtype=np.uint8
        )


def test_update_dqn_hyperparameters() -> None:
    env = DummyEnv()
    agent = DQNAgent(
        env,
        policy="CnnPolicy",
        buffer_size=1,
        learning_starts=0,
        train_freq=1,
        gradient_steps=1,
    )
    model = agent.model

    # Simulate previous training
    model.exploration_rate = 0.2
    model.policy.optimizer.param_groups[0]["lr"] = 1e-4

    # Progress remaining before update for the default schedule
    prev_progress_remaining = 1 - model.exploration_fraction * (
        (model.exploration_rate - model.exploration_initial_eps)
        / (model.exploration_final_eps - model.exploration_initial_eps)
    )

    update_dqn_hyperparameters(
        model,
        learning_rate=1e-3,
        gamma=0.95,
        exploration_fraction=0.5,
        exploration_final_eps=0.1,
    )

    assert model.learning_rate == pytest.approx(1e-3)
    assert all(
        pg["lr"] == pytest.approx(1e-3) for pg in model.policy.optimizer.param_groups
    )
    assert model.gamma == pytest.approx(0.95)
    assert model.exploration_rate == pytest.approx(0.2)
    assert model.exploration_final_eps == pytest.approx(0.1)
    assert model.exploration_fraction == pytest.approx(0.5)
    schedule = model.exploration_schedule
    # Schedule starts from the previous exploration rate
    assert schedule(prev_progress_remaining) == pytest.approx(0.2)
    # and anneals to the new final epsilon
    assert schedule(0.5) == pytest.approx(0.1)


def test_load_or_create_dqn_agent_handles_space_mismatch(tmp_path) -> None:
    env = DummyEnv()
    agent = DQNAgent(
        env,
        policy="CnnPolicy",
        buffer_size=1,
        learning_starts=0,
        train_freq=1,
        gradient_steps=1,
    )
    model_path = tmp_path / "model.zip"
    agent.save(str(model_path))

    stacked_env = StackedDummyEnv()
    new_agent, loaded = load_or_create_dqn_agent(
        model_path,
        stacked_env,
        policy="CnnPolicy",
        buffer_size=1,
        learning_starts=0,
        train_freq=1,
        gradient_steps=1,
    )
    assert isinstance(new_agent, DQNAgent)
    assert loaded is False
