"""Tests for training utility functions."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pytest

from src.agent import DQNAgent
from src.training.utils import update_dqn_hyperparameters


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

    update_dqn_hyperparameters(
        model,
        learning_rate=1e-3,
        exploration_fraction=0.5,
        exploration_final_eps=0.1,
    )

    assert model.learning_rate == pytest.approx(1e-3)
    assert all(
        pg["lr"] == pytest.approx(1e-3) for pg in model.policy.optimizer.param_groups
    )
    assert model.exploration_rate == pytest.approx(1.0)
    assert model.exploration_final_eps == pytest.approx(0.1)
    assert model.exploration_fraction == pytest.approx(0.5)
    schedule = model.exploration_schedule
    assert schedule(1.0) == pytest.approx(1.0)
    assert schedule(0.0) == pytest.approx(0.1)
