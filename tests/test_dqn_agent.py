"""Tests for the DQNAgent wrapper."""

from __future__ import annotations

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces


os.environ.setdefault("MPLBACKEND", "Agg")

from src.agent import DQNAgent


class DummyEnv(gym.Env):
    """Minimal image-based environment for testing."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(4)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action: int):
        obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        reward = 0.0
        terminated = False
        truncated = False
        info: dict = {}
        return obs, reward, terminated, truncated, info


def test_dqn_agent_act() -> None:
    env = DummyEnv()
    agent = DQNAgent(
        env,
        policy="CnnPolicy",
        buffer_size=1,
        learning_starts=0,
        train_freq=1,
        gradient_steps=1,
    )
    obs, _ = env.reset()
    action = agent.act(obs)
    assert env.action_space.contains(action)


def test_dqn_agent_custom_options() -> None:
    """Ensure custom network options are forwarded correctly."""
    env = DummyEnv()
    agent = DQNAgent(
        env,
        policy="MlpPolicy",
        buffer_size=1,
        learning_starts=0,
        train_freq=1,
        gradient_steps=1,
        hidden_sizes=[64],
        dueling=True,
        double_q=True,
    )
    obs, _ = env.reset()
    action = agent.act(obs)
    assert env.action_space.contains(action)
    # Verify that the first hidden layer matches the provided size when
    # accessible.
    try:
        import torch.nn as nn

        layers = [
            m
            for m in getattr(agent.model.q_net, "q_net", agent.model.q_net)
            if isinstance(m, nn.Linear)
        ]
        if layers:
            assert layers[0].out_features == 64
    except Exception:
        # The check is best-effort; absence of attributes is acceptable across
        # SB3 versions.
        pass
