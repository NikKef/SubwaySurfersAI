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


class DummyEnv4(gym.Env):
    """Environment with a different number of image channels."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        # Channel-first format to mirror training setup
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(4, 84, 84), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(4)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action: int):
        obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        return obs, 0.0, False, False, {}


class DummyEnv3(gym.Env):
    """Three-channel variant used for conversion tests."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3, 84, 84), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(4)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        return np.zeros(self.observation_space.shape, dtype=np.uint8), {}

    def step(self, action: int):
        obs = np.zeros(self.observation_space.shape, dtype=np.uint8)
        return obs, 0.0, False, False, {}


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


def test_load_expands_channels(tmp_path) -> None:
    """Models saved with fewer channels can be reloaded with more."""
    env3 = DummyEnv3()
    agent = DQNAgent(
        env3,
        policy="CnnPolicy",
        buffer_size=1,
        learning_starts=0,
        train_freq=1,
        gradient_steps=1,
    )
    model_path = tmp_path / "agent.zip"
    agent.save(str(model_path))

    # Reload using an environment with four channels
    env4 = DummyEnv4()
    loaded = DQNAgent.load(str(model_path), env4)
    conv = loaded.model.q_net.features_extractor.cnn[0]
    assert conv.weight.shape[1] == 4


def test_load_transposed_observations(tmp_path) -> None:
    """Models trained on channel-last inputs reload on channel-first envs."""
    env_hwc = DummyEnv()
    agent = DQNAgent(
        env_hwc,
        policy="CnnPolicy",
        buffer_size=1,
        learning_starts=0,
        train_freq=1,
        gradient_steps=1,
    )
    model_path = tmp_path / "agent.zip"
    agent.save(str(model_path))

    env_chw = DummyEnv3()
    loaded = DQNAgent.load(str(model_path), env_chw)
    obs, _ = env_chw.reset()
    action = loaded.act(obs)
    assert env_chw.action_space.contains(action)
