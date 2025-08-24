"""Baseline DQN agent using stable-baselines3.

This thin wrapper exposes a minimal API around :class:`stable_baselines3.DQN`
so that training and inference logic can be encapsulated in a lightweight
object. The agent is instantiated with a Gymnasium environment and can be
trained or used to produce actions given observations.
"""

from __future__ import annotations

from typing import Any

from gymnasium import Env
from stable_baselines3 import DQN


class DQNAgent:
    """Deep Q-Network agent powered by Stable-Baselines3."""

    def __init__(
        self, env: Env, *, policy: str = "CnnPolicy", **dqn_kwargs: Any
    ) -> None:
        """Create a new DQN agent.

        Parameters
        ----------
        env:
            Gymnasium environment the agent will interact with.
        policy:
            Policy architecture. ``"CnnPolicy"`` expects image observations while
            ``"MlpPolicy"`` works with vector observations. Defaults to
            ``"CnnPolicy"``.
        **dqn_kwargs:
            Additional keyword arguments forwarded to
            :class:`stable_baselines3.DQN`.
        """

        self.env = env
        # ``verbose`` defaults to 0 if not provided to keep logs quiet by default.
        self.model = DQN(
            policy, env, verbose=dqn_kwargs.pop("verbose", 0), **dqn_kwargs
        )

    def train(self, total_timesteps: int) -> None:
        """Train the agent for ``total_timesteps`` environment steps."""
        self.model.learn(total_timesteps=total_timesteps)

    def act(self, observation: Any) -> int:
        """Return the action selected by the current policy for ``observation``."""
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)

    def save(self, path: str) -> None:
        """Persist model parameters to ``path``."""
        self.model.save(path)

    @classmethod
    def load(cls, path: str, env: Env) -> "DQNAgent":
        """Load a saved agent from ``path`` and attach ``env``."""
        model = DQN.load(path, env=env)
        agent = cls.__new__(cls)
        agent.env = env
        agent.model = model
        return agent
