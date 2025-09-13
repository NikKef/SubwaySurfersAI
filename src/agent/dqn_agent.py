"""Baseline DQN agent using stable-baselines3.

This thin wrapper exposes a minimal API around :class:`stable_baselines3.DQN`
so that training and inference logic can be encapsulated in a lightweight
object. The agent is instantiated with a Gymnasium environment and can be
trained or used to produce actions given observations.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path

from gymnasium import Env
from .prioritized_dqn import PrioritizedDQN
from stable_baselines3.common.callbacks import BaseCallback


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
        self.model = PrioritizedDQN(
            policy, env, verbose=dqn_kwargs.pop("verbose", 0), **dqn_kwargs
        )

    def train(
        self, total_timesteps: int, *, callback: BaseCallback | None = None
    ) -> None:
        """Train the agent for ``total_timesteps`` environment steps.

        Parameters
        ----------
        total_timesteps:
            Number of environment steps to train for.
        callback:
            Optional Stable-Baselines3 callback (e.g. for checkpointing).
        """

        # ``reset_num_timesteps=False`` allows seamless continuation when
        # calling ``learn`` multiple times (e.g. when resuming from a
        # checkpoint).
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            reset_num_timesteps=False,
        )

    def act(self, observation: Any) -> int:
        """Return the action selected by the current policy for ``observation``."""
        action, _ = self.model.predict(observation, deterministic=True)
        return int(action)

    def save(self, path: str) -> None:
        """Persist model parameters and replay buffer to ``path``."""
        self.model.save(path)
        # Save replay buffer alongside the model so that training can resume
        # without losing past experiences. Stable-Baselines3 appends the
        # appropriate extension when saving the buffer.
        self.model.save_replay_buffer(path + "_replay_buffer")

    @classmethod
    def load(cls, path: str, env: Env) -> "DQNAgent":
        """Load a saved agent from ``path`` and attach ``env``.

        Besides the model parameters, this also restores the replay buffer if a
        matching ``*_replay_buffer.pkl`` file is found. This enables seamless
        resumption of training from checkpoints saved with
        :func:`CheckpointCallback` or via :meth:`save`.
        """
        model = PrioritizedDQN.load(path, env=env)
        # Determine potential replay buffer file locations. When checkpoints are
        # saved via ``CheckpointCallback`` the buffer file does not include the
        # ``.zip`` suffix, whereas :meth:`save` keeps the suffix in place.
        p = Path(path)
        candidates = [Path(path + "_replay_buffer.pkl"), p.with_name(p.stem + "_replay_buffer.pkl")]
        for buffer_path in candidates:
            if buffer_path.exists():
                model.load_replay_buffer(str(buffer_path))
                break
        agent = cls.__new__(cls)
        agent.env = env
        agent.model = model
        return agent
