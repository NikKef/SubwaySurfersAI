"""Baseline DQN agent using stable-baselines3.

This thin wrapper exposes a minimal API around :class:`stable_baselines3.DQN`
so that training and inference logic can be encapsulated in a lightweight
object. The agent is instantiated with a Gymnasium environment and can be
trained or used to produce actions given observations.
"""

from __future__ import annotations

from typing import Any, Sequence

from gymnasium import Env
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.dqn.policies import DQNPolicy
import inspect


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
        verbose = dqn_kwargs.pop("verbose", 0)

        # Merge custom policy kwargs with defaults.
        policy_kwargs = dqn_kwargs.pop("policy_kwargs", {})

        hidden_sizes: Sequence[int] | None = dqn_kwargs.pop("hidden_sizes", None)
        if hidden_sizes is not None:
            policy_kwargs["net_arch"] = list(hidden_sizes)
        else:
            policy_kwargs.setdefault("net_arch", [256, 256])

        # ``dueling`` is only supported in some versions of SB3; include it when
        # available and requested.
        if "dueling" in inspect.signature(DQNPolicy.__init__).parameters:
            policy_kwargs["dueling"] = dqn_kwargs.pop("dueling", True)
        else:
            dqn_kwargs.pop("dueling", None)

        dqn_kwargs["policy_kwargs"] = policy_kwargs

        # ``double_q`` is optional depending on SB3 version; default to ``True``
        # when supported.
        if "double_q" in inspect.signature(DQN.__init__).parameters:
            dqn_kwargs.setdefault("double_q", True)
        else:
            dqn_kwargs.pop("double_q", None)

        # ``verbose`` defaults to 0 if not provided to keep logs quiet by default.
        self.model = DQN(policy, env, verbose=verbose, **dqn_kwargs)

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
