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
        """Load a saved agent from ``path`` and attach ``env``.

        If the observation space of ``env`` differs from the one used when the
        model was saved (e.g. due to a different number of stacked frames),
        the underlying convolutional network is adapted so training can
        continue without starting from scratch.  Additional channels are
        initialized by copying the final existing channel; surplus channels are
        discarded.
        """

        try:
            # Attempt standard loading with environment which performs a strict
            # space compatibility check.
            model = DQN.load(path, env=env)
        except ValueError:
            # Fallback: load without env and reshape the first convolutional
            # layer to match the new number of channels.  This mirrors the
            # logic Stable-Baselines3 would use but keeps as much of the
            # pretrained weights as possible.
            model = DQN.load(path, env=None)

            def _channel_count(shape: tuple[int, ...]) -> int:
                """Heuristically determine channel dimension of ``shape``."""
                if not shape:
                    return 0
                return min(shape)

            old_shape = getattr(model.observation_space, "shape", ())
            new_shape = getattr(env.observation_space, "shape", ())

            if (
                old_shape
                and new_shape
                and sorted(old_shape) == sorted(new_shape)
                and old_shape != new_shape
            ):
                import numpy as np
                import gymnasium as gym
                from gymnasium import spaces
                from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper

                def _find_perm(src: tuple[int, ...], dst: tuple[int, ...]) -> list[int]:
                    """Return axes permutation to transform ``dst`` into ``src``."""
                    perm: list[int] = []
                    used = [False] * len(dst)
                    for dim in src:
                        for i, d in enumerate(dst):
                            if not used[i] and d == dim:
                                perm.append(i)
                                used[i] = True
                                break
                    return perm

                axes = _find_perm(old_shape, new_shape)
                if axes and len(axes) == len(old_shape):
                    if isinstance(env, VecEnv):
                        from stable_baselines3.common.vec_env import VecTransposeImage

                        if axes == [2, 0, 1]:
                            env = VecTransposeImage(env)
                        else:
                            class _VecTransposeObs(VecEnvWrapper):
                                def __init__(self, venv: VecEnv):
                                    obs_space = venv.observation_space
                                    assert isinstance(obs_space, spaces.Box)
                                    transposed = spaces.Box(
                                        low=np.transpose(obs_space.low, axes),
                                        high=np.transpose(obs_space.high, axes),
                                        dtype=obs_space.dtype,
                                    )
                                    super().__init__(venv, observation_space=transposed)
                                    self.axes = tuple(axes)

                                def reset(self):
                                    obs = self.venv.reset()
                                    return np.transpose(obs, (0, *self.axes))

                                def step_wait(self):
                                    obs, rewards, dones, infos = self.venv.step_wait()
                                    obs = np.transpose(obs, (0, *self.axes))
                                    for idx, done in enumerate(dones):
                                        if done and "terminal_observation" in infos[idx]:
                                            infos[idx]["terminal_observation"] = np.transpose(
                                                infos[idx]["terminal_observation"], self.axes
                                            )
                                    return obs, rewards, dones, infos

                            env = _VecTransposeObs(env)
                    else:
                        class _TransposeObs(gym.ObservationWrapper):
                            def __init__(self, env: gym.Env):
                                super().__init__(env)
                                self.axes = tuple(axes)
                                obs_space = env.observation_space
                                if isinstance(obs_space, spaces.Box):
                                    self.observation_space = spaces.Box(
                                        low=np.transpose(obs_space.low, self.axes),
                                        high=np.transpose(obs_space.high, self.axes),
                                        dtype=obs_space.dtype,
                                    )

                            def observation(self, observation: np.ndarray) -> np.ndarray:
                                return np.transpose(observation, self.axes)

                        env = _TransposeObs(env)

                    new_shape = getattr(env.observation_space, "shape", ())

            old_c = _channel_count(old_shape)
            new_c = _channel_count(new_shape)

            if old_c != new_c:
                import torch
                import torch.nn as nn

                def _adjust_conv(net) -> None:
                    fe = getattr(net, "features_extractor", None)
                    cnn = getattr(fe, "cnn", None)
                    if cnn is None or not isinstance(cnn[0], nn.Conv2d):
                        return
                    old_conv: nn.Conv2d = cnn[0]
                    new_conv = nn.Conv2d(
                        new_c,
                        old_conv.out_channels,
                        old_conv.kernel_size,
                        old_conv.stride,
                        old_conv.padding,
                    )
                    with torch.no_grad():
                        # Copy existing channels
                        num_copy = min(old_c, new_c)
                        new_conv.weight[:, :num_copy] = old_conv.weight[:, :num_copy]
                        if new_c > old_c:
                            # Repeat the last channel for any additional ones
                            for i in range(old_c, new_c):
                                new_conv.weight[:, i] = old_conv.weight[:, old_c - 1]
                        new_conv.bias[:] = old_conv.bias
                    cnn[0] = new_conv

                # Adjust both the online and target networks (handle attribute
                # naming differences across SB3 versions).
                for net_attr in ["q_net", "q_net_target", "target_q_net"]:
                    net = getattr(model, net_attr, None)
                    if net is not None:
                        _adjust_conv(net)
                if hasattr(model, "policy"):
                    for net_attr in ["q_net", "q_net_target", "target_q_net"]:
                        net = getattr(model.policy, net_attr, None)
                        if net is not None:
                            _adjust_conv(net)

            # Update spaces and attach environment
            model.observation_space = env.observation_space
            if hasattr(model, "policy"):
                model.policy.observation_space = env.observation_space
            model.set_env(env)

        agent = cls.__new__(cls)
        agent.env = env
        agent.model = model
        return agent
