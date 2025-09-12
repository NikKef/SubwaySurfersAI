"""Training callbacks.

This module contains callbacks used during training, such as logging
per-episode metrics to TensorBoard.
"""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback


class EpisodeMetricsCallback(BaseCallback):
    """Log episode reward and length to TensorBoard."""

    def _on_step(self) -> bool:  # pragma: no cover - simple wrapper
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is None or infos is None:
            return True
        for done, info in zip(dones, infos):
            if done:
                reward = info.get("episode_reward")
                length = info.get("episode_length")
                if reward is not None:
                    self.logger.record("episode/reward", float(reward))
                if length is not None:
                    self.logger.record("episode/length", float(length))
                self.logger.dump(step=self.num_timesteps)
        return True
