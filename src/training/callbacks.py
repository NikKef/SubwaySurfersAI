"""Training callbacks.

This module contains callbacks used during training, such as logging
per-episode metrics to TensorBoard.
"""

from __future__ import annotations

from typing import List

import logging
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeMetricsCallback(BaseCallback):
    """Log per-episode metrics and running averages.

    Episode reward and length are recorded for TensorBoard after each game
    without printing to stdout.  At the end of each rollout the callback
    logs the average reward and length so they appear in the regular
    ``rollout/`` statistics.
    """

    def __init__(self) -> None:
        super().__init__()
        self._episode_rewards: List[float] = []
        self._episode_lengths: List[float] = []

    def _on_step(self) -> bool:  # pragma: no cover - simple wrapper
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is None or infos is None:
            return True
        for done, info in zip(dones, infos):
            if done:
                reward = info.get("episode_reward")
                length = info.get("episode_length")
                has_log = False
                if reward is not None:
                    self.logger.record(
                        "episode/reward", float(reward), exclude="stdout"
                    )
                    self._episode_rewards.append(float(reward))
                    has_log = True
                if length is not None:
                    self.logger.record(
                        "episode/length", float(length), exclude="stdout"
                    )
                    self._episode_lengths.append(float(length))
                    has_log = True
                if has_log:
                    # Dump so values appear in TensorBoard without printing
                    # a summary box to stdout.
                    self.logger.dump(step=self.num_timesteps)
                    if reward is not None and length is not None:
                        logging.info(
                            "Episode finished: reward=%.2f length=%d",
                            float(reward),
                            int(length),
                        )
        return True

    def _on_rollout_end(self) -> None:  # pragma: no cover - simple wrapper
        if self._episode_rewards:
            avg_reward = sum(self._episode_rewards) / len(self._episode_rewards)
            self.logger.record("rollout/avg_reward", avg_reward)
            self._episode_rewards.clear()
        if self._episode_lengths:
            avg_length = sum(self._episode_lengths) / len(self._episode_lengths)
            self.logger.record("rollout/avg_length", avg_length)
            self._episode_lengths.clear()
