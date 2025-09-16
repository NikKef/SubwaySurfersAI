"""Training callbacks.

This module contains callbacks used during training, such as logging
per-episode metrics to TensorBoard.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback


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


class LatestCheckpointCallback(CheckpointCallback):
    """Keep only the most recent checkpoint on disk."""

    _STEP_PATTERN = re.compile(r"_(\d+)_steps")

    def _init_callback(self) -> None:
        super()._init_callback()
        self._prune_existing_checkpoints()

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = Path(self._checkpoint_path(extension="zip"))
            self.model.save(str(model_path))
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {model_path}")

            replay_buffer_path: Path | None = None
            if (
                self.save_replay_buffer
                and hasattr(self.model, "replay_buffer")
                and self.model.replay_buffer is not None
            ):
                replay_buffer_path = Path(
                    self._checkpoint_path("replay_buffer_", extension="pkl")
                )
                self.model.save_replay_buffer(str(replay_buffer_path))
                if self.verbose > 1:
                    print(
                        "Saving model replay buffer checkpoint to"
                        f" {replay_buffer_path}"
                    )

            vecnormalize_path: Path | None = None
            vec_env = self.model.get_vec_normalize_env()
            if self.save_vecnormalize and vec_env is not None:
                vecnormalize_path = Path(
                    self._checkpoint_path("vecnormalize_", extension="pkl")
                )
                vec_env.save(str(vecnormalize_path))
                if self.verbose >= 2:
                    print(
                        "Saving model VecNormalize to"
                        f" {vecnormalize_path}"
                    )

            self._cleanup_old_checkpoints(
                keep_model=model_path,
                keep_replay_buffer=replay_buffer_path,
                keep_vecnormalize=vecnormalize_path,
            )

        return True

    def _prune_existing_checkpoints(self) -> None:
        if self.save_path is None:
            return

        checkpoint_dir = Path(self.save_path)
        latest_model = self._find_latest_checkpoint(
            checkpoint_dir, suffix="zip", checkpoint_type=""
        )
        latest_step = self._extract_step(latest_model) if latest_model else None

        latest_replay = self._find_latest_checkpoint(
            checkpoint_dir,
            suffix="pkl",
            checkpoint_type="replay_buffer_",
            preferred_step=latest_step,
        )
        latest_vecnormalize = self._find_latest_checkpoint(
            checkpoint_dir,
            suffix="pkl",
            checkpoint_type="vecnormalize_",
            preferred_step=latest_step,
        )

        self._cleanup_old_checkpoints(
            keep_model=latest_model,
            keep_replay_buffer=latest_replay,
            keep_vecnormalize=latest_vecnormalize,
        )

    def _find_latest_checkpoint(
        self,
        checkpoint_dir: Path,
        *,
        suffix: str,
        checkpoint_type: str,
        preferred_step: int | None = None,
    ) -> Path | None:
        pattern = f"{self.name_prefix}_{checkpoint_type}*_steps.{suffix}"
        candidates = sorted(
            checkpoint_dir.glob(pattern),
            key=self._extract_step,
            reverse=True,
        )
        if not candidates:
            return None

        if preferred_step is not None:
            for candidate in candidates:
                if self._extract_step(candidate) == preferred_step:
                    return candidate

        return candidates[0]

    def _extract_step(self, path: Path | None) -> int:
        if path is None:
            return -1
        match = self._STEP_PATTERN.search(path.name)
        if match is None:
            return -1
        return int(match.group(1))

    def _cleanup_old_checkpoints(
        self,
        *,
        keep_model: Path | None,
        keep_replay_buffer: Path | None,
        keep_vecnormalize: Path | None,
    ) -> None:
        if self.save_path is None:
            return

        checkpoint_dir = Path(self.save_path)

        for path in checkpoint_dir.glob(f"{self.name_prefix}_*_steps.zip"):
            if keep_model is None or path != keep_model:
                path.unlink(missing_ok=True)

        for path in checkpoint_dir.glob(
            f"{self.name_prefix}_replay_buffer_*_steps.pkl"
        ):
            if keep_replay_buffer is None or path != keep_replay_buffer:
                path.unlink(missing_ok=True)

        for path in checkpoint_dir.glob(
            f"{self.name_prefix}_vecnormalize_*_steps.pkl"
        ):
            if keep_vecnormalize is None or path != keep_vecnormalize:
                path.unlink(missing_ok=True)
