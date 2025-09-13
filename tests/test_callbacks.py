"""Tests for training callbacks."""

from __future__ import annotations

from unittest.mock import Mock

from src.training import EpisodeMetricsCallback


def test_episode_metrics_callback_logs():
    cb = EpisodeMetricsCallback()
    mock_logger = Mock()
    cb.model = Mock()
    cb.model.logger = mock_logger
    cb.locals = {
        "dones": [True],
        "infos": [{"episode_reward": 1.0, "episode_length": 2}],
    }
    cb.num_timesteps = 42
    cb._on_step()
    mock_logger.record.assert_any_call(
        "episode/reward", 1.0, exclude="stdout"
    )
    mock_logger.record.assert_any_call(
        "episode/length", 2.0, exclude="stdout"
    )
    mock_logger.dump.assert_called_once_with(step=42)


def test_episode_metrics_callback_rollout_averages():
    cb = EpisodeMetricsCallback()
    mock_logger = Mock()
    cb.model = Mock()
    cb.model.logger = mock_logger
    cb._episode_rewards = [1.0, 2.0]
    cb._episode_lengths = [10.0, 20.0]
    cb._on_rollout_end()
    mock_logger.record.assert_any_call("rollout/avg_reward", 1.5)
    mock_logger.record.assert_any_call("rollout/avg_length", 15.0)
