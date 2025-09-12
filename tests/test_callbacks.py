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
    mock_logger.record.assert_any_call("episode/reward", 1.0)
    mock_logger.record.assert_any_call("episode/length", 2.0)
    mock_logger.dump.assert_called_once_with(step=42)
