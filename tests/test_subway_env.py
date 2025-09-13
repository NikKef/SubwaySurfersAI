from __future__ import annotations

import io
import time
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image, ImageDraw
import logging

from src.env import SubwaySurfersEnv, ADBController
from src.env.subway_env import PLAY_BUTTON_COORD, CRASH_DISMISS_COORD


def _fake_png(color) -> bytes:
    if isinstance(color, int):
        color = (color, color, color)
    img = Image.new("RGB", (1080, 2400), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _fake_crash_png() -> bytes:
    """Return an image with a red patch at the crash dismiss coordinate."""
    img = Image.new("RGB", (1080, 2400), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    x, y = CRASH_DISMISS_COORD
    size = 20
    draw.rectangle(
        [x - size // 2, y - size // 2, x + size // 2, y + size // 2], fill=(255, 0, 0)
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_reset_returns_observation():
    controller = Mock(spec=ADBController)
    controller.screencap.return_value = _fake_png(0)
    env = SubwaySurfersEnv(controller=controller, frame_size=(50, 50))
    obs, info = env.reset()
    assert obs.shape == (50, 50, 3)
    assert info == {}


@pytest.mark.parametrize(
    "action,coords",
    [
        (0, (540, 1600, 140, 1600)),
        (1, (540, 1600, 940, 1600)),
        (2, (540, 1600, 540, 900)),
        (3, (540, 1600, 540, 2000)),
    ],
)
def test_step_swipe_called(monkeypatch, action, coords):
    controller = Mock(spec=ADBController)
    controller.screencap.side_effect = [
        _fake_png(0),  # ensure_playing
        _fake_png(0),  # reset frame
        _fake_png(0),  # step start
        _fake_png(255),  # step result
    ]
    monkeypatch.setattr(time, "time", lambda: 0.0)
    env = SubwaySurfersEnv(controller=controller)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
    controller.swipe.assert_called_once_with(*coords)
    assert obs.shape == env.observation_space.shape
    # Non-crash steps should never return a negative reward.
    assert reward >= 0.0
    assert terminated is False
    assert truncated is False
    assert info == {"steps_survived": 1}


def test_is_crash_false_on_normal_frame():
    controller = Mock(spec=ADBController)
    env = SubwaySurfersEnv(controller=controller)
    image = Image.new("RGB", (1080, 2400), (0, 0, 0))
    assert env._is_crash(image) is False


def test_step_detects_crash_and_skips(monkeypatch):
    controller = Mock(spec=ADBController)
    controller.screencap.side_effect = [
        _fake_png(0),  # ensure_playing
        _fake_png(0),  # reset frame
        _fake_png(0),  # step start
        _fake_crash_png(),  # crash frame
    ]
    monkeypatch.setattr(time, "time", lambda: 0.0)
    env = SubwaySurfersEnv(controller=controller)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    controller.tap.assert_called_with(520, 1700)
    assert reward == -1.0
    assert terminated is True
    assert info == {
        "steps_survived": 1,
        "episode_reward": -1.0,
        "episode_length": 1,
    }


def test_template_matching_detects_menu(tmp_path):
    # Create a base image with a distinctive block
    base = np.zeros((50, 50, 3), dtype=np.uint8)
    base[10:20, 10:20] = (0, 255, 0)
    template = base[10:20, 10:20]

    tmpl_path = tmp_path / "menu.png"
    Image.fromarray(template).save(tmpl_path)

    controller = Mock(spec=ADBController)
    buf = io.BytesIO()
    Image.fromarray(base).save(buf, format="PNG")
    controller.screencap.return_value = buf.getvalue()

    env = SubwaySurfersEnv(controller=controller, menu_template_path=tmpl_path)
    image = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    assert env._is_menu(image) is True


def test_log_state_reports_menu(caplog):
    controller = Mock(spec=ADBController)
    env = SubwaySurfersEnv(controller=controller)
    image = Image.new("RGB", (1080, 2400), (0, 255, 0))
    with caplog.at_level(logging.DEBUG):
        env._log_state(image)
    assert "Game state: menu" in caplog.text


def test_reset_logs_game_started(monkeypatch, caplog):
    controller = Mock(spec=ADBController)
    controller.screencap.side_effect = [
        _fake_png((0, 255, 0)),  # menu frame triggers tap
        _fake_png(0),  # playing frame after tap
        _fake_png(0),  # reset frame
    ]
    env = SubwaySurfersEnv(controller=controller)
    with caplog.at_level(logging.INFO):
        env.reset()
    assert "game started" in caplog.text.lower()


def test_menu_retry_after_timeout(monkeypatch):
    controller = Mock(spec=ADBController)
    controller.screencap.return_value = _fake_png((0, 255, 0))
    env = SubwaySurfersEnv(controller=controller)
    env.reset()

    times = iter([0.0, 0.0, 1.1, 1.1])
    monkeypatch.setattr(time, "time", lambda: next(times))
    env.step(0)  # first menu detection, set _menu_since
    env.step(0)  # after timeout, should tap again
    controller.tap.assert_called_with(*PLAY_BUTTON_COORD)


def test_episode_end_logs_length_and_reward(caplog, monkeypatch):
    controller = Mock(spec=ADBController)
    controller.screencap.side_effect = [
        _fake_png(0),  # ensure_playing
        _fake_png(0),  # reset frame
        _fake_png(0),  # step start
        _fake_crash_png(),  # crash frame
    ]
    monkeypatch.setattr(time, "time", lambda: 0.0)
    env = SubwaySurfersEnv(controller=controller)
    env.reset()
    with caplog.at_level(logging.INFO):
        env.step(0)
    assert "game finished" in caplog.text.lower()


def test_no_extra_reward_or_log_after_crash(monkeypatch, caplog):
    """Ensure crashing doesn't grant additional rewards on repeated steps."""
    controller = Mock(spec=ADBController)
    controller.screencap.side_effect = [
        _fake_png(0),  # ensure_playing
        _fake_png(0),  # reset frame
        _fake_png(0),  # step start
        _fake_crash_png(),  # crash frame (terminates episode)
        _fake_crash_png(),  # second step while crash screen is visible
    ]
    monkeypatch.setattr(time, "time", lambda: 0.0)
    env = SubwaySurfersEnv(controller=controller)
    env.reset()
    with caplog.at_level(logging.INFO):
        env.step(0)  # first crash
        obs, reward, terminated, truncated, info = env.step(0)
    assert reward == 0.0
    assert terminated is True
    assert info == {}
    # Only one game finished log should be emitted
    assert caplog.text.lower().count("game finished") == 1
