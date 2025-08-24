from __future__ import annotations

import io
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image
import logging

from src.env import SubwaySurfersEnv, ADBController


def _fake_png(color) -> bytes:
    if isinstance(color, int):
        color = (color, color, color)
    img = Image.new("RGB", (1080, 2400), color)
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
def test_step_swipe_called(action, coords):
    controller = Mock(spec=ADBController)
    controller.screencap.return_value = _fake_png(0)
    env = SubwaySurfersEnv(controller=controller)
    env.reset()
    controller.screencap.return_value = _fake_png(255)
    obs, reward, terminated, truncated, info = env.step(action)
    controller.swipe.assert_called_once_with(*coords)
    assert obs.shape == env.observation_space.shape
    assert reward == 0.0
    assert terminated is False
    assert truncated is False
    assert info == {}


def test_step_detects_crash_and_skips():
    controller = Mock(spec=ADBController)
    controller.screencap.side_effect = [
        _fake_png(0),  # ensure_playing
        _fake_png(0),  # reset frame
        _fake_png((255, 0, 0)),  # crash frame
    ]
    env = SubwaySurfersEnv(controller=controller)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    controller.tap.assert_called_once_with(520, 1700)
    assert terminated is True


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
    with caplog.at_level(logging.INFO):
        env._log_state(image)
    assert "Game state: menu" in caplog.text
