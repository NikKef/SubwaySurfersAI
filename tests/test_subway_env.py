from __future__ import annotations

import io
from unittest.mock import Mock

import pytest
from PIL import Image

from src.env import SubwaySurfersEnv, ADBController


def _fake_png(color: int) -> bytes:
    img = Image.new("RGB", (100, 100), (color, color, color))
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
