from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image

from .adb_controller import ADBController

# Default swipe coordinates for a 1080x2400 portrait emulator screen.
# Values are (x1, y1, x2, y2).
DEFAULT_ACTION_COORDS: Dict[int, Tuple[int, int, int, int]] = {
    0: (540, 1600, 140, 1600),  # left
    1: (540, 1600, 940, 1600),  # right
    2: (540, 1600, 540, 900),  # jump
    3: (540, 1600, 540, 2000),  # roll
}


@dataclass
class SubwaySurfersEnv(gym.Env[np.ndarray, int]):
    """Gymnasium-compatible environment for Subway Surfers.

    The environment communicates with an Android emulator via ``ADBController``.
    Observations are RGB frames resized to ``frame_size``.  Actions are discrete
    swipes: left, right, jump, roll.
    """

    controller: Optional[ADBController] = None
    frame_size: Tuple[int, int] = (160, 90)  # (width, height)
    action_coords: Dict[int, Tuple[int, int, int, int]] = field(
        default_factory=lambda: DEFAULT_ACTION_COORDS.copy()
    )

    metadata = {"render_modes": ["rgb_array"]}

    def __post_init__(self) -> None:
        self.controller = self.controller or ADBController()
        width, height = self.frame_size
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(height, width, 3),
            dtype=np.uint8,
        )
        self.action_space = spaces.Discrete(len(self.action_coords))

    # ------------------------------------------------------------------
    def _get_frame(self) -> np.ndarray:
        """Capture and preprocess the current emulator frame."""
        png_bytes = self.controller.screencap()
        image = Image.open(io.BytesIO(png_bytes)).convert("RGB")
        image = image.resize(self.frame_size, Image.BILINEAR)
        return np.asarray(image, dtype=np.uint8)

    # Gymnasium API ----------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        observation = self._get_frame()
        return observation, {}

    def step(self, action: int):
        if action not in self.action_coords:
            raise gym.error.InvalidAction(f"Invalid action: {action}")
        x1, y1, x2, y2 = self.action_coords[action]
        self.controller.swipe(x1, y1, x2, y2)
        observation = self._get_frame()
        reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, float] = {}
        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        return self._get_frame()

    def close(self) -> None:  # pragma: no cover - nothing to clean up
        return
