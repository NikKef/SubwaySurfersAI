from __future__ import annotations

import io
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image

from .adb_controller import ADBController

LOGGER = logging.getLogger(__name__)

# Default swipe coordinates for a 1080x2400 portrait emulator screen.
# Values are (x1, y1, x2, y2).
DEFAULT_ACTION_COORDS: Dict[int, Tuple[int, int, int, int]] = {
    0: (540, 1600, 140, 1600),  # left
    1: (540, 1600, 940, 1600),  # right
    2: (540, 1600, 540, 900),  # jump
    3: (540, 1600, 540, 2000),  # roll
}

# UI element coordinates for a 1080x2400 screen.
PLAY_BUTTON_COORD = (854, 2287)
CRASH_DISMISS_COORD = (520, 1700)


@dataclass
class SubwaySurfersEnv(gym.Env[np.ndarray, int]):
    """Gymnasium-compatible environment for Subway Surfers.

    The environment communicates with an Android emulator via ``ADBController``.
    Observations are RGB frames resized to ``frame_size``. Actions are discrete
    swipes: left, right, jump, roll. If ``templates/menu_full.png`` and
    ``templates/crash_full.png`` exist, they are used for template matching to
    detect the menu and crash screens. The environment logs the detected game
    state every ``state_log_interval`` seconds.
    """

    controller: Optional[ADBController] = None
    frame_size: Tuple[int, int] = (160, 90)  # (width, height)
    action_coords: Dict[int, Tuple[int, int, int, int]] = field(
        default_factory=lambda: DEFAULT_ACTION_COORDS.copy()
    )
    menu_template_path: Optional[Path] = Path("templates/menu_full.png")
    crash_template_path: Optional[Path] = Path("templates/crash_full.png")
    menu_template: Optional[np.ndarray] = field(init=False, default=None)
    crash_template: Optional[np.ndarray] = field(init=False, default=None)
    state_log_interval: float = 2.0
    _last_state_log: float = field(init=False, default_factory=lambda: 0.0)

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

        if self.menu_template_path and Path(self.menu_template_path).exists():
            self.menu_template = cv2.imread(
                str(self.menu_template_path), cv2.IMREAD_GRAYSCALE
            )
        if self.crash_template_path and Path(self.crash_template_path).exists():
            self.crash_template = cv2.imread(
                str(self.crash_template_path), cv2.IMREAD_GRAYSCALE
            )

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _capture_raw(self) -> Image.Image:
        """Return the current emulator frame as a PIL image."""
        png_bytes = self.controller.screencap()
        return Image.open(io.BytesIO(png_bytes)).convert("RGB")

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.resize(self.frame_size, Image.BILINEAR)
        return np.asarray(image, dtype=np.uint8)

    def _match_template(
        self, image: Image.Image, template: np.ndarray, threshold: float = 0.9
    ) -> bool:
        img_gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        method = cv2.TM_CCOEFF_NORMED
        if float(template.std()) == 0.0:
            res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF_NORMED)
            return float(res.min()) <= (1 - threshold)
        res = cv2.matchTemplate(img_gray, template, method)
        return float(res.max()) >= threshold

    def _is_menu(self, image: Image.Image) -> bool:
        if self.menu_template is not None and self._match_template(
            image, self.menu_template
        ):
            return True
        try:
            r, g, b = image.getpixel(PLAY_BUTTON_COORD)
        except IndexError:
            return False
        return g > 150 and r < 100 and b < 100

    def _is_crash(self, image: Image.Image) -> bool:
        if self.crash_template is not None and self._match_template(
            image, self.crash_template
        ):
            return True
        try:
            r, g, b = image.getpixel(CRASH_DISMISS_COORD)
        except IndexError:
            return False
        return r > 200 and g < 100 and b < 100

    def _log_state(self, image: Image.Image) -> None:
        now = time.time()
        if now - self._last_state_log < self.state_log_interval:
            return
        state = "playing"
        if self._is_crash(image):
            state = "crashed"
        elif self._is_menu(image):
            state = "menu"
        LOGGER.info("Game state: %s", state)
        self._last_state_log = now

    def _ensure_playing(self) -> None:
        """Press buttons to ensure the game is in a running state."""
        while True:
            img = self._capture_raw()
            self._log_state(img)
            if self._is_crash(img):
                self.controller.tap(*CRASH_DISMISS_COORD)
                time.sleep(1)
                continue
            if self._is_menu(img):
                self.controller.tap(*PLAY_BUTTON_COORD)
                time.sleep(1)
                continue
            break

    def _get_frame(self) -> np.ndarray:
        image = self._capture_raw()
        return self._preprocess(image)

    # Gymnasium API ----------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._ensure_playing()
        image = self._capture_raw()
        self._log_state(image)
        observation = self._preprocess(image)
        return observation, {}

    def step(self, action: int):
        if action not in self.action_coords:
            raise gym.error.InvalidAction(f"Invalid action: {action}")
        x1, y1, x2, y2 = self.action_coords[action]
        self.controller.swipe(x1, y1, x2, y2)
        image = self._capture_raw()
        observation = self._preprocess(image)
        self._log_state(image)
        terminated = False
        if self._is_crash(image):
            self.controller.tap(*CRASH_DISMISS_COORD)
            terminated = True
        reward = 0.0
        truncated = False
        info: Dict[str, float] = {}
        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        return self._get_frame()

    def close(self) -> None:  # pragma: no cover - nothing to clean up
        return
