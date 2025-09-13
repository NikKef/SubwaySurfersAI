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
    ``templates/templates_crash_full.png`` exist, they are used for template matching
    to detect the menu and crash screens. The environment logs the detected game
    state every ``state_log_interval`` seconds.
    """

    controller: Optional[ADBController] = None
    frame_size: Tuple[int, int] = (160, 90)  # (width, height)
    action_coords: Dict[int, Tuple[int, int, int, int]] = field(
        default_factory=lambda: DEFAULT_ACTION_COORDS.copy()
    )
    menu_template_path: Optional[Path] = Path("templates/menu_full.png")
    crash_template_path: Optional[Path] = Path("templates/templates_crash_full.png")
    menu_template: Optional[np.ndarray] = field(init=False, default=None)
    crash_template: Optional[np.ndarray] = field(init=False, default=None)
    state_log_interval: float = 2.0
    _last_state_log: float = field(init=False, default_factory=lambda: 0.0)
    _episode_reward: float = field(init=False, default_factory=lambda: 0.0)
    _episode_length: int = field(init=False, default_factory=lambda: 0)
    _menu_since: Optional[float] = field(init=False, default=None)

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
        if self.crash_template is not None:
            return self._match_template(image, self.crash_template)
        try:
            np_img = np.array(image)
        except Exception:
            return False

        x, y = CRASH_DISMISS_COORD
        x0 = max(x - 5, 0)
        x1 = min(x + 5, np_img.shape[1])
        y0 = max(y - 5, 0)
        y1 = min(y + 5, np_img.shape[0])
        patch = np_img[y0:y1, x0:x1]
        if patch.size == 0:
            return False
        red_mask = (
            (patch[:, :, 0] > 200) & (patch[:, :, 1] < 100) & (patch[:, :, 2] < 100)
        )
        return float(red_mask.mean()) > 0.6

    def _log_state(self, image: Image.Image) -> None:
        now = time.time()
        if now - self._last_state_log < self.state_log_interval:
            return
        state = self._detect_state(image)
        LOGGER.debug("Game state: %s", state)
        self._last_state_log = now

    def _detect_state(self, image: Image.Image) -> str:
        if self._is_crash(image):
            return "crashed"
        if self._is_menu(image):
            return "menu"
        return "playing"

    def _ensure_playing(self) -> None:
        """Press buttons to ensure the game is in a running state."""
        while True:
            img = self._capture_raw()
            state = self._detect_state(img)
            self._log_state(img)
            if state == "crashed":
                self.controller.tap(*CRASH_DISMISS_COORD)
                time.sleep(1)
                continue
            if state == "menu":
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
        self._episode_reward = 0.0
        self._episode_length = 0
        self._menu_since = None
        self._log_state(image)
        observation = self._preprocess(image)
        return observation, {}

    def step(self, action: int):
        image = self._capture_raw()
        state = self._detect_state(image)
        now = time.time()
        self._log_state(image)

        # Handle non-playing states before executing the action.
        if state == "menu":
            if self._menu_since is None:
                self._menu_since = now
            elif now - self._menu_since > 5.0:
                self.controller.tap(*PLAY_BUTTON_COORD)
                self._menu_since = now
            observation = self._preprocess(image)
            return observation, 0.0, False, False, {}

        if state == "crashed":
            self.controller.tap(*CRASH_DISMISS_COORD)
            observation = self._preprocess(image)
            self._episode_reward += -1.0
            self._episode_length += 1
            info = {
                "steps_survived": self._episode_length,
                "episode_reward": self._episode_reward,
                "episode_length": self._episode_length,
            }
            LOGGER.info(
                "Game finished: length=%d, reward=%.2f",
                self._episode_length,
                self._episode_reward,
            )
            self._episode_reward = 0.0
            self._episode_length = 0
            self._menu_since = None
            # Penalize crashes with a negative reward.
            return observation, -1.0, True, False, info

        # Playing: execute action and compute time-based reward.
        if action not in self.action_coords:
            raise gym.error.InvalidAction(f"Invalid action: {action}")
        x1, y1, x2, y2 = self.action_coords[action]
        self.controller.swipe(x1, y1, x2, y2)

        image = self._capture_raw()
        state = self._detect_state(image)
        now2 = time.time()
        self._log_state(image)
        time_reward = 1.0

        terminated = False
        reward = time_reward
        if state == "crashed":
            self.controller.tap(*CRASH_DISMISS_COORD)
            terminated = True
            reward = -1.0
        if state == "menu":
            self._menu_since = now2
        else:
            self._menu_since = None

        self._episode_reward += reward
        self._episode_length += 1

        observation = self._preprocess(image)
        info: Dict[str, float] = {"steps_survived": self._episode_length}
        if terminated:
            LOGGER.info(
                "Game finished: length=%d, reward=%.2f",
                self._episode_length,
                self._episode_reward,
            )
            info = {
                "steps_survived": self._episode_length,
                "episode_reward": self._episode_reward,
                "episode_length": self._episode_length,
            }
            self._episode_reward = 0.0
            self._episode_length = 0
        return observation, reward, terminated, False, info

    def render(self) -> np.ndarray:
        return self._get_frame()

    def close(self) -> None:  # pragma: no cover - nothing to clean up
        return
