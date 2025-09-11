from __future__ import annotations

import io
import re
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
import pytesseract

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
    _menu_since: Optional[float] = field(init=False, default=None)
    _last_run_score: int = field(init=False, default=0)
    _last_run_coins: int = field(init=False, default=0)
    _pending_reward: float = field(init=False, default=0.0)
    score_weight: float = 1.0
    coin_weight: float = 1.0

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

    def _extract_score_from_menu(self, image: Image.Image) -> Optional[int]:
        """Extract the score shown on the menu screen."""
        w, h = image.size
        # Crop the area containing the "Score" widget (upper-middle right).
        region = image.crop(
            (
                int(w * 0.55),
                int(h * 0.15),
                int(w * 0.95),
                int(h * 0.35),
            )
        )
        gray = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2GRAY)
        text = pytesseract.image_to_string(
            gray, config="--psm 7 -c tessedit_char_whitelist=0123456789"
        )
        match = re.search(r"\d+", text)
        if match:
            return int(match.group())
        return None

    def _extract_coin_count(self, image: Image.Image) -> Optional[int]:
        """Extract the coin total shown on the menu screen."""
        w, h = image.size
        region = image.crop(
            (
                int(w * 0.55),
                int(h * 0.35),
                int(w * 0.95),
                int(h * 0.55),
            )
        )
        gray = cv2.cvtColor(np.array(region), cv2.COLOR_RGB2GRAY)
        text = pytesseract.image_to_string(
            gray, config="--psm 7 -c tessedit_char_whitelist=0123456789"
        )
        match = re.search(r"\d+", text)
        if match:
            return int(match.group())
        return None

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
        state = self._detect_state(image)
        LOGGER.info("Game state: %s", state)
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
                try:
                    menu_img = self._capture_raw()
                    score = self._extract_score_from_menu(menu_img)
                    coins = self._extract_coin_count(menu_img)
                    if score is not None:
                        self._last_run_score = score
                    if coins is not None:
                        self._last_run_coins = coins
                except Exception:
                    pass
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
        self._menu_since = None
        self._log_state(image)
        observation = self._preprocess(image)
        return observation, {}

    def step(self, action: int):
        image = self._capture_raw()
        state = self._detect_state(image)
        now = time.time()
        self._log_state(image)

        reward = 0.0
        if state == "playing" and self._pending_reward:
            reward = self._pending_reward
            self._pending_reward = 0.0

        # Handle non-playing states before executing the action.
        if state == "menu":
            if self._menu_since is None:
                self._menu_since = now
            elif now - self._menu_since > 5.0:
                self.controller.tap(*PLAY_BUTTON_COORD)
                self._menu_since = now
            observation = self._preprocess(image)
            return observation, 0.0, False, False, {"time_survived": 0.0}

        if state == "crashed":
            self.controller.tap(*CRASH_DISMISS_COORD)
            time.sleep(1)
            try:
                menu_img = self._capture_raw()
            except Exception:
                menu_img = image
            score = self._extract_score_from_menu(menu_img)
            coins = self._extract_coin_count(menu_img)
            score_delta = 0
            coin_delta = 0
            if score is not None:
                score_delta = score - self._last_run_score
                self._last_run_score = score
            if coins is not None:
                coin_delta = coins - self._last_run_coins
                self._last_run_coins = coins
            self._pending_reward = (
                self.score_weight * score_delta + self.coin_weight * coin_delta
            )
            LOGGER.info("Run finished: score=%s, coins=%s", score, coins)
            observation = self._preprocess(menu_img)
            self._menu_since = now
            return observation, -1.0, True, False, {"time_survived": 0.0}

        # Playing: execute action with no intermediate reward.
        if action not in self.action_coords:
            raise gym.error.InvalidAction(f"Invalid action: {action}")
        x1, y1, x2, y2 = self.action_coords[action]
        self.controller.swipe(x1, y1, x2, y2)

        image = self._capture_raw()
        state = self._detect_state(image)
        now2 = time.time()
        self._log_state(image)

        if state == "crashed":
            self.controller.tap(*CRASH_DISMISS_COORD)
            time.sleep(1)
            try:
                menu_img = self._capture_raw()
            except Exception:
                menu_img = image
            score = self._extract_score_from_menu(menu_img)
            coins = self._extract_coin_count(menu_img)
            score_delta = 0
            coin_delta = 0
            if score is not None:
                score_delta = score - self._last_run_score
                self._last_run_score = score
            if coins is not None:
                coin_delta = coins - self._last_run_coins
                self._last_run_coins = coins
            self._pending_reward = (
                self.score_weight * score_delta + self.coin_weight * coin_delta
            )
            LOGGER.info("Run finished: score=%s, coins=%s", score, coins)
            observation = self._preprocess(menu_img)
            self._menu_since = now2
            return observation, -1.0, True, False, {"time_survived": 0.0}
        if state == "menu":
            self._menu_since = now2
        else:
            self._menu_since = None

        observation = self._preprocess(image)
        return observation, reward, False, False, {"time_survived": 0.0}

    def render(self) -> np.ndarray:
        return self._get_frame()

    def close(self) -> None:  # pragma: no cover - nothing to clean up
        return
