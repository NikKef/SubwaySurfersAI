"""ADB-based interface for interacting with the Android emulator.

This module provides a thin wrapper around the ``adb`` command-line tool to
capture emulator frames and send tap or swipe inputs.  It relies only on the
standard library and therefore works on both macOS and Windows, provided the
``adb`` binary is available in the ``PATH``.
"""

from __future__ import annotations

from dataclasses import dataclass
from subprocess import run, CompletedProcess, TimeoutExpired
from time import sleep
from typing import List, Optional


@dataclass
class ADBController:
    """Minimal wrapper for issuing commands to an Android emulator.

    Parameters
    ----------
    adb_path:
        Path to the ``adb`` executable.  Defaults to ``"adb"`` and therefore
        expects ``adb`` to be discoverable in ``PATH``.
    device_id:
        Optional serial of the target emulator/device.  If omitted the first
        available device will be used.
    timeout:
        Maximum number of seconds to wait for a single ``adb`` invocation.
    max_retries:
        Number of times to retry a command that times out.  Retries reuse the
        previous command and progressively increase the timeout.
    retry_backoff:
        Multiplicative factor applied to the timeout after each failed attempt.
    """

    adb_path: str = "adb"
    device_id: Optional[str] = None
    timeout: float = 5.0
    max_retries: int = 2
    retry_backoff: float = 2.0

    def _prefix(self) -> List[str]:
        """Build the base adb command list."""
        cmd = [self.adb_path]
        if self.device_id:
            cmd.extend(["-s", self.device_id])
        return cmd

    def _run(self, cmd: List[str], **kwargs) -> CompletedProcess:
        """Execute an adb command with a built-in timeout.

        A timeout prevents the training loop from hanging indefinitely if no
        emulator is connected or ``adb`` becomes unresponsive.
        """

        timeout = self.timeout
        attempt = 0
        while True:
            try:
                return run(cmd, timeout=timeout, **kwargs)
            except TimeoutExpired as exc:  # pragma: no cover - edge case
                attempt += 1
                if attempt > self.max_retries:
                    raise RuntimeError(
                        "ADB command timed out; ensure an emulator is running"
                    ) from exc
                timeout *= self.retry_backoff
                sleep(0.5)

    def screencap(self) -> bytes:
        """Return a PNG screenshot taken from the emulator.

        This method executes ``adb exec-out screencap -p`` and returns the raw
        PNG bytes.  The caller is responsible for decoding the image (e.g., via
        ``Pillow`` or ``cv2``).
        """

        cmd = self._prefix() + ["exec-out", "screencap", "-p"]
        result: CompletedProcess[bytes] = self._run(
            cmd, check=True, capture_output=True
        )
        return result.stdout

    def tap(self, x: int, y: int) -> None:
        """Send a tap at the given coordinates."""
        cmd = self._prefix() + ["shell", "input", "tap", str(x), str(y)]
        self._run(cmd, check=True)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 200) -> None:
        """Swipe from (x1, y1) to (x2, y2) over ``duration_ms`` milliseconds."""
        cmd = self._prefix() + [
            "shell",
            "input",
            "swipe",
            str(x1),
            str(y1),
            str(x2),
            str(y2),
            str(duration_ms),
        ]
        self._run(cmd, check=True)
