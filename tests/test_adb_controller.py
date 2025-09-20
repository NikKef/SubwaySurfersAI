"""Tests for the ADBController."""

from subprocess import TimeoutExpired
from unittest import mock

from src.env import ADBController


def test_tap_invokes_adb_with_coordinates() -> None:
    controller = ADBController(adb_path="adb", device_id="emulator-5554")
    with mock.patch("src.env.adb_controller.run") as run_mock:
        controller.tap(10, 20)
        run_mock.assert_called_once_with(
            [
                "adb",
                "-s",
                "emulator-5554",
                "shell",
                "input",
                "tap",
                "10",
                "20",
            ],
            check=True,
            timeout=controller.timeout,
        )


def test_screencap_returns_bytes() -> None:
    controller = ADBController()
    fake_result = mock.Mock(stdout=b"pngbytes")
    with mock.patch("src.env.adb_controller.run", return_value=fake_result) as run_mock:
        data = controller.screencap()
        run_mock.assert_called_once_with(
            ["adb", "exec-out", "screencap", "-p"],
            check=True,
            capture_output=True,
            timeout=controller.timeout,
    )
    assert data == b"pngbytes"


def test_timeout_is_retried_with_backoff() -> None:
    controller = ADBController(timeout=1.0, max_retries=1, retry_backoff=2.0)

    def run_side_effect(cmd, timeout, **kwargs):
        if timeout < 2.0:
            raise TimeoutExpired(cmd=cmd, timeout=timeout)
        return mock.Mock()

    with mock.patch("src.env.adb_controller.run", side_effect=run_side_effect) as run_mock:
        controller.tap(1, 2)

    assert run_mock.call_count == 2
    first_timeout = run_mock.call_args_list[0].kwargs["timeout"]
    second_timeout = run_mock.call_args_list[1].kwargs["timeout"]
    assert first_timeout == controller.timeout
    assert second_timeout == controller.timeout * controller.retry_backoff
