"""Tests for the ADBController."""

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
        )


def test_screencap_returns_bytes() -> None:
    controller = ADBController()
    fake_result = mock.Mock(stdout=b"pngbytes")
    with mock.patch("src.env.adb_controller.run", return_value=fake_result) as run_mock:
        data = controller.screencap()
    run_mock.assert_called_once()
    assert data == b"pngbytes"
