"""Run a trained agent inside the Android emulator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import logging

# Allow running as a script without installing the package
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import DQNAgent  # noqa: E402
from src.env import SubwaySurfersEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play using a trained model")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/dqn_subway_agent"),
        help="Path to the trained model",
    )
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=4,
        help="Number of stacked grayscale frames expected by the model",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    env = SubwaySurfersEnv(frame_stack=args.frame_stack)
    agent = DQNAgent.load(str(args.model_path), env)

    obs, _ = env.reset()
    terminated = False
    while not terminated:
        action = agent.act(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        terminated = terminated or truncated

    env.close()


if __name__ == "__main__":
    main()
