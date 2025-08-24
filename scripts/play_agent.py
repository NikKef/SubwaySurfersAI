"""Run a trained agent inside the Android emulator."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.agent import DQNAgent
from src.env import SubwaySurfersEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play using a trained model")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/dqn_subway_agent"),
        help="Path to the trained model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    env = SubwaySurfersEnv()
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
