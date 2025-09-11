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
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


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
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    base_env = DummyVecEnv([SubwaySurfersEnv])
    env = VecFrameStack(base_env, n_stack=4)
    agent = DQNAgent.load(str(args.model_path), env)

    obs = env.reset()
    done = False
    while not done:
        action = agent.act(obs)
        obs, _, done, _ = env.step([action])
        done = bool(done[0])

    env.close()


if __name__ == "__main__":
    main()
