"""Train a DQN agent on the Subway Surfers environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Allow running as a script without installing the package
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import DQNAgent  # noqa: E402
from src.env import SubwaySurfersEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Subway Surfers agent")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to hyper-parameter configuration file",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/dqn_subway_agent"),
        help="Where to store the trained model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.config.open() as fh:
        cfg = yaml.safe_load(fh)

    env = SubwaySurfersEnv()
    agent = DQNAgent(
        env,
        learning_rate=cfg["learning_rate"],
        buffer_size=cfg["buffer_size"],
        gamma=cfg["gamma"],
        batch_size=cfg["batch_size"],
    )
    agent.train(cfg["train_steps"])

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(args.model_path))
    env.close()


if __name__ == "__main__":
    main()
