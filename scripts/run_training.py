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

    # ``yaml.safe_load`` treats values such as ``3e-4`` as strings under
    # the YAML 1.2 specification.  Cast numeric hyper-parameters explicitly so
    # that Stable-Baselines3 receives proper ``float``/``int`` types.
    learning_rate = float(cfg["learning_rate"])
    buffer_size = int(cfg["buffer_size"])
    gamma = float(cfg["gamma"])
    batch_size = int(cfg["batch_size"])
    train_steps = int(cfg["train_steps"])

    env = SubwaySurfersEnv()
    agent = DQNAgent(
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        gamma=gamma,
        batch_size=batch_size,
    )
    agent.train(train_steps)

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(args.model_path))
    env.close()


if __name__ == "__main__":
    main()
