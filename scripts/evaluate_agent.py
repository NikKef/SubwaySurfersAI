"""Evaluate a trained DQN agent on the Subway Surfers environment."""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
from pathlib import Path
from typing import List

from torch.utils.tensorboard import SummaryWriter

# Allow running as a script without installing the package
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import DQNAgent  # noqa: E402
from src.env import SubwaySurfersEnv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained agent")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/dqn_subway_agent"),
        help="Path to the trained model",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_file = args.model_path
    if not model_file.suffix:
        model_file = model_file.with_suffix(".zip")

    log_dir = model_file.parent
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "eval.log"
    logging.basicConfig(
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file, mode="a")],
    )

    tb_dir = log_dir / "tb_eval"
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))

    env = SubwaySurfersEnv()
    agent = DQNAgent.load(str(model_file), env)

    rewards: List[float] = []
    times: List[float] = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        terminated = False
        total_reward = 0.0
        time_survived = 0.0
        while not terminated:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            terminated = terminated or truncated
            total_reward += reward
            time_survived = float(info.get("time_survived", time_survived))
        rewards.append(total_reward)
        times.append(time_survived)
        writer.add_scalar("eval/episode_reward", total_reward, ep)
        writer.add_scalar("eval/time_survived", time_survived, ep)
        logging.info(
            "Episode %d: reward=%.2f, time_survived=%.2f",
            ep + 1,
            total_reward,
            time_survived,
        )

    avg_reward = statistics.mean(rewards)
    avg_time = statistics.mean(times)
    summary = (
        f"Evaluated {args.episodes} episodes — "
        f"reward: mean={avg_reward:.2f}, min={min(rewards):.2f}, max={max(rewards):.2f}; "
        f"time_survived: mean={avg_time:.2f}, min={min(times):.2f}, max={max(times):.2f}"
    )
    logging.info(summary)
    writer.add_text("eval/summary", summary)

    writer.close()
    env.close()


if __name__ == "__main__":
    main()
