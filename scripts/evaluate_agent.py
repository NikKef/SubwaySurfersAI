"""Evaluate a trained agent inside the Android emulator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import logging

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Allow running as a script without installing the package
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import DQNAgent  # noqa: E402
from src.env import SubwaySurfersEnv  # noqa: E402


def find_latest_checkpoint(model_file: Path) -> Path | None:
    """Return path to the most recent checkpoint if present."""

    checkpoint_dir = model_file.parent / "checkpoints"
    pattern = f"{model_file.stem}_*.zip"
    candidates = sorted(checkpoint_dir.glob(pattern))
    return candidates[-1] if candidates else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to the trained model or checkpoint (default: latest checkpoint)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to evaluate",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    base_model = Path("models/dqn_subway_agent")
    model_path = args.model_path
    if model_path is None:
        latest = find_latest_checkpoint(base_model)
        model_path = latest if latest is not None else base_model
    if not model_path.suffix:
        model_path = model_path.with_suffix(".zip")

    base_env = DummyVecEnv([SubwaySurfersEnv])
    env = VecFrameStack(base_env, n_stack=4)
    agent = DQNAgent.load(str(model_path), env)

    mean_reward, std_reward = evaluate_policy(
        agent.model, env, n_eval_episodes=args.episodes
    )
    print(
        f"Mean reward over {args.episodes} episodes: {mean_reward:.2f} ± {std_reward:.2f}"
    )

    log_dir = model_path.parent / "tb"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = configure(str(log_dir), ["tensorboard"])
    logger.record("eval/mean_reward", float(mean_reward))
    logger.record("eval/std_reward", float(std_reward))
    logger.dump(0)

    env.close()

    latest_checkpoint = find_latest_checkpoint(base_model)
    if latest_checkpoint is not None:
        print(
            "To evaluate the latest checkpoint run:\n"
            f"python scripts/evaluate_agent.py --model-path {latest_checkpoint}"
        )


if __name__ == "__main__":
    main()
