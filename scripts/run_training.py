"""Train a DQN agent on the Subway Surfers environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import logging

import yaml
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

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
    logging.basicConfig(level=logging.INFO)
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
    hidden_sizes = cfg.get("hidden_sizes", [256, 256])
    dueling = bool(cfg.get("dueling", True))
    double_q = bool(cfg.get("double_q", True))

    # Resolve model file (Stable-Baselines appends ``.zip`` if missing).
    model_file = args.model_path
    if not model_file.suffix:
        model_file = model_file.with_suffix(".zip")

    env = SubwaySurfersEnv()

    log_dir = model_file.parent / "tb"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = configure(str(log_dir), ["tensorboard", "stdout"])

    # Load existing model or checkpoint if available.
    if model_file.exists():
        agent = DQNAgent.load(str(model_file), env)
        agent.model.set_logger(logger)
        replay_path = model_file.with_name(f"{model_file.stem}_replay_buffer.pkl")
        if replay_path.exists():
            agent.model.load_replay_buffer(str(replay_path))
            print(f"Loaded replay buffer from {replay_path}")
        print(f"Loaded existing model from {model_file}")
    else:
        latest = find_latest_checkpoint(model_file)
        if latest is not None:
            agent = DQNAgent.load(str(latest), env)
            agent.model.set_logger(logger)
            replay_path = latest.with_name(f"{latest.stem}_replay_buffer.pkl")
            if replay_path.exists():
                agent.model.load_replay_buffer(str(replay_path))
                print(f"Loaded replay buffer from {replay_path}")
            print(f"Loaded checkpoint {latest}")
        else:
            agent = DQNAgent(
                env,
                learning_rate=learning_rate,
                buffer_size=buffer_size,
                gamma=gamma,
                batch_size=batch_size,
                verbose=1,
                tensorboard_log=str(log_dir),
                hidden_sizes=hidden_sizes,
                dueling=dueling,
                double_q=double_q,
            )
            print("Initialized new agent")

    # Setup checkpointing
    checkpoint_dir = model_file.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=int(cfg.get("checkpoint_freq", 10000)),
        save_path=str(checkpoint_dir),
        name_prefix=model_file.stem,
        save_replay_buffer=True,
    )

    steps_done = agent.model.num_timesteps
    steps_remaining = max(train_steps - steps_done, 0)
    print(
        f"Training for {steps_remaining} additional timesteps "
        f"(already trained: {steps_done})"
    )

    if steps_remaining > 0:
        agent.train(steps_remaining, callback=checkpoint_callback)
        model_file.parent.mkdir(parents=True, exist_ok=True)
        agent.save(str(model_file))
        replay_path = model_file.with_name(f"{model_file.stem}_replay_buffer.pkl")
        agent.model.save_replay_buffer(str(replay_path))
        print(f"Saved model to {model_file}")
        print(f"Saved replay buffer to {replay_path}")
    else:
        print("Target timesteps already reached; skipping training")

    env.close()


if __name__ == "__main__":
    main()
