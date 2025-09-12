"""Train a DQN agent on the Subway Surfers environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import logging

import yaml
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Allow running as a script without installing the package
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import DQNAgent  # noqa: E402
from src.env import SubwaySurfersEnv  # noqa: E402
from src.training.utils import (
    update_dqn_hyperparameters,
    load_or_create_dqn_agent,
)  # noqa: E402


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
    exploration_fraction = float(cfg.get("exploration_fraction", 0.1))
    exploration_final_eps = float(cfg.get("exploration_final_eps", 0.05))

    # Resolve model file (Stable-Baselines appends ``.zip`` if missing).
    model_file = args.model_path
    if not model_file.suffix:
        model_file = model_file.with_suffix(".zip")

    # Stack consecutive frames to give the agent a sense of motion.
    base_env = DummyVecEnv([SubwaySurfersEnv])
    env = VecFrameStack(base_env, n_stack=4)

    log_dir = model_file.parent / "tb"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = configure(str(log_dir), ["tensorboard", "stdout"])

    agent_kwargs = dict(
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        gamma=gamma,
        batch_size=batch_size,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        verbose=1,
        tensorboard_log=str(log_dir),
    )

    # Load existing model or checkpoint if available, falling back to a new
    # agent when loading fails (e.g. due to mismatched observation spaces).
    if model_file.exists():
        agent, loaded = load_or_create_dqn_agent(model_file, env, **agent_kwargs)
        agent.model.set_logger(logger)
        if loaded:
            print(f"Loaded existing model from {model_file}")
        else:
            print(
                f"Existing model at {model_file} could not be loaded; "
                "initialized new agent"
            )
    else:
        latest = find_latest_checkpoint(model_file)
        if latest is not None:
            agent, loaded = load_or_create_dqn_agent(latest, env, **agent_kwargs)
            agent.model.set_logger(logger)
            if loaded:
                print(f"Loaded checkpoint {latest}")
            else:
                print(
                    f"Checkpoint {latest} could not be loaded; initialized new agent"
                )
        else:
            agent = DQNAgent(env, **agent_kwargs)
            agent.model.set_logger(logger)
            print("Initialized new agent")

    # Apply potentially updated hyper-parameters when resuming training
    update_dqn_hyperparameters(
        agent.model,
        learning_rate=learning_rate,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
    )

    # Setup checkpointing
    checkpoint_dir = model_file.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=int(cfg.get("checkpoint_freq", 10000)),
        save_path=str(checkpoint_dir),
        name_prefix=model_file.stem,
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
        print(f"Saved model to {model_file}")
    else:
        print("Target timesteps already reached; skipping training")

    env.close()


if __name__ == "__main__":
    main()
