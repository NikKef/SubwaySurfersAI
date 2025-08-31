# Architecture Overview

This document will track design decisions and architecture diagrams for the Subway Surfers AI project.

- **Environment**: Interfaces with Android emulator via ADB, captures frames, and sends actions.
  - `ADBController` provides a thin subprocess-based wrapper around the `adb` binary for
    cross-platform screen capture and input events.
  - `SubwaySurfersEnv` exposes a Gymnasium-compatible interface built on top of
    `ADBController` for training and evaluation. Observations are grayscale
    frames that can be stacked (``frame_stack``) to provide temporal context.
    The environment automatically taps the menu **PLAY** button and dismisses
    the *Save Me?* dialog using template matching with `templates/menu_full.png`
    and `templates/crash_full.png` (falling back to color checks if templates
    are missing). It logs the current game state every few seconds, retries
    tapping **PLAY** if the game stays on the menu for more than five seconds,
    and measures rewards based on time spent actually playing.
- **Agent**: Reinforcement learning algorithms (starting with DQN) implemented in PyTorch.
  - `DQNAgent` wraps the Stable-Baselines3 DQN implementation and provides a
    small API for training and action selection.
- **Training**: Scripts and utilities to optimize the agent.
  - `run_training.py` reads hyper-parameters from a YAML config, prints
    progress during training, writes TensorBoard logs, and periodically saves
    checkpoints so runs can be resumed seamlessly.
  - `find_play_button.py` captures a screenshot and reports click coordinates to
    help identify UI elements like the green PLAY button.
