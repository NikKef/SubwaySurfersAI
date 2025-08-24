# Architecture Overview

This document will track design decisions and architecture diagrams for the Subway Surfers AI project.

- **Environment**: Interfaces with Android emulator via ADB, captures frames, and sends actions.
  - `ADBController` provides a thin subprocess-based wrapper around the `adb` binary for
    cross-platform screen capture and input events.
  - `SubwaySurfersEnv` exposes a Gymnasium-compatible interface built on top of
    `ADBController` for training and evaluation.
- **Agent**: Reinforcement learning algorithms (starting with DQN) implemented in PyTorch.
- **Training**: Scripts and utilities to optimize the agent.
