# Subway Surfers AI

An open-source project to train a reinforcement-learning agent to play the original *Subway Surfers* game through an Android emulator. The project aims to be cross-platform (macOS and Windows) and uses a virtual Python environment for development.

## Getting Started

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\\Scripts\\activate
```

### 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Set up the Android emulator

Install [Android Studio](https://developer.android.com/studio) and configure a device with *Subway Surfers*. Make sure the `adb` command is available in your PATH.

### 4. Run formatting and linting

The project uses [pre-commit](https://pre-commit.com/) hooks for formatting and linting.

```bash
pre-commit install
pre-commit run --files $(git ls-files '*.py')
```

## Project Structure

```
├── src/
│   ├── agent/        # RL algorithms and models
│   ├── env/          # Emulator interface and wrappers
│   └── training/     # Training loops and utilities
├── configs/          # Hyper-parameter configs
├── scripts/          # Helper scripts (run training, play agent)
├── tests/            # Unit tests
└── docs/             # Additional documentation
```

## Notes

- Keep API keys or other secrets in a local `.env` file (ignored by git).
- The repository does **not** include any game assets.
- For sensitive operations (e.g., downloading the game APK), run commands locally and do not commit them.

