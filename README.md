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

### 4. Interact with the emulator

The `ADBController` class offers a minimal wrapper around the `adb` binary for screen capture and input events. Example:

```python
from src.env import ADBController

ctrl = ADBController()
image_bytes = ctrl.screencap()  # PNG bytes of the current screen
ctrl.tap(100, 200)              # tap at x=100, y=200
```

### 5. Run formatting and linting

The project uses [pre-commit](https://pre-commit.com/) hooks for formatting and linting.

```bash
pre-commit install
pre-commit run --files $(git ls-files '*.py')
```

### 6. Train the baseline agent

With the emulator running and *Subway Surfers* open, launch training. The
default configuration stacks four consecutive **grayscale** frames to provide
temporal context for the agent:

```bash
python scripts/run_training.py --config configs/default.yaml --model-path models/dqn_subway_agent
```

The script prints training progress and automatically saves checkpoints under
`models/checkpoints/`. If a model or checkpoint already exists, training
resumes from the latest state and continues with the correct learning rate
schedules. If the observation shape has changed (e.g. after enabling frame
stacking), the existing checkpoint will be ignored and training starts with a
fresh model. Rewards reflect **time survived while the game is actually
playing**, with a small negative penalty applied when a crash occurs. Time spent
in menus or on crash screens does not contribute to the reward or episode
length.

To visualize learning curves, launch TensorBoard in another terminal:

```bash
tensorboard --logdir models/tb
```

### 7. Play using a trained model

After training, watch the agent play:

```bash
python scripts/play_agent.py --model-path models/dqn_subway_agent
```

### 8. Evaluate a trained model

Run a fixed number of episodes and log reward and survival time:

```bash
python scripts/evaluate_agent.py --model-path models/dqn_subway_agent --episodes 10
```

Evaluation metrics are written to `models/tb_eval` and can be viewed with:

```bash
tensorboard --logdir models/tb_eval
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

## Helper utilities

- `scripts/find_play_button.py` — capture the emulator screen and print the
  coordinates of the pixel you click. By default the environment presses the
  **PLAY** button at `(854, 2287)` and dismisses the *Save Me?* dialog at
  `(520, 1700)`. Use this script to adjust these coordinates for your device.
- `scripts/capture_template.py` — save the current emulator screen to a PNG
  file. Capture reference images of the menu and crash screens and store them
  as `templates/menu_full.png` and `templates/crash_full.png`; the environment will use
  template matching to detect when it should tap the PLAY button or dismiss
  the *Save Me?* dialog.

During play and training, the environment logs the current game state (menu,
playing, or crashed) to the terminal every couple of seconds, which helps
diagnose whether template matching is working as expected.
If the agent remains on the menu for more than five seconds, it automatically
attempts to tap the **PLAY** button again.

## Notes

- Keep API keys or other secrets in a local `.env` file (ignored by git).
- The repository does **not** include any game assets.
- For sensitive operations (e.g., downloading the game APK), run commands locally and do not commit them.
