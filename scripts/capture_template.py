"""Save a screenshot from the emulator to a local file.

Use this to capture reference templates of the Subway Surfers menu or crash
screen for template matching. Run while the emulator displays the desired
screen:

```
python scripts/capture_template.py templates/menu_full.png
```

The script writes a PNG file to the given path.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow running without installing the package
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env import ADBController  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture emulator screenshot")
    parser.add_argument("output", type=Path, help="File path for the PNG screenshot")
    args = parser.parse_args()

    controller = ADBController()
    png_bytes = controller.screencap()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(png_bytes)
    print(f"Saved screenshot to {args.output}")


if __name__ == "__main__":
    main()
