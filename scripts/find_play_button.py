"""Capture a screenshot and report click coordinates.

Run this script while the emulator is displaying the Subway Surfers menu.
Click on the green PLAY button; the script will print the pixel coordinates
so that they can be used for automated tapping.
"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Allow running as a script without installing the package
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env import ADBController  # noqa: E402


def main() -> None:
    controller = ADBController()
    raw = controller.screencap()

    image = Image.open(BytesIO(raw))
    array = np.array(image)

    fig, ax = plt.subplots()
    ax.imshow(array)
    ax.set_title("Click the PLAY button")
    ax.axis("off")

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        print(f"Clicked at ({int(event.xdata)}, {int(event.ydata)})")
        fig.canvas.mpl_disconnect(cid)
        plt.close(fig)

    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()


if __name__ == "__main__":
    main()
