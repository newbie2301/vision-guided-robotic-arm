"""
main.py
Vision-Guided Robotic Arm – Application Entry Point

Launches the Tkinter UI which wires together VisionThread, CoordMapper,
Coordinator, and ArduinoComm.

Usage:
    python main.py
"""

from __future__ import annotations

import logging
import sys
import tkinter as tk
from pathlib import Path

# Allow running as a script (python main.py) or as a module (python -m src.main)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ui import ArmUI


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    root = tk.Tk()
    root.geometry("1300x800")
    app = ArmUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
