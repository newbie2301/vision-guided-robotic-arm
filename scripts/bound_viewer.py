"""
bound_viewer.py
Vision-Guided Robotic Arm – Interactive Label Verification Tool

Displays training images with bounding box overlays for visual
quality control of annotations. Step through the dataset with
arrow keys / buttons.

Usage:
    python bound_viewer.py --images datasets/raw/images --labels datasets/raw/labels
    python bound_viewer.py --images datasets/raw/images --labels datasets/raw/labels --classes Metal-Object,Tray

Controls:
    → / D       next image
    ← / A       previous image
    Delete      flag current image+label for review (writes to flagged.txt)
    Esc / Q     quit
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

VALID_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# BGR colours per class index
CLASS_COLOURS = [
    (0, 200, 0),     # class 0 – green   (Metal-Object)
    (200, 120, 0),   # class 1 – blue    (Tray)
    (0, 0, 200),     # class 2 – red     (fallback / error class)
]


def load_labels(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    if not label_path.is_file():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                cx, cy, w, h = (float(p) for p in parts[1:])
                boxes.append((cls_id, cx, cy, w, h))
    return boxes


def draw_boxes(
    img: np.ndarray,
    boxes: list[tuple[int, float, float, float, float]],
    class_names: list[str],
) -> np.ndarray:
    h, w = img.shape[:2]
    out = img.copy()
    for cls_id, cx, cy, bw, bh in boxes:
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        colour = CLASS_COLOURS[cls_id % len(CLASS_COLOURS)]
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, 2)
        label = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def run_viewer(
    images_dir: str,
    labels_dir: str,
    class_names: list[str],
    start_index: int = 0,
) -> None:
    idir = Path(images_dir)
    ldir = Path(labels_dir)

    img_paths = sorted(p for p in idir.iterdir() if p.suffix.lower() in VALID_IMG_EXT)
    if not img_paths:
        logger.error(f"No images found in {idir}")
        return

    flagged_file = Path("flagged.txt")
    flagged: set[str] = set()
    if flagged_file.is_file():
        flagged = set(flagged_file.read_text().splitlines())

    idx = max(0, min(start_index, len(img_paths) - 1))
    window_name = "Label Viewer  [A/D: prev/next]  [Del: flag]  [Q/Esc: quit]"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1100, 750)

    while True:
        img_path = img_paths[idx]
        label_path = ldir / img_path.with_suffix(".txt").name

        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning(f"Cannot read image: {img_path}")
            idx = (idx + 1) % len(img_paths)
            continue

        boxes = load_labels(label_path)
        annotated = draw_boxes(img, boxes, class_names)

        flag_status = " [FLAGGED]" if img_path.name in flagged else ""
        header = f"[{idx + 1}/{len(img_paths)}] {img_path.name}  boxes={len(boxes)}{flag_status}"
        cv2.putText(annotated, header, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow(window_name, annotated)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), 27):           # Q or Esc
            break
        elif key in (ord('d'), 83):         # D or → arrow
            idx = (idx + 1) % len(img_paths)
        elif key in (ord('a'), 81):         # A or ← arrow
            idx = (idx - 1) % len(img_paths)
        elif key in (8, 255):               # Delete / Backspace
            flagged.add(img_path.name)
            flagged_file.write_text("\n".join(sorted(flagged)) + "\n")
            logger.info(f"Flagged: {img_path.name}")

    cv2.destroyAllWindows()
    if flagged:
        logger.info(f"{len(flagged)} image(s) flagged → {flagged_file}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive YOLO label viewer")
    parser.add_argument("--images",  required=True, help="Path to images/ directory")
    parser.add_argument("--labels",  required=True, help="Path to labels/ directory")
    parser.add_argument("--classes", default="Metal-Object,Tray",
                        help="Comma-separated class names in index order")
    parser.add_argument("--start",   type=int, default=0, help="Starting image index")
    args = parser.parse_args()

    class_names = [c.strip() for c in args.classes.split(",")]
    run_viewer(args.images, args.labels, class_names, args.start)
