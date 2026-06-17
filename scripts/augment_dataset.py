"""
augment_dataset.py
Vision-Guided Robotic Arm – Dataset Augmentation Pipeline

Generates AUGMENT_PER_IMAGE synthetic variants per source image using
random combinations of:
    - Horizontal / vertical flip
    - Random rotation (±30°) with bounding box coordinate transform
    - Brightness and contrast jitter
    - Gaussian blur
    - Random crop-and-resize (preserving labels)
    - Mosaic: 4-image composite

Usage:
    python augment_dataset.py --src datasets/raw --dst datasets/augmented
    python augment_dataset.py --src datasets/raw --dst datasets/augmented --n 20

YOLO label format:  <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
"""

from __future__ import annotations

import argparse
import logging
import math
import multiprocessing as mp
import os
import random
import shutil
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ── Configuration ─────────────────────────────────────────────────────────────
AUGMENT_PER_IMAGE = 20
IMG_SIZE          = 640
NUM_CLASSES       = 2        # 0 = Metal-Object,  1 = Tray
VALID_EXTENSIONS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ── Random transform ranges ───────────────────────────────────────────────────
ROT_MAX_DEG    = 30.0
BRIGHT_LOW     = 0.5
BRIGHT_HIGH    = 1.5
BLUR_KERNELS   = (3, 5)


# ─────────────────────────────────────────────────────────────────────────────
# Label helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_labels(label_path: Path) -> list[list[float]]:
    """Return list of [class_id, cx, cy, w, h] (normalised 0–1)."""
    boxes = []
    if not label_path.is_file():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                boxes.append([float(p) for p in parts])
            elif len(parts) > 5:
                # Polygon format → convert to bounding box
                cls  = int(parts[0])
                coords = [float(p) for p in parts[1:]]
                xs = coords[0::2]; ys = coords[1::2]
                cx = (min(xs) + max(xs)) / 2
                cy = (min(ys) + max(ys)) / 2
                w  = max(xs) - min(xs)
                h  = max(ys) - min(ys)
                boxes.append([float(cls), cx, cy, w, h])
    return boxes


def write_labels(label_path: Path, boxes: list[list[float]]) -> None:
    with open(label_path, "w") as f:
        for b in boxes:
            cls_id = int(b[0])
            # Remap Roboflow 3-class export → correct 2-class
            if cls_id > NUM_CLASSES - 1:
                cls_id = NUM_CLASSES - 1
            cx, cy, w, h = b[1], b[2], b[3], b[4]
            cx = max(0.001, min(0.999, cx))
            cy = max(0.001, min(0.999, cy))
            w  = max(0.001, min(0.999, w))
            h  = max(0.001, min(0.999, h))
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def clip_boxes(boxes: list[list[float]]) -> list[list[float]]:
    """Remove boxes that have drifted outside the frame."""
    valid = []
    for b in boxes:
        cx, cy, w, h = b[1], b[2], b[3], b[4]
        x1 = cx - w / 2; y1 = cy - h / 2
        x2 = cx + w / 2; y2 = cy + h / 2
        x1 = max(0.0, x1); y1 = max(0.0, y1)
        x2 = min(1.0, x2); y2 = min(1.0, y2)
        if x2 - x1 > 0.01 and y2 - y1 > 0.01:
            valid.append([b[0], (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
    return valid


# ─────────────────────────────────────────────────────────────────────────────
# Individual augmentation transforms
# ─────────────────────────────────────────────────────────────────────────────

def flip_h(img: np.ndarray, boxes: list) -> tuple[np.ndarray, list]:
    img = cv2.flip(img, 1)
    new = [[b[0], 1.0 - b[1], b[2], b[3], b[4]] for b in boxes]
    return img, new


def flip_v(img: np.ndarray, boxes: list) -> tuple[np.ndarray, list]:
    img = cv2.flip(img, 0)
    new = [[b[0], b[1], 1.0 - b[2], b[3], b[4]] for b in boxes]
    return img, new


def rotate(img: np.ndarray, boxes: list, angle_deg: float) -> tuple[np.ndarray, list]:
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    new_boxes = []
    for b in boxes:
        bx, by = b[1] * w, b[2] * h
        bw, bh = b[3] * w, b[4] * h
        corners = np.array([
            [bx - bw / 2, by - bh / 2],
            [bx + bw / 2, by - bh / 2],
            [bx + bw / 2, by + bh / 2],
            [bx - bw / 2, by + bh / 2],
        ])
        ones = np.ones((4, 1))
        corners_h = np.hstack([corners, ones])
        rotated = (M @ corners_h.T).T
        x_min, y_min = rotated[:, 0].min(), rotated[:, 1].min()
        x_max, y_max = rotated[:, 0].max(), rotated[:, 1].max()
        new_boxes.append([
            b[0],
            ((x_min + x_max) / 2) / w,
            ((y_min + y_max) / 2) / h,
            (x_max - x_min) / w,
            (y_max - y_min) / h,
        ])
    return img, new_boxes


def brightness_contrast(img: np.ndarray, boxes: list) -> tuple[np.ndarray, list]:
    alpha = random.uniform(BRIGHT_LOW, BRIGHT_HIGH)   # contrast
    beta  = random.uniform(-30, 30)                    # brightness
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img, boxes


def gaussian_blur(img: np.ndarray, boxes: list) -> tuple[np.ndarray, list]:
    k = random.choice(BLUR_KERNELS)
    img = cv2.GaussianBlur(img, (k, k), 0)
    return img, boxes


def random_crop(img: np.ndarray, boxes: list,
                min_crop: float = 0.7) -> tuple[np.ndarray, list]:
    h, w = img.shape[:2]
    scale = random.uniform(min_crop, 1.0)
    new_w = int(w * scale); new_h = int(h * scale)
    x0 = random.randint(0, w - new_w)
    y0 = random.randint(0, h - new_h)

    img = img[y0:y0 + new_h, x0:x0 + new_w]
    img = cv2.resize(img, (w, h))

    new_boxes = []
    for b in boxes:
        # Transform box coordinates relative to crop window
        bx = (b[1] * w - x0) / new_w
        by = (b[2] * h - y0) / new_h
        bw = b[3] * w / new_w
        bh = b[4] * h / new_h
        new_boxes.append([b[0], bx, by, bw, bh])
    return img, clip_boxes(new_boxes)


def mosaic(
    imgs:   list[np.ndarray],
    labels: list[list[list[float]]],
    size:   int = IMG_SIZE,
) -> tuple[np.ndarray, list]:
    """4-image mosaic (takes exactly 4 images)."""
    s = size // 2
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    all_boxes = []

    positions = [(0, 0), (s, 0), (0, s), (s, s)]
    for i, (img, boxes) in enumerate(zip(imgs[:4], labels[:4])):
        ri = cv2.resize(img, (s, s))
        ox, oy = positions[i]
        canvas[oy:oy + s, ox:ox + s] = ri
        for b in boxes:
            bx = (b[1] * s + ox) / size
            by = (b[2] * s + oy) / size
            bw = b[3] * s / size
            bh = b[4] * s / size
            all_boxes.append([b[0], bx, by, bw, bh])

    return canvas, clip_boxes(all_boxes)


# ─────────────────────────────────────────────────────────────────────────────
# Per-image augmentation worker
# ─────────────────────────────────────────────────────────────────────────────

def _augment_one(args: tuple) -> int:
    src_img_path, src_lbl_path, dst_images, dst_labels, n, neighbours = args
    img = cv2.imread(str(src_img_path))
    if img is None:
        return 0
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    boxes = read_labels(src_lbl_path)
    stem = src_img_path.stem
    generated = 0

    for i in range(n):
        aug_img   = img.copy()
        aug_boxes = [list(b) for b in boxes]

        # Random sequence of transforms
        if random.random() < 0.5:
            aug_img, aug_boxes = flip_h(aug_img, aug_boxes)
        if random.random() < 0.3:
            aug_img, aug_boxes = flip_v(aug_img, aug_boxes)
        if random.random() < 0.6:
            angle = random.uniform(-ROT_MAX_DEG, ROT_MAX_DEG)
            aug_img, aug_boxes = rotate(aug_img, aug_boxes, angle)
        if random.random() < 0.7:
            aug_img, aug_boxes = brightness_contrast(aug_img, aug_boxes)
        if random.random() < 0.4:
            aug_img, aug_boxes = gaussian_blur(aug_img, aug_boxes)
        if random.random() < 0.4:
            aug_img, aug_boxes = random_crop(aug_img, aug_boxes)
        # Mosaic every ~5th image
        if random.random() < 0.2 and len(neighbours) >= 3:
            mosaic_imgs   = [aug_img] + [cv2.resize(cv2.imread(str(p)), (IMG_SIZE, IMG_SIZE))
                                         for p in random.sample(neighbours, 3)
                                         if cv2.imread(str(p)) is not None][:3]
            mosaic_lbls   = [aug_boxes] + [read_labels(p.with_suffix(".txt"))
                                            for p in random.sample(neighbours, 3)][:3]
            if len(mosaic_imgs) == 4:
                aug_img, aug_boxes = mosaic(mosaic_imgs, mosaic_lbls)

        aug_boxes = clip_boxes(aug_boxes)

        out_name = f"{stem}_aug_{i:04d}"
        cv2.imwrite(str(Path(dst_images) / (out_name + ".jpg")), aug_img)
        write_labels(Path(dst_labels) / (out_name + ".txt"), aug_boxes)
        generated += 1

    return generated


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_augmentation(
    src_dir: str,
    dst_dir: str,
    n: int = AUGMENT_PER_IMAGE,
    workers: int = 0,
) -> None:
    src = Path(src_dir)
    dst = Path(dst_dir)

    images_src = src / "images"
    labels_src = src / "labels"
    images_dst = dst / "images"
    labels_dst = dst / "labels"

    for d in (images_dst, labels_dst):
        d.mkdir(parents=True, exist_ok=True)

    # Copy originals first
    img_paths = [
        p for p in images_src.iterdir()
        if p.suffix.lower() in VALID_EXTENSIONS
    ]
    logger.info(f"Found {len(img_paths)} source images.")

    for p in img_paths:
        shutil.copy2(p, images_dst / p.name)
        lbl = labels_src / p.with_suffix(".txt").name
        if lbl.is_file():
            shutil.copy2(lbl, labels_dst / lbl.name)

    neighbours = img_paths  # used for mosaic

    tasks = [
        (
            p,
            labels_src / p.with_suffix(".txt").name,
            str(images_dst),
            str(labels_dst),
            n,
            [x for x in neighbours if x != p],
        )
        for p in img_paths
    ]

    num_workers = workers or max(1, mp.cpu_count() - 1)
    logger.info(f"Augmenting {len(tasks)} images × {n} variants using {num_workers} workers…")
    t0 = time.time()

    with mp.Pool(num_workers) as pool:
        results = pool.map(_augment_one, tasks)

    total = sum(results) + len(img_paths)
    elapsed = time.time() - t0
    logger.info(f"Done: {total} total images in {elapsed:.1f}s → {dst}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 dataset augmentation pipeline")
    parser.add_argument("--src", required=True, help="Source dataset root (contains images/ labels/)")
    parser.add_argument("--dst", required=True, help="Destination dataset root")
    parser.add_argument("--n",   type=int, default=AUGMENT_PER_IMAGE, help="Augmentations per image")
    parser.add_argument("--workers", type=int, default=0, help="CPU workers (0 = auto)")
    args = parser.parse_args()
    run_augmentation(args.src, args.dst, args.n, args.workers)
