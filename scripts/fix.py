"""
fix.py
Vision-Guided Robotic Arm – Dataset Diagnostics and Label Correction

Provides utilities for:
    - Detecting and correcting label format inconsistencies
      (Roboflow 3-class export → 2-class YOLO, polygon → bbox)
    - Reporting dataset integrity statistics
    - Removing orphan label files (no matching image)
    - Removing images with empty label files

Usage:
    python fix.py --labels datasets/raw/labels --nc 2
    python fix.py --labels datasets/raw/labels --nc 2 --dry-run
    python fix.py --labels datasets/raw/labels --nc 2 --remove-orphans
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

VALID_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class LabelStats(NamedTuple):
    total_files:       int
    fixed_class_remap: int
    fixed_polygon:     int
    fixed_clipped:     int
    empty_labels:      int
    orphan_labels:     int


# ─────────────────────────────────────────────────────────────────────────────
# Core fix functions
# ─────────────────────────────────────────────────────────────────────────────

def _polygon_to_bbox(parts: list[str]) -> list[str] | None:
    """
    Convert polygon annotation (class + N×2 coords) to YOLO bbox.
    Returns None if conversion is not possible.
    """
    if len(parts) < 5:
        return None
    cls_id = parts[0]
    try:
        coords = [float(p) for p in parts[1:]]
    except ValueError:
        return None
    if len(coords) % 2 != 0:
        return None
    xs = coords[0::2]
    ys = coords[1::2]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    w  = max(xs) - min(xs)
    h  = max(ys) - min(ys)
    return [cls_id, f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]


def fix_label_file(
    label_path: Path,
    nc: int,
    dry_run: bool = False,
) -> tuple[bool, int, int, int]:
    """
    Fix a single YOLO label file.

    Returns:
        (changed, n_class_remaps, n_polygon_fixes, n_clip_fixes)
    """
    try:
        with open(label_path) as f:
            raw_lines = f.readlines()
    except Exception as exc:
        logger.error(f"Cannot read {label_path}: {exc}")
        return False, 0, 0, 0

    new_lines   = []
    n_remap     = 0
    n_poly      = 0
    n_clip      = 0
    changed     = False

    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()

        # ── Polygon → bbox ────────────────────────────────────────────────────
        if len(parts) > 5:
            fixed = _polygon_to_bbox(parts)
            if fixed:
                parts = fixed
                n_poly += 1
                changed = True
            else:
                logger.warning(f"  Cannot convert polygon: {label_path.name}: {line[:60]}")
                continue

        if len(parts) != 5:
            logger.warning(f"  Skipping malformed line in {label_path.name}: {line[:60]}")
            continue

        cls_id_raw = int(parts[0])

        # ── Class index remap ─────────────────────────────────────────────────
        cls_id = cls_id_raw
        if cls_id >= nc:
            cls_id = nc - 1
            n_remap += 1
            changed = True
        if cls_id < 0:
            cls_id = 0
            n_remap += 1
            changed = True

        # ── Coordinate clipping ───────────────────────────────────────────────
        try:
            cx, cy, w, h = (float(p) for p in parts[1:])
        except ValueError:
            logger.warning(f"  Non-numeric coords in {label_path.name}: {line[:60]}")
            continue

        orig = (cx, cy, w, h)
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w  = max(0.001, min(1.0, w))
        h  = max(0.001, min(1.0, h))

        # Ensure box doesn't extend beyond image
        x1 = cx - w / 2; x2 = cx + w / 2
        y1 = cy - h / 2; y2 = cy + h / 2
        x1 = max(0.0, x1); x2 = min(1.0, x2)
        y1 = max(0.0, y1); y2 = min(1.0, y2)
        if x2 - x1 < 0.005 or y2 - y1 < 0.005:
            logger.debug(f"  Dropping degenerate box in {label_path.name}")
            changed = True
            continue
        cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
        w  = x2 - x1;        h  = y2 - y1

        if (cx, cy, w, h) != orig:
            n_clip += 1
            changed = True

        new_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    if changed and not dry_run:
        backup = label_path.with_suffix(".bak")
        shutil.copy2(label_path, backup)
        with open(label_path, "w") as f:
            f.write("\n".join(new_lines) + ("\n" if new_lines else ""))

    return changed, n_remap, n_poly, n_clip


# ─────────────────────────────────────────────────────────────────────────────
# Dataset-level operations
# ─────────────────────────────────────────────────────────────────────────────

def fix_all_labels(
    labels_dir: str,
    nc: int,
    dry_run: bool = False,
) -> LabelStats:
    ldir = Path(labels_dir)
    if not ldir.is_dir():
        logger.error(f"Labels directory not found: {ldir}")
        return LabelStats(0, 0, 0, 0, 0, 0)

    label_files = list(ldir.glob("*.txt"))
    logger.info(f"Found {len(label_files)} label files in {ldir}")

    total_remap = total_poly = total_clip = 0
    empty_count = 0

    for lf in label_files:
        if lf.stat().st_size == 0:
            empty_count += 1
            continue
        changed, n_r, n_p, n_c = fix_label_file(lf, nc=nc, dry_run=dry_run)
        total_remap += n_r
        total_poly  += n_p
        total_clip  += n_c

    mode_str = "[DRY RUN] " if dry_run else ""
    logger.info(
        f"{mode_str}Fix summary: "
        f"{total_remap} class remaps, "
        f"{total_poly} polygon→bbox conversions, "
        f"{total_clip} coordinate clips, "
        f"{empty_count} empty label files."
    )
    return LabelStats(len(label_files), total_remap, total_poly, total_clip, empty_count, 0)


def find_orphans(
    images_dir: str,
    labels_dir: str,
) -> tuple[list[Path], list[Path]]:
    """
    Return (images_without_labels, labels_without_images).
    """
    idir = Path(images_dir)
    ldir = Path(labels_dir)

    img_stems = {p.stem for p in idir.iterdir() if p.suffix.lower() in VALID_IMG_EXT}
    lbl_stems = {p.stem for p in ldir.glob("*.txt")}

    imgs_missing_lbl = [idir / (s + ".jpg") for s in img_stems - lbl_stems]
    lbls_missing_img = [ldir / (s + ".txt") for s in lbl_stems - img_stems]

    logger.info(
        f"Orphan check: "
        f"{len(imgs_missing_lbl)} images without labels, "
        f"{len(lbls_missing_img)} labels without images."
    )
    return imgs_missing_lbl, lbls_missing_img


def remove_orphan_labels(labels_dir: str, images_dir: str, dry_run: bool = False) -> int:
    _, orphan_labels = find_orphans(images_dir, labels_dir)
    removed = 0
    for lf in orphan_labels:
        logger.info(f"  {'[DRY] ' if dry_run else ''}Removing orphan label: {lf.name}")
        if not dry_run:
            lf.unlink()
        removed += 1
    return removed


def print_dataset_report(images_dir: str, labels_dir: str, nc: int) -> None:
    """Print a concise dataset integrity report."""
    idir = Path(images_dir)
    ldir = Path(labels_dir)

    img_files = [p for p in idir.iterdir() if p.suffix.lower() in VALID_IMG_EXT]
    lbl_files = list(ldir.glob("*.txt"))

    class_counts = {i: 0 for i in range(nc)}
    boxes_total  = 0

    for lf in lbl_files:
        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls = int(parts[0])
                    if 0 <= cls < nc:
                        class_counts[cls] = class_counts.get(cls, 0) + 1
                        boxes_total += 1

    print("\n══ Dataset Report ══════════════════════════════════")
    print(f"  Images      : {len(img_files)}")
    print(f"  Label files : {len(lbl_files)}")
    print(f"  Total boxes : {boxes_total}")
    for cls_id, count in class_counts.items():
        pct = count / boxes_total * 100 if boxes_total else 0
        print(f"  Class {cls_id:>2d}     : {count:>6d} boxes  ({pct:.1f}%)")
    print("════════════════════════════════════════════════════\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset label fixer and diagnostics")
    parser.add_argument("--labels",          required=True, help="Path to labels/ directory")
    parser.add_argument("--images",          default="",    help="Path to images/ directory (for orphan check)")
    parser.add_argument("--nc",              type=int, default=2, help="Number of classes")
    parser.add_argument("--dry-run",         action="store_true", help="Preview changes without writing")
    parser.add_argument("--remove-orphans",  action="store_true", help="Delete orphan label files")
    parser.add_argument("--report",          action="store_true", help="Print dataset integrity report")
    args = parser.parse_args()

    fix_all_labels(args.labels, nc=args.nc, dry_run=args.dry_run)

    if args.images:
        if args.remove_orphans:
            remove_orphan_labels(args.labels, args.images, dry_run=args.dry_run)
        else:
            find_orphans(args.images, args.labels)

        if args.report:
            print_dataset_report(args.images, args.labels, nc=args.nc)
