"""
dbtest.py
Vision-Guided Robotic Arm – Dataset Integrity Test Utility

Lightweight diagnostic tool that runs a battery of checks against
a YOLO-format dataset before training:

    1. data.yaml exists and parses correctly
    2. Class count (nc) matches names list length
    3. Every split directory referenced in data.yaml exists
    4. Image/label count balance per split
    5. No duplicate filenames across splits (train/val leakage check)
    6. Label files contain only valid class indices and normalised coords
    7. Basic image readability spot-check

Usage:
    python dbtest.py --data datasets/augmented/data.yaml
    python dbtest.py --data datasets/augmented/data.yaml --sample 50
"""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import cv2
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

VALID_IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class TestResult:
    def __init__(self):
        self.passed: list[str] = []
        self.failed: list[str] = []
        self.warnings: list[str] = []

    def ok(self, msg: str) -> None:
        self.passed.append(msg)
        logger.info(f"  ✓ {msg}")

    def fail(self, msg: str) -> None:
        self.failed.append(msg)
        logger.error(f"  ✗ {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning(f"  ! {msg}")

    @property
    def success(self) -> bool:
        return len(self.failed) == 0

    def summary(self) -> str:
        return (f"{len(self.passed)} passed, "
                f"{len(self.warnings)} warnings, "
                f"{len(self.failed)} failed")


def _resolve_split_dir(yaml_path: Path, split_value: str) -> Path:
    p = Path(split_value)
    if p.is_absolute():
        return p
    return (yaml_path.parent / split_value).resolve()


def _find_label_dir(img_dir: Path) -> Path:
    """Standard YOLO layout: .../images/<split> ↔ .../labels/<split>"""
    parts = list(img_dir.parts)
    if "images" in parts:
        idx = parts.index("images")
        label_parts = parts[:idx] + ["labels"] + parts[idx + 1:]
        return Path(*label_parts)
    return img_dir.parent / "labels"


def run_tests(data_yaml: str, sample_size: int = 30) -> TestResult:
    result = TestResult()
    yaml_path = Path(data_yaml)

    # ── Test 1: data.yaml exists and parses ───────────────────────────────────
    if not yaml_path.is_file():
        result.fail(f"data.yaml not found: {yaml_path}")
        return result
    try:
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        result.ok("data.yaml parsed successfully")
    except Exception as exc:
        result.fail(f"data.yaml failed to parse: {exc}")
        return result

    # ── Test 2: class count matches names ─────────────────────────────────────
    nc    = cfg.get("nc")
    names = cfg.get("names")
    if nc is None or names is None:
        result.fail("data.yaml missing 'nc' or 'names' field")
    elif isinstance(names, dict):
        if len(names) != nc:
            result.fail(f"nc={nc} but names has {len(names)} entries")
        else:
            result.ok(f"Class count matches: nc={nc}, names={list(names.values())}")
    elif isinstance(names, list):
        if len(names) != nc:
            result.fail(f"nc={nc} but names list has {len(names)} entries")
        else:
            result.ok(f"Class count matches: nc={nc}, names={names}")

    # ── Test 3 & 4: split directories and image/label balance ─────────────────
    all_stems: dict[str, str] = {}   # stem → split (for leakage check)
    splits_found = []

    for split in ("train", "val", "test"):
        split_value = cfg.get(split)
        if split_value is None:
            continue

        img_dir = _resolve_split_dir(yaml_path, split_value)
        if not img_dir.is_dir():
            result.fail(f"[{split}] directory does not exist: {img_dir}")
            continue

        label_dir = _find_label_dir(img_dir)
        if not label_dir.is_dir():
            result.fail(f"[{split}] label directory not found: {label_dir}")
            continue

        img_files = [p for p in img_dir.iterdir() if p.suffix.lower() in VALID_IMG_EXT]
        lbl_files = {p.stem: p for p in label_dir.glob("*.txt")}

        missing = [p for p in img_files if p.stem not in lbl_files]
        if missing:
            result.warn(f"[{split}] {len(missing)}/{len(img_files)} images missing labels")
        else:
            result.ok(f"[{split}] all {len(img_files)} images have matching labels")

        for p in img_files:
            if p.stem in all_stems and all_stems[p.stem] != split:
                result.warn(f"Possible leakage: '{p.stem}' appears in both "
                           f"'{all_stems[p.stem]}' and '{split}'")
            all_stems[p.stem] = split

        splits_found.append((split, img_dir, label_dir, img_files))

    if not splits_found:
        result.fail("No valid splits found – cannot proceed with further checks")
        return result

    # ── Test 5: label content validation ──────────────────────────────────────
    total_bad_lines = 0
    nc_value = cfg.get("nc", 0)
    for split, img_dir, label_dir, img_files in splits_found:
        for lbl_path in label_dir.glob("*.txt"):
            try:
                with open(lbl_path) as f:
                    for line_no, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) != 5:
                            total_bad_lines += 1
                            continue
                        cls_id = int(parts[0])
                        cx, cy, w, h = (float(p) for p in parts[1:])
                        if not (0 <= cls_id < nc_value):
                            total_bad_lines += 1
                        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                            total_bad_lines += 1
            except Exception:
                total_bad_lines += 1

    if total_bad_lines == 0:
        result.ok("All label lines have valid format and normalised coordinates")
    else:
        result.warn(f"{total_bad_lines} malformed/out-of-range label line(s) found")

    # ── Test 6: image readability spot-check ──────────────────────────────────
    all_imgs = [p for _, _, _, files in splits_found for p in files]
    sample = random.sample(all_imgs, min(sample_size, len(all_imgs))) if all_imgs else []
    unreadable = []
    for p in sample:
        img = cv2.imread(str(p))
        if img is None:
            unreadable.append(p)

    if unreadable:
        result.fail(f"{len(unreadable)}/{len(sample)} sampled images failed to load: "
                    f"{[p.name for p in unreadable[:5]]}")
    else:
        result.ok(f"Image readability spot-check passed ({len(sample)} images sampled)")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset integrity test suite")
    parser.add_argument("--data",   required=True, help="Path to data.yaml")
    parser.add_argument("--sample", type=int, default=30, help="Number of images to spot-check")
    args = parser.parse_args()

    print("\n══ Dataset Integrity Tests ════════════════════════════")
    result = run_tests(args.data, sample_size=args.sample)
    print("─────────────────────────────────────────────────────────")
    print(f"Result: {result.summary()}")
    print("══════════════════════════════════════════════════════════\n")

    import sys
    sys.exit(0 if result.success else 1)
