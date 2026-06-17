"""
train_yolov.py
Vision-Guided Robotic Arm – YOLOv8 Training Script

Wraps the Ultralytics training API with:
    - Dataset sanity checking (image/label count balance, path validation)
    - Timestamped run directory management
    - Early stopping (patience 30 epochs)
    - Automatic GPU/CPU selection

Usage:
    python train_yolov.py --data datasets/augmented/data.yaml
    python train_yolov.py --data datasets/augmented/data.yaml --epochs 150 --batch -1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL   = "yolov8n.pt"    # nano; pretrained on COCO
DEFAULT_EPOCHS  = 150
DEFAULT_IMGSZ   = 640
DEFAULT_BATCH   = -1              # auto
DEFAULT_WORKERS = 4
DEFAULT_PATIENCE = 30             # early stopping patience
VALID_IMG_EXT   = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset sanity check
# ─────────────────────────────────────────────────────────────────────────────

def check_dataset(data_yaml: str) -> bool:
    """
    Verify that every image file has a corresponding label file and
    that label class indices are within [0, num_classes).
    Returns True if the dataset passes all checks.
    """
    import yaml  # type: ignore

    yaml_path = Path(data_yaml)
    if not yaml_path.is_file():
        logger.error(f"data.yaml not found: {yaml_path}")
        return False

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    nc = cfg.get("nc", 0)
    splits_checked = 0
    errors = 0

    for split in ("train", "val", "test"):
        split_path = cfg.get(split)
        if split_path is None:
            continue

        # Resolve relative paths against yaml location
        img_dir = yaml_path.parent / split_path
        if not img_dir.is_dir():
            logger.warning(f"  [{split}] image dir not found: {img_dir}")
            continue

        lbl_dir = img_dir.parent.parent / "labels" / img_dir.name
        if not lbl_dir.is_dir():
            # Try sibling labels/ folder
            lbl_dir = img_dir.parent / "labels"

        img_files = [p for p in img_dir.iterdir() if p.suffix.lower() in VALID_IMG_EXT]
        lbl_files = {p.stem for p in lbl_dir.iterdir() if p.suffix == ".txt"} if lbl_dir.is_dir() else set()

        missing_lbls = [p for p in img_files if p.stem not in lbl_files]
        if missing_lbls:
            logger.warning(f"  [{split}] {len(missing_lbls)} images missing labels.")
            errors += len(missing_lbls)

        # Check class indices
        if lbl_dir.is_dir():
            for lbl_file in lbl_dir.glob("*.txt"):
                with open(lbl_file) as lf:
                    for line_no, line in enumerate(lf, 1):
                        parts = line.strip().split()
                        if not parts:
                            continue
                        cls_id = int(parts[0])
                        if cls_id < 0 or cls_id >= nc:
                            logger.warning(
                                f"  [{split}] {lbl_file.name}:{line_no} "
                                f"class {cls_id} out of range [0,{nc})"
                            )
                            errors += 1

        logger.info(f"  [{split}] {len(img_files)} images, {len(lbl_files)} labels. "
                    f"Missing: {len(missing_lbls)}")
        splits_checked += 1

    if splits_checked == 0:
        logger.error("No valid splits found in data.yaml.")
        return False

    if errors > 0:
        logger.warning(f"Dataset check completed with {errors} issue(s). Review before training.")
    else:
        logger.info("Dataset check passed – no issues found.")

    return errors == 0


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train(
    data_yaml: str,
    model_weights: str = DEFAULT_MODEL,
    epochs: int        = DEFAULT_EPOCHS,
    imgsz: int         = DEFAULT_IMGSZ,
    batch: int         = DEFAULT_BATCH,
    workers: int       = DEFAULT_WORKERS,
    patience: int      = DEFAULT_PATIENCE,
    device: str        = "",
    run_name: str      = "",
) -> Path:
    """
    Train YOLOv8n on the specified dataset.

    Returns:
        Path to the best.pt weights file.
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Timestamped run name
    if not run_name:
        run_name = f"arm_{time.strftime('%Y%m%d_%H%M%S')}"

    runs_dir = Path("runs") / run_name
    runs_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading base model: {model_weights}")
    model = YOLO(model_weights)

    logger.info(f"Starting training → {runs_dir}")
    logger.info(f"  data={data_yaml}  epochs={epochs}  imgsz={imgsz}  "
                f"batch={batch}  workers={workers}  patience={patience}")

    results = model.train(
        data      = data_yaml,
        epochs    = epochs,
        imgsz     = imgsz,
        batch     = batch,
        workers   = workers,
        patience  = patience,
        device    = device if device else None,
        project   = str(runs_dir.parent),
        name      = run_name,
        exist_ok  = True,
        save      = True,
        plots     = True,
        verbose   = True,
    )

    best_weights = runs_dir / "weights" / "best.pt"
    if best_weights.is_file():
        logger.info(f"Training complete. Best weights: {best_weights}")
    else:
        # Ultralytics may use a sub-folder
        candidates = list(runs_dir.rglob("best.pt"))
        if candidates:
            best_weights = candidates[0]
            logger.info(f"Best weights found at: {best_weights}")
        else:
            logger.warning("best.pt not found after training.")

    return best_weights


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate(weights_path: str, data_yaml: str, imgsz: int = DEFAULT_IMGSZ) -> None:
    """Run validation and print metrics."""
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        logger.error("ultralytics not installed.")
        return

    logger.info(f"Validating {weights_path} on {data_yaml} …")
    model = YOLO(weights_path)
    metrics = model.val(data=data_yaml, imgsz=imgsz)

    print("\n── Validation Metrics ─────────────────────────────")
    print(f"  Precision  (B): {metrics.box.mp:.4f}")
    print(f"  Recall     (B): {metrics.box.mr:.4f}")
    print(f"  mAP@50     (B): {metrics.box.map50:.4f}")
    print(f"  mAP@50-95  (B): {metrics.box.map:.4f}")
    print("────────────────────────────────────────────────────\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="YOLOv8 training wrapper for robotic arm dataset")
    sub = p.add_subparsers(dest="command")

    # train
    tr = sub.add_parser("train", help="Train YOLOv8n model")
    tr.add_argument("--data",    required=True,              help="Path to data.yaml")
    tr.add_argument("--model",   default=DEFAULT_MODEL,      help="Base weights")
    tr.add_argument("--epochs",  type=int, default=DEFAULT_EPOCHS)
    tr.add_argument("--imgsz",   type=int, default=DEFAULT_IMGSZ)
    tr.add_argument("--batch",   type=int, default=DEFAULT_BATCH)
    tr.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    tr.add_argument("--patience",type=int, default=DEFAULT_PATIENCE)
    tr.add_argument("--device",  default="",                 help="cuda:0 / cpu / mps")
    tr.add_argument("--name",    default="",                 help="Run name")
    tr.add_argument("--skip-check", action="store_true",     help="Skip dataset sanity check")

    # validate
    va = sub.add_parser("validate", help="Validate trained weights")
    va.add_argument("--weights", required=True, help="Path to best.pt")
    va.add_argument("--data",    required=True, help="Path to data.yaml")
    va.add_argument("--imgsz",   type=int, default=DEFAULT_IMGSZ)

    # check
    ck = sub.add_parser("check", help="Run dataset sanity check only")
    ck.add_argument("--data", required=True, help="Path to data.yaml")

    # Shortcut: no subcommand → default to train
    p.add_argument("--data",    default=None)
    p.add_argument("--model",   default=DEFAULT_MODEL)
    p.add_argument("--epochs",  type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--imgsz",   type=int, default=DEFAULT_IMGSZ)
    p.add_argument("--batch",   type=int, default=DEFAULT_BATCH)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--patience",type=int, default=DEFAULT_PATIENCE)
    p.add_argument("--device",  default="")
    p.add_argument("--name",    default="")
    p.add_argument("--skip-check", action="store_true")

    return p


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    cmd = getattr(args, "command", None)

    if cmd == "validate":
        validate(args.weights, args.data, args.imgsz)

    elif cmd == "check":
        ok = check_dataset(args.data)
        sys.exit(0 if ok else 1)

    else:
        # Default: train
        data = getattr(args, "data", None)
        if not data:
            parser.print_help()
            sys.exit(1)

        if not getattr(args, "skip_check", False):
            logger.info("Running dataset sanity check…")
            check_dataset(data)

        train(
            data_yaml      = data,
            model_weights  = args.model,
            epochs         = args.epochs,
            imgsz          = args.imgsz,
            batch          = args.batch,
            workers        = args.workers,
            patience       = args.patience,
            device         = args.device,
            run_name       = args.name,
        )
