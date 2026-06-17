"""
coord_mapper.py
Vision-Guided Robotic Arm – Homography Coordinate Mapper

Transforms camera pixel coordinates to real-world centimetre positions
in the robot base frame using a 4-point perspective homography.

Calibration procedure:
    1. Place four physical markers at known cm positions on the workspace.
    2. Call begin_calibration() then click each marker in the live feed.
    3. The 3x3 homography matrix H is computed via OpenCV and persisted
       to calibration.json for automatic reload on subsequent runs.

Robot base frame convention:
    x_cm  – lateral (positive = right)
    y_cm  – depth   (positive = forward / away from base)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CALIBRATION_FILE = "calibration.json"


@dataclass
class CalibPoint:
    px: float       # pixel x
    py: float       # pixel y
    cx: float       # real-world x (cm)
    cy: float       # real-world y (cm)


class CoordMapper:
    """
    4-point homography coordinate mapper.

    Usage:
        mapper = CoordMapper()
        mapper.load()                   # auto-load saved calibration

        # --- calibration mode ---
        mapper.begin_calibration(real_world_points)  # list of (cx, cy) tuples
        mapper.add_pixel_point(px, py)               # called 4 times
        mapper.finish_calibration()
        mapper.save()

        # --- inference ---
        x_cm, y_cm = mapper.pixel_to_world(px, py)
    """

    def __init__(self, calibration_file: str = CALIBRATION_FILE):
        self._cal_file = calibration_file
        self._H: Optional[np.ndarray] = None          # 3x3 homography matrix
        self._cal_real: list[tuple[float, float]] = []
        self._cal_pixels: list[tuple[float, float]] = []
        self._calibrating = False
        self.is_calibrated = False

    # ─────────────────────────────────────────────────────────────────────────
    # Calibration API
    # ─────────────────────────────────────────────────────────────────────────

    def begin_calibration(self, real_world_points: list[tuple[float, float]]) -> None:
        """
        Start a new calibration session.

        Args:
            real_world_points: ordered list of 4 (x_cm, y_cm) reference positions.
        """
        if len(real_world_points) != 4:
            raise ValueError("Exactly 4 real-world reference points required.")
        self._cal_real   = list(real_world_points)
        self._cal_pixels = []
        self._calibrating = True
        self.is_calibrated = False
        logger.info("Calibration started – click 4 markers in the camera feed.")

    def add_pixel_point(self, px: float, py: float) -> bool:
        """
        Register one pixel click during calibration.

        Returns True when all 4 points have been collected and H is computed.
        """
        if not self._calibrating:
            return False
        self._cal_pixels.append((px, py))
        logger.info(f"  Cal point {len(self._cal_pixels)}/4: pixel ({px:.0f}, {py:.0f})")

        if len(self._cal_pixels) == 4:
            self._compute_homography()
            return True
        return False

    def finish_calibration(self) -> bool:
        """Force-finish calibration if exactly 4 pixel points have been collected."""
        if len(self._cal_pixels) == 4 and self._H is None:
            self._compute_homography()
        return self.is_calibrated

    def _compute_homography(self) -> None:
        src = np.array(self._cal_pixels,  dtype=np.float32)   # pixel points
        dst = np.array(self._cal_real,    dtype=np.float32)   # world points (cm)
        self._H, _ = cv2.findHomography(src, dst)
        self._calibrating = False
        self.is_calibrated = self._H is not None
        if self.is_calibrated:
            logger.info("Homography computed successfully.")
        else:
            logger.error("Homography computation failed.")

    # ─────────────────────────────────────────────────────────────────────────
    # Coordinate conversion
    # ─────────────────────────────────────────────────────────────────────────

    def pixel_to_world(self, px: float, py: float) -> Optional[tuple[float, float]]:
        """
        Map pixel coordinate to real-world (x_cm, y_cm).

        Returns None if not calibrated.
        """
        if not self.is_calibrated or self._H is None:
            return None

        pt = np.array([[[px, py]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self._H)
        x_cm = float(result[0][0][0])
        y_cm = float(result[0][0][1])
        return x_cm, y_cm

    def world_to_pixel(self, x_cm: float, y_cm: float) -> Optional[tuple[float, float]]:
        """Inverse mapping: world cm → pixel (requires invertible H)."""
        if not self.is_calibrated or self._H is None:
            return None
        H_inv = np.linalg.inv(self._H)
        pt = np.array([[[x_cm, y_cm]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, H_inv)
        return float(result[0][0][0]), float(result[0][0][1])

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> bool:
        """Persist homography and calibration points to JSON."""
        path = path or self._cal_file
        if not self.is_calibrated or self._H is None:
            logger.warning("Cannot save: mapper not calibrated.")
            return False
        data = {
            "H":           self._H.tolist(),
            "cal_real":    self._cal_real,
            "cal_pixels":  self._cal_pixels,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Calibration saved → {path}")
        return True

    def load(self, path: Optional[str] = None) -> bool:
        """Load persisted homography from JSON. Returns True on success."""
        path = path or self._cal_file
        if not os.path.isfile(path):
            logger.info(f"No calibration file at {path} – manual calibration required.")
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            self._H = np.array(data["H"], dtype=np.float64)
            self._cal_real   = [tuple(p) for p in data.get("cal_real",   [])]
            self._cal_pixels = [tuple(p) for p in data.get("cal_pixels", [])]
            self.is_calibrated = True
            logger.info(f"Calibration loaded ← {path}")
            return True
        except Exception as exc:
            logger.error(f"Failed to load calibration: {exc}")
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def calibration_points_collected(self) -> int:
        return len(self._cal_pixels)

    @property
    def calibration_in_progress(self) -> bool:
        return self._calibrating

    @property
    def homography_matrix(self) -> Optional[np.ndarray]:
        return self._H.copy() if self._H is not None else None


# ─────────────────────────────────────────────────────────────────────────────
# CLI test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mapper = CoordMapper()
    # Simulate a 4-point calibration with known pixel↔world pairs
    real_pts = [(0.0, 10.0), (20.0, 10.0), (20.0, 25.0), (0.0, 25.0)]
    mapper.begin_calibration(real_pts)
    pixel_pts = [(100, 400), (500, 400), (500, 100), (100, 100)]
    for px, py in pixel_pts:
        mapper.add_pixel_point(px, py)
    # Test round-trip
    for (px, py), (cx, cy) in zip(pixel_pts, real_pts):
        result = mapper.pixel_to_world(px, py)
        print(f"pixel ({px},{py}) → world {result}  (expected ({cx},{cy}))")
