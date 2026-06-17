"""
infer.py
Vision-Guided Robotic Arm – YOLOv8 Vision Thread

Runs YOLOv8 inference in a dedicated background thread, continuously
reading frames from a camera source and annotating detections.

Camera source options:
    CAMERA_INDEX = 0         → local USB webcam
    CAMERA_INDEX = "http://IP:PORT/video"  → DroidCam Wi-Fi stream

Detection overlay colours:
    dim green   – raw Metal-Object bounding boxes
    bright green – stabilised Metal-Object tracks
    dim blue    – raw Tray bounding boxes
    purple dashed – locked tray position
    cyan        – selected pick target
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
CAMERA_INDEX     = 0                            # int or URL string
MODEL_PATH       = "models/best.pt"            # trained YOLOv8n weights
CONF_THRESHOLD   = 0.40
IOU_THRESHOLD    = 0.45
INPUT_SIZE       = 640
TARGET_FPS       = 30

# Class indices (must match dataset YAML)
CLASS_METAL      = 0
CLASS_TRAY       = 1


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Detection:
    """Single YOLO detection result."""
    class_id: int
    confidence: float
    x1: float; y1: float; x2: float; y2: float
    cx: float = 0.0; cy: float = 0.0          # bounding box centroid (pixels)
    x_cm: float = 0.0; y_cm: float = 0.0      # world coords (filled by CoordMapper)

    def __post_init__(self):
        self.cx = (self.x1 + self.x2) / 2.0
        self.cy = (self.y1 + self.y2) / 2.0

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)

    def iou(self, other: "Detection") -> float:
        """Intersection-over-Union with another detection."""
        ix1 = max(self.x1, other.x1); iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2); iy2 = min(self.y2, other.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_self  = (self.x2 - self.x1)  * (self.y2 - self.y1)
        area_other = (other.x2 - other.x1) * (other.y2 - other.y1)
        return inter / (area_self + area_other - inter)


@dataclass
class FrameResult:
    """All detections for one camera frame."""
    metals:           list[Detection] = field(default_factory=list)
    trays:            list[Detection] = field(default_factory=list)
    stable_metals:    list[Detection] = field(default_factory=list)
    pick_target:      Optional[Detection] = None
    locked_tray:      Optional[Detection] = None
    fps:              float = 0.0
    state_label:      str = "IDLE"
    annotated_frame:  Optional[np.ndarray] = None


# ── Colours (BGR) ─────────────────────────────────────────────────────────────
C_METAL_RAW      = (0,   160, 0)
C_METAL_STABLE   = (0,   255, 0)
C_TRAY_RAW       = (160, 0,   0)
C_TRAY_LOCKED    = (200, 0,   200)
C_PICK_TARGET    = (255, 255, 0)
C_TEXT           = (220, 220, 220)


class VisionThread:
    """
    Background thread that runs continuous YOLOv8 inference.

    Callbacks:
        on_detection(FrameResult) – called after every processed frame.
    """

    def __init__(
        self,
        model_path: str = MODEL_PATH,
        camera_index = CAMERA_INDEX,
        conf: float = CONF_THRESHOLD,
        iou: float  = IOU_THRESHOLD,
        on_detection: Optional[Callable[[FrameResult], None]] = None,
    ):
        self._model_path    = model_path
        self._camera_index  = camera_index
        self._conf          = conf
        self._iou           = iou
        self._on_detection  = on_detection

        self._model         = None
        self._cap           = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event    = threading.Event()
        self._lock          = threading.Lock()
        self._latest_result: Optional[FrameResult] = None

        # External state hints (set by Coordinator)
        self._state_label   = "IDLE"
        self._locked_tray:  Optional[Detection] = None
        self._pick_target:  Optional[Detection] = None
        self._stable_metals: list[Detection]    = []

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Load model, open camera, start inference thread."""
        if not self._load_model():
            return False
        if not self._open_camera():
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._inference_loop,
            name="VisionThread",
            daemon=True,
        )
        self._thread.start()
        logger.info("VisionThread started.")
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        if self._cap:
            self._cap.release()
        logger.info("VisionThread stopped.")

    # ─────────────────────────────────────────────────────────────────────────
    # Thread-safe getters
    # ─────────────────────────────────────────────────────────────────────────

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest_result and self._latest_result.annotated_frame is not None:
                return self._latest_result.annotated_frame.copy()
        return None

    def get_result(self) -> Optional[FrameResult]:
        with self._lock:
            return self._latest_result

    # ─────────────────────────────────────────────────────────────────────────
    # State hints from Coordinator
    # ─────────────────────────────────────────────────────────────────────────

    def set_state_label(self, label: str) -> None:
        self._state_label = label

    def set_locked_tray(self, det: Optional[Detection]) -> None:
        self._locked_tray = det

    def set_pick_target(self, det: Optional[Detection]) -> None:
        self._pick_target = det

    def set_stable_metals(self, metals: list[Detection]) -> None:
        self._stable_metals = metals

    # ─────────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────────

    def _load_model(self) -> bool:
        try:
            from ultralytics import YOLO  # type: ignore
            self._model = YOLO(self._model_path)
            logger.info(f"YOLOv8 model loaded: {self._model_path}")
            return True
        except Exception as exc:
            logger.error(f"Failed to load model: {exc}")
            return False

    def _open_camera(self) -> bool:
        try:
            self._cap = cv2.VideoCapture(self._camera_index)
            if not self._cap.isOpened():
                raise RuntimeError("Cannot open camera.")
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
            logger.info(f"Camera opened: {self._camera_index}")
            return True
        except Exception as exc:
            logger.error(f"Camera open failed: {exc}")
            return False

    def _inference_loop(self) -> None:
        frame_times: list[float] = []
        while not self._stop_event.is_set():
            t0 = time.time()
            ret, frame = self._cap.read()
            if not ret:
                logger.warning("Camera read failed – retrying.")
                time.sleep(0.1)
                continue

            # Run YOLO inference
            try:
                results = self._model.predict(
                    frame,
                    conf=self._conf,
                    iou=self._iou,
                    imgsz=INPUT_SIZE,
                    verbose=False,
                )
            except Exception as exc:
                logger.error(f"Inference error: {exc}")
                continue

            metals, trays = self._parse_results(results, frame.shape)

            # Annotate frame
            annotated = frame.copy()
            self._draw_detections(annotated, metals, trays)
            self._draw_hud(annotated, frame_times)

            # Compute FPS
            frame_times.append(time.time() - t0)
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0

            result = FrameResult(
                metals=metals,
                trays=trays,
                stable_metals=list(self._stable_metals),
                pick_target=self._pick_target,
                locked_tray=self._locked_tray,
                fps=fps,
                state_label=self._state_label,
                annotated_frame=annotated,
            )

            with self._lock:
                self._latest_result = result

            if self._on_detection:
                try:
                    self._on_detection(result)
                except Exception as exc:
                    logger.error(f"on_detection callback error: {exc}")

    def _parse_results(
        self, results, frame_shape
    ) -> tuple[list[Detection], list[Detection]]:
        metals: list[Detection] = []
        trays:  list[Detection] = []
        h, w = frame_shape[:2]

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls  = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                # Clip to frame
                x1 = max(0.0, min(x1, w)); x2 = max(0.0, min(x2, w))
                y1 = max(0.0, min(y1, h)); y2 = max(0.0, min(y2, h))
                det = Detection(cls, conf, x1, y1, x2, y2)
                if cls == CLASS_METAL:
                    metals.append(det)
                elif cls == CLASS_TRAY:
                    trays.append(det)
        return metals, trays

    def _draw_detections(
        self,
        frame: np.ndarray,
        metals: list[Detection],
        trays:  list[Detection],
    ) -> None:
        # Raw metal detections
        for d in metals:
            cv2.rectangle(frame, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)),
                          C_METAL_RAW, 1)

        # Stable metal detections
        for d in self._stable_metals:
            cv2.rectangle(frame, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)),
                          C_METAL_STABLE, 2)
            cv2.putText(frame, f"Metal {d.confidence:.2f}",
                        (int(d.x1), int(d.y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_METAL_STABLE, 1)

        # Pick target
        if self._pick_target:
            d = self._pick_target
            cv2.rectangle(frame, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)),
                          C_PICK_TARGET, 3)
            cv2.putText(frame, "TARGET",
                        (int(d.x1), int(d.y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_PICK_TARGET, 2)

        # Raw tray detections
        for d in trays:
            cv2.rectangle(frame, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)),
                          C_TRAY_RAW, 1)

        # Locked tray (dashed border approximation)
        if self._locked_tray:
            d = self._locked_tray
            self._draw_dashed_rect(frame, int(d.x1), int(d.y1), int(d.x2), int(d.y2),
                                   C_TRAY_LOCKED, 2)
            cv2.putText(frame, "TRAY LOCKED",
                        (int(d.x1), int(d.y2) + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_TRAY_LOCKED, 1)

    @staticmethod
    def _draw_dashed_rect(
        img: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
        color: tuple, thickness: int, dash: int = 10
    ) -> None:
        pts = [(x1, y1, x2, y1), (x2, y1, x2, y2),
               (x2, y2, x1, y2), (x1, y2, x1, y1)]
        for ax, ay, bx, by in pts:
            length = int(((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5)
            if length == 0:
                continue
            dx = (bx - ax) / length; dy = (by - ay) / length
            seg = 0
            while seg < length:
                seg_end = min(seg + dash, length)
                p1 = (int(ax + dx * seg),     int(ay + dy * seg))
                p2 = (int(ax + dx * seg_end), int(ay + dy * seg_end))
                cv2.line(img, p1, p2, color, thickness)
                seg += dash * 2

    def _draw_hud(self, frame: np.ndarray, frame_times: list[float]) -> None:
        fps = (1.0 / (sum(frame_times) / len(frame_times))) if frame_times else 0.0
        cv2.putText(frame, f"FPS: {fps:.1f}  State: {self._state_label}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_TEXT, 1)
