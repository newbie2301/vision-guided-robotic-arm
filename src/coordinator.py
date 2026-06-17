"""
coordinator.py
Vision-Guided Robotic Arm – Coordinator / State Machine

Implements the ten-state task lifecycle FSM:
    IDLE → INIT → WAIT_TRAY → SCANNING → PICKING →
    WAIT_CONFIRM → PLACING → VERIFYING → SUCCESS → ERROR

Also contains DetectionStabiliser: a per-track IoU history filter
that promotes consistently-detected objects to stable pick candidates.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional

from .infer        import Detection, FrameResult
from .ik_solver    import (
    JointAngles, IKResult,
    build_pick_sequence, build_place_sequence,
    home_angles, safe_pose_angles,
    DOF_CHECK_SEQUENCE, HOME, SAFE_POSE,
)
from .arduino_comm import ArduinoComm
from .coord_mapper import CoordMapper

logger = logging.getLogger(__name__)

# ── Tuning constants ───────────────────────────────────────────────────────────
TRAY_CONFIRM_FRAMES    = 8    # consecutive frames with stable tray before lock
STABLE_TRACK_FRAMES    = 5    # frames a metal detection must persist before pick
IOU_MATCH_THRESHOLD    = 0.35 # minimum IoU for track association
MAX_PICK_RETRIES       = 3
VERIFY_TIMEOUT_S       = 4.0  # seconds to wait for object-in-tray confirmation
SCAN_TIMEOUT_S         = 15.0 # seconds before ERROR if no object found


class State(Enum):
    IDLE         = auto()
    INIT         = auto()
    WAIT_TRAY    = auto()
    SCANNING     = auto()
    PICKING      = auto()
    WAIT_CONFIRM = auto()
    PLACING      = auto()
    VERIFYING    = auto()
    SUCCESS      = auto()
    ERROR        = auto()


# State → UI accent colour (hex)
STATE_COLOURS: dict[State, str] = {
    State.IDLE:         "#3B8BD4",
    State.INIT:         "#3B8BD4",
    State.WAIT_TRAY:    "#7F77DD",
    State.SCANNING:     "#1D9E75",
    State.PICKING:      "#EF9F27",
    State.WAIT_CONFIRM: "#7F77DD",
    State.PLACING:      "#EF9F27",
    State.VERIFYING:    "#1D9E75",
    State.SUCCESS:      "#1D9E75",
    State.ERROR:        "#E24B4A",
}


# ─────────────────────────────────────────────────────────────────────────────
# Detection Stabiliser
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Track:
    det: Detection
    age: int = 1          # frames this track has been confirmed


class DetectionStabiliser:
    """
    Maintains per-track detection history for Metal-Object detections.
    A track promoted to 'stable' when it persists for STABLE_TRACK_FRAMES frames.
    """

    def __init__(
        self,
        min_frames: int = STABLE_TRACK_FRAMES,
        iou_thresh: float = IOU_MATCH_THRESHOLD,
    ):
        self._min = min_frames
        self._iou = iou_thresh
        self._tracks: list[_Track] = []

    def update(self, raw_detections: list[Detection]) -> list[Detection]:
        """
        Feed one frame of raw detections; return currently stable detections.
        """
        matched_new: list[bool] = [False] * len(raw_detections)
        next_tracks: list[_Track] = []

        for track in self._tracks:
            best_idx   = -1
            best_iou   = self._iou
            for i, det in enumerate(raw_detections):
                if matched_new[i]:
                    continue
                iou = track.det.iou(det)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_idx >= 0:
                matched_new[best_idx] = True
                track.det = raw_detections[best_idx]
                track.age += 1
                next_tracks.append(track)
            # else: track lost this frame – drop it

        # New unmatched detections start fresh tracks
        for i, det in enumerate(raw_detections):
            if not matched_new[i]:
                next_tracks.append(_Track(det=det, age=1))

        self._tracks = next_tracks
        return [t.det for t in self._tracks if t.age >= self._min]

    def reset(self) -> None:
        self._tracks = []


# ─────────────────────────────────────────────────────────────────────────────
# Coordinator
# ─────────────────────────────────────────────────────────────────────────────

class Coordinator:
    """
    Central state machine that wires Vision → CoordMapper → IKSolver → Arduino.
    """

    def __init__(
        self,
        arduino: ArduinoComm,
        mapper:  CoordMapper,
        on_state_change: Optional[Callable[[State], None]] = None,
        on_tray_lock:    Optional[Callable[[bool], None]]  = None,
        on_log:          Optional[Callable[[str], None]]   = None,
    ):
        self._arduino         = arduino
        self._mapper          = mapper
        self._on_state_change = on_state_change
        self._on_tray_lock    = on_tray_lock
        self._on_log          = on_log

        self._state           = State.IDLE
        self._lock            = threading.Lock()
        self._stabiliser      = DetectionStabiliser()

        self._locked_tray:    Optional[Detection] = None
        self._tray_conf_count = 0
        self._pick_target:    Optional[Detection] = None
        self._stable_metals:  list[Detection]     = []
        self._balls_in_tray   = 0
        self._pick_retries    = 0
        self._scan_start_time = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Public control API
    # ─────────────────────────────────────────────────────────────────────────

    def start(self) -> None:
        with self._lock:
            if self._state not in (State.IDLE, State.ERROR, State.SUCCESS):
                return
            self._transition(State.INIT)
            self._exec_init()

    def stop(self) -> None:
        self._arduino.flush_queue()
        self._arduino.send_home()
        with self._lock:
            self._transition(State.IDLE)

    def emergency_stop(self) -> None:
        self._arduino.flush_queue()
        self._arduino.send_home()
        with self._lock:
            self._transition(State.IDLE)
        self._log("EMERGENCY STOP")

    def confirm_pick(self) -> None:
        """Called by UI/operator to confirm grasp before placement."""
        with self._lock:
            if self._state == State.WAIT_CONFIRM:
                self._transition(State.PLACING)
                threading.Thread(target=self._exec_place, daemon=True).start()

    def reset_tray(self) -> None:
        with self._lock:
            self._locked_tray    = None
            self._tray_conf_count = 0
            self._balls_in_tray  = 0
            if self._on_tray_lock:
                self._on_tray_lock(False)
        self._log("Tray lock reset.")

    def retry(self) -> None:
        with self._lock:
            if self._state == State.ERROR:
                self._pick_retries = 0
                self._transition(State.SCANNING)

    def skip_state(self) -> None:
        """Debug helper: advance to next logical state."""
        with self._lock:
            advance = {
                State.WAIT_TRAY:    State.SCANNING,
                State.SCANNING:     State.IDLE,
                State.PICKING:      State.WAIT_CONFIRM,
                State.WAIT_CONFIRM: State.PLACING,
                State.PLACING:      State.VERIFYING,
                State.VERIFYING:    State.SCANNING,
            }
            next_s = advance.get(self._state)
            if next_s:
                self._transition(next_s)

    # ─────────────────────────────────────────────────────────────────────────
    # Detection feed (called by VisionThread callback)
    # ─────────────────────────────────────────────────────────────────────────

    def feed_detection(self, result: FrameResult) -> None:
        with self._lock:
            state = self._state

        if state == State.WAIT_TRAY:
            self._process_tray_detection(result.trays)

        elif state == State.SCANNING:
            stable = self._stabiliser.update(result.metals)
            with self._lock:
                self._stable_metals = stable
            if stable:
                self._select_pick_target(stable)
            elif (time.time() - self._scan_start_time) > SCAN_TIMEOUT_S:
                self._transition(State.ERROR)
                self._log("Scan timeout – no stable metal object found.")

        elif state == State.VERIFYING:
            self._verify_placement(result)

    # ─────────────────────────────────────────────────────────────────────────
    # State getters (UI reads these)
    # ─────────────────────────────────────────────────────────────────────────

    def get_state(self) -> State:
        return self._state

    def get_state_colour(self) -> str:
        return STATE_COLOURS.get(self._state, "#ffffff")

    def get_locked_tray(self) -> Optional[Detection]:
        return self._locked_tray

    def get_stable_metals(self) -> list[Detection]:
        return list(self._stable_metals)

    def get_pick_target(self) -> Optional[Detection]:
        return self._pick_target

    def get_balls_in_tray(self) -> int:
        return self._balls_in_tray

    def get_tray_progress(self) -> float:
        """0.0 – 1.0 progress toward tray lock."""
        return min(1.0, self._tray_conf_count / TRAY_CONFIRM_FRAMES)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal state machine actions
    # ─────────────────────────────────────────────────────────────────────────

    def _exec_init(self) -> None:
        """Send DOF check sequence; transition to WAIT_TRAY on completion."""
        def _run():
            self._log("INIT: running DOF check sequence…")
            for pose in DOF_CHECK_SEQUENCE:
                self._arduino.send_angles(pose)
                time.sleep(2.0)   # each smoothMove ≈ 1.6 s + margin
            with self._lock:
                self._transition(State.WAIT_TRAY)
            self._log("INIT done → WAIT_TRAY")
        threading.Thread(target=_run, daemon=True).start()

    def _process_tray_detection(self, trays: list[Detection]) -> None:
        if self._locked_tray:
            return
        if trays:
            self._tray_conf_count += 1
        else:
            self._tray_conf_count = max(0, self._tray_conf_count - 1)

        if self._tray_conf_count >= TRAY_CONFIRM_FRAMES:
            # Lock on the highest-confidence tray
            best = max(trays, key=lambda d: d.confidence)
            # Map to world coordinates
            if self._mapper.is_calibrated:
                world = self._mapper.pixel_to_world(best.cx, best.cy)
                if world:
                    best.x_cm, best.y_cm = world
            self._locked_tray = best
            if self._on_tray_lock:
                self._on_tray_lock(True)
            self._log(f"Tray LOCKED at pixel ({best.cx:.0f},{best.cy:.0f}) "
                      f"→ world ({best.x_cm:.1f},{best.y_cm:.1f}) cm")
            self._stabiliser.reset()
            self._scan_start_time = time.time()
            self._transition(State.SCANNING)

    def _select_pick_target(self, stable: list[Detection]) -> None:
        """Choose nearest stable metal object as pick target."""
        # Select closest to base (smallest y_cm after world mapping)
        candidates = []
        for d in stable:
            if self._mapper.is_calibrated:
                world = self._mapper.pixel_to_world(d.cx, d.cy)
                if world:
                    d.x_cm, d.y_cm = world
                    candidates.append(d)
        if not candidates:
            return

        target = min(candidates, key=lambda d: d.y_cm)
        with self._lock:
            self._pick_target = target
            self._transition(State.PICKING)

        threading.Thread(target=self._exec_pick, args=(target,), daemon=True).start()

    def _exec_pick(self, target: Detection) -> None:
        self._log(f"PICKING: target at ({target.x_cm:.1f},{target.y_cm:.1f}) cm")
        sequence = build_pick_sequence(target.x_cm, target.y_cm)
        if not sequence:
            self._log("IK failed for pick target.")
            self._pick_retries += 1
            if self._pick_retries >= MAX_PICK_RETRIES:
                self._transition(State.ERROR)
            else:
                self._transition(State.SCANNING)
            return

        for angles in sequence:
            self._arduino.send_angles(angles.as_list())
            time.sleep(2.0)

        self._log(f"  Pick approach complete – awaiting confirm.")
        self._transition(State.WAIT_CONFIRM)

    def _exec_place(self) -> None:
        if not self._locked_tray:
            self._log("No locked tray – cannot place.")
            self._transition(State.ERROR)
            return

        tray = self._locked_tray
        self._log(f"PLACING at ({tray.x_cm:.1f},{tray.y_cm:.1f}) cm")
        sequence = build_place_sequence(tray.x_cm, tray.y_cm)
        if not sequence:
            self._log("IK failed for placement.")
            self._transition(State.ERROR)
            return

        # Lift first
        self._arduino.send_angles(safe_pose_angles().as_list())
        time.sleep(2.0)

        for angles in sequence:
            self._arduino.send_angles(angles.as_list())
            time.sleep(2.0)

        # Retract to safe pose
        self._arduino.send_angles(safe_pose_angles().as_list())
        time.sleep(2.0)

        self._log("Placement sequence complete → VERIFYING")
        self._transition(State.VERIFYING)
        self._verify_start = time.time()

    def _verify_placement(self, result: FrameResult) -> None:
        if not self._locked_tray:
            return

        tray = self._locked_tray
        for metal in result.metals:
            # Check if metal centroid lies within tray bbox
            if (tray.x1 <= metal.cx <= tray.x2 and
                    tray.y1 <= metal.cy <= tray.y2):
                self._balls_in_tray += 1
                self._pick_target = None
                self._stabiliser.reset()
                self._log(f"Verification OK – ball {self._balls_in_tray} in tray.")
                # Check if more objects remain
                with self._lock:
                    if self._stable_metals:
                        self._transition(State.SCANNING)
                        self._scan_start_time = time.time()
                    else:
                        self._exec_success()
                return

        # Timeout check
        if hasattr(self, '_verify_start'):
            elapsed = time.time() - self._verify_start
            if elapsed > VERIFY_TIMEOUT_S:
                self._pick_retries += 1
                self._log(f"Verify timeout ({elapsed:.1f}s) – retry {self._pick_retries}/{MAX_PICK_RETRIES}")
                if self._pick_retries >= MAX_PICK_RETRIES:
                    self._transition(State.ERROR)
                else:
                    self._transition(State.SCANNING)
                    self._scan_start_time = time.time()

    def _exec_success(self) -> None:
        self._log(f"SUCCESS – all {self._balls_in_tray} object(s) in tray.")
        self._arduino.send_home()
        self._transition(State.SUCCESS)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _transition(self, new_state: State) -> None:
        old = self._state
        self._state = new_state
        if old != new_state:
            logger.info(f"State: {old.name} → {new_state.name}")
            if self._on_state_change:
                try:
                    self._on_state_change(new_state)
                except Exception:
                    pass

    def _log(self, msg: str) -> None:
        logger.info(msg)
        if self._on_log:
            try:
                self._on_log(msg)
            except Exception:
                pass
