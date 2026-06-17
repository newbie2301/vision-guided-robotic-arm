"""
ik_solver.py
Vision-Guided Robotic Arm – Inverse Kinematics Solver

Computes closed-form 2-DOF planar IK for the shoulder (J2) and elbow (J3)
joints, plus base yaw (J1) and gripper (J4) angle mappings.

Physical arm parameters (measured from fabricated prototype):
    L1        = 12.6 cm   shoulder-to-elbow link
    L2        = 10.2 cm   elbow-to-gripper link
    BASE_H    =  3.2 cm   shoulder pivot height above table surface
    REACH_MAX = 22.8 cm   theoretical maximum reach (L1 + L2)
    REACH_MIN =  6.0 cm   practical minimum (elbow collapses)

IK convention:
    x_cm  – lateral offset from base centre (positive = right)
    y_cm  – depth from base centre (positive = forward)
    Origin at shoulder pivot projected onto table surface.

Servo mappings (empirically calibrated against physical arm):
    J2 servo = 26 + theta1_deg
    J3 real_angle = 180 - |theta2_deg|;  servo = (209.04 - real_angle) / 0.98
    J1 fixed at 90 deg for all forward pick-place ops
    J4 OPEN=100, CLOSED=42, HOME=60
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


# ── Physical parameters ───────────────────────────────────────────────────────
L1: float = 12.6          # cm  upper arm link length
L2: float = 10.2          # cm  forearm link length
BASE_H: float = 3.2       # cm  shoulder pivot height above table

REACH_MAX: float = L1 + L2          # 22.8 cm
REACH_MIN: float = 6.0              # cm  practical minimum

# ── Joint limits (degrees) ────────────────────────────────────────────────────
J1_MIN, J1_MAX = 0,  180
J2_MIN, J2_MAX = 10, 170
J3_MIN, J3_MAX = 40, 180
J4_MIN, J4_MAX = 42, 100

# ── Gripper constants ─────────────────────────────────────────────────────────
GRIPPER_OPEN   = 100
GRIPPER_CLOSED =  42
GRIPPER_HOME   =  60

# ── Named poses [J1, J2, J3, J4] ─────────────────────────────────────────────
HOME       = [90, 125, 180, GRIPPER_HOME]
SAFE_POSE  = [90, 60,  180, GRIPPER_HOME]   # raised collision-free transit

# Multi-step motion sequences (list of poses executed in order)
DOF_CHECK_SEQUENCE = [
    HOME,
    SAFE_POSE,
    HOME,
]

PICK_APPROACH_SEQUENCE_TEMPLATE = []   # filled by build_pick_sequence()
PICK_COMPLETE_SEQUENCE_TEMPLATE = []   # filled by build_pick_sequence()


@dataclass
class JointAngles:
    """Four joint angles sent to Arduino as 'j1,j2,j3,j4\\n'."""
    j1: int = 90
    j2: int = 125
    j3: int = 180
    j4: int = GRIPPER_HOME

    def as_list(self) -> list[int]:
        return [self.j1, self.j2, self.j3, self.j4]

    def as_command(self) -> str:
        return f"{self.j1},{self.j2},{self.j3},{self.j4}"

    def __str__(self) -> str:
        return f"J1={self.j1}° J2={self.j2}° J3={self.j3}° J4={self.j4}°"


@dataclass
class IKResult:
    """Result of an IK solve attempt."""
    success: bool
    angles: Optional[JointAngles] = None
    error: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Core IK solve
# ─────────────────────────────────────────────────────────────────────────────

def solve(x_cm: float, y_cm: float, gripper: int = GRIPPER_OPEN) -> IKResult:
    """
    Compute joint angles to position gripper tip at (x_cm, y_cm) on the table.

    Args:
        x_cm:    lateral position in robot base frame (right positive)
        y_cm:    depth position in robot base frame (forward positive)
        gripper: gripper servo angle (GRIPPER_OPEN or GRIPPER_CLOSED)

    Returns:
        IKResult with success flag and JointAngles.
    """
    # Adjust for shoulder pivot height above table
    y_adj = y_cm - BASE_H

    # Planar reach distance
    r = math.sqrt(x_cm ** 2 + y_adj ** 2)

    # Reachability check
    if r < REACH_MIN:
        return IKResult(False, error=f"Target too close: r={r:.2f} cm < {REACH_MIN} cm")
    if r > REACH_MAX:
        return IKResult(False, error=f"Target too far: r={r:.2f} cm > {REACH_MAX} cm")

    # Elbow angle via law of cosines
    cos_theta2 = (r ** 2 - L1 ** 2 - L2 ** 2) / (2.0 * L1 * L2)
    cos_theta2 = max(-1.0, min(1.0, cos_theta2))   # clamp for numerical safety
    theta2 = math.acos(cos_theta2)                 # radians, always >= 0

    # Shoulder angle (arctan2 with elbow correction)
    alpha = math.atan2(y_adj, x_cm)
    beta  = math.atan2(L2 * math.sin(theta2), L1 + L2 * math.cos(theta2))
    theta1 = alpha - beta                          # radians

    # Convert to degrees
    theta1_deg = math.degrees(theta1)
    theta2_deg = math.degrees(theta2)

    # ── Empirical servo mappings ──────────────────────────────────────────────
    j2_servo = int(round(26.0 + theta1_deg))

    real_angle_j3 = 180.0 - abs(theta2_deg)
    j3_servo = int(round((209.04 - real_angle_j3) / 0.98))

    # Base yaw: fixed at 90 (centred) for all forward pick-place operations
    j1_servo = 90

    # ── Apply joint limits ────────────────────────────────────────────────────
    j1_servo = int(_clamp(j1_servo, J1_MIN, J1_MAX))
    j2_servo = int(_clamp(j2_servo, J2_MIN, J2_MAX))
    j3_servo = int(_clamp(j3_servo, J3_MIN, J3_MAX))
    gripper   = int(_clamp(gripper,  J4_MIN, J4_MAX))

    return IKResult(
        success=True,
        angles=JointAngles(j1=j1_servo, j2=j2_servo, j3=j3_servo, j4=gripper),
    )


def build_pick_sequence(x_cm: float, y_cm: float) -> list[JointAngles]:
    """
    Generate the multi-step approach → descend → grasp sequence for a pick.

    Steps:
        1. approach_above  – arm moves to target XY at safe height
        2. descend         – arm descends to grasp height
        3. close_gripper   – gripper closes

    Returns list of JointAngles in execution order (retract is handled
    separately by the coordinator via SAFE_POSE).
    """
    # Approach: solve IK for a point slightly raised (y offset)
    approach_result = solve(x_cm, y_cm + 3.0, gripper=GRIPPER_OPEN)
    descend_result  = solve(x_cm, y_cm,        gripper=GRIPPER_OPEN)
    grasp_result    = solve(x_cm, y_cm,        gripper=GRIPPER_CLOSED)

    sequence: list[JointAngles] = []

    if approach_result.success and approach_result.angles:
        sequence.append(approach_result.angles)
    if descend_result.success and descend_result.angles:
        sequence.append(descend_result.angles)
    if grasp_result.success and grasp_result.angles:
        sequence.append(grasp_result.angles)

    return sequence


def build_place_sequence(x_cm: float, y_cm: float) -> list[JointAngles]:
    """
    Generate the approach → descend → release sequence for a placement.

    Returns list of JointAngles in execution order.
    """
    approach_result = solve(x_cm, y_cm + 3.0, gripper=GRIPPER_CLOSED)
    descend_result  = solve(x_cm, y_cm,        gripper=GRIPPER_CLOSED)
    release_result  = solve(x_cm, y_cm,        gripper=GRIPPER_OPEN)

    sequence: list[JointAngles] = []

    if approach_result.success and approach_result.angles:
        sequence.append(approach_result.angles)
    if descend_result.success and descend_result.angles:
        sequence.append(descend_result.angles)
    if release_result.success and release_result.angles:
        sequence.append(release_result.angles)

    return sequence


def home_angles() -> JointAngles:
    return JointAngles(*HOME)


def safe_pose_angles() -> JointAngles:
    return JointAngles(*SAFE_POSE)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def is_reachable(x_cm: float, y_cm: float) -> bool:
    """Quick reachability check without computing full IK."""
    y_adj = y_cm - BASE_H
    r = math.sqrt(x_cm ** 2 + y_adj ** 2)
    return REACH_MIN <= r <= REACH_MAX


# ─────────────────────────────────────────────────────────────────────────────
# CLI test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_points = [
        (0.0,  15.0),
        (5.0,  15.0),
        (-5.0, 15.0),
        (0.0,  20.0),
        (0.0,   8.0),
        (10.0, 10.0),
    ]
    print(f"{'x_cm':>8} {'y_cm':>8}  {'Result'}")
    print("-" * 55)
    for x, y in test_points:
        result = solve(x, y)
        if result.success:
            print(f"{x:>8.1f} {y:>8.1f}  {result.angles}")
        else:
            print(f"{x:>8.1f} {y:>8.1f}  FAIL – {result.error}")
