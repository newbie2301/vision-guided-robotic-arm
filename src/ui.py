"""
ui.py
Vision-Guided Robotic Arm – Tkinter Graphical User Interface

Dark colour palette:
    Background  #0f1117
    Accent blue #3B8BD4
    Success     #1D9E75
    Amber       #EF9F27
    Red         #E24B4A
    Purple      #7F77DD

Layout:
    Left  (60 %)  – live camera feed with YOLO annotation overlay
    Right (40 %)  – state machine panel, joint sliders, tray lock,
                    Arduino status, controls, event log
"""

from __future__ import annotations

import logging
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Optional

import cv2
import numpy as np

from .coordinator  import Coordinator, State, STATE_COLOURS
from .arduino_comm import ArduinoComm
from .coord_mapper import CoordMapper
from .infer        import VisionThread

logger = logging.getLogger(__name__)

# ── Palette ────────────────────────────────────────────────────────────────────
BG          = "#0f1117"
FG          = "#e0e0e0"
ACCENT      = "#3B8BD4"
SUCCESS     = "#1D9E75"
AMBER       = "#EF9F27"
RED         = "#E24B4A"
PURPLE      = "#7F77DD"
PANEL_BG    = "#1a1d26"
BTN_BG      = "#252836"

UI_REFRESH_MS  = 50    # 20 Hz
CAMERA_W       = 800
CAMERA_H       = 480
LOG_MAX_LINES  = 200


class ArmUI:
    """Main application window."""

    def __init__(self, root: tk.Tk):
        self._root = root
        root.title("Vision-Guided Robotic Arm  –  Control Panel")
        root.configure(bg=BG)
        root.resizable(True, True)

        # ── Subsystem instances ───────────────────────────────────────────────
        self._mapper   = CoordMapper()
        self._arduino  = ArduinoComm(
            stub_mode=True,                           # change to False for hardware
            on_status_change=self._on_arduino_status,
        )
        self._coordinator = Coordinator(
            arduino=self._arduino,
            mapper=self._mapper,
            on_state_change=self._on_state_change,
            on_tray_lock=self._on_tray_lock,
            on_log=self._log,
        )
        self._vision = VisionThread(
            on_detection=self._on_detection,
        )

        # ── Shared state ──────────────────────────────────────────────────────
        self._calibrating    = False
        self._cal_points:    list[tuple[float, float]] = []
        self._cal_real_pts:  list[tuple[float, float]] = []

        # ── Build UI ──────────────────────────────────────────────────────────
        self._build_layout()
        self._load_calibration()

        # ── Connect Arduino ───────────────────────────────────────────────────
        threading.Thread(target=self._arduino.connect, daemon=True).start()

        # ── Start vision (will fail gracefully if model absent) ───────────────
        threading.Thread(target=self._vision.start, daemon=True).start()

        # ── Start UI refresh loop ─────────────────────────────────────────────
        self._root.after(UI_REFRESH_MS, self._refresh_ui)

        # ── Shutdown hook ─────────────────────────────────────────────────────
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ─────────────────────────────────────────────────────────────────────────
    # Layout builders
    # ─────────────────────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        self._root.columnconfigure(0, weight=6)
        self._root.columnconfigure(1, weight=4)
        self._root.rowconfigure(0, weight=1)

        self._build_camera_panel()
        self._build_control_panel()

    def _build_camera_panel(self) -> None:
        frame = tk.Frame(self._root, bg=BG)
        frame.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)

        self._camera_canvas = tk.Canvas(
            frame, width=CAMERA_W, height=CAMERA_H, bg="#000000",
            highlightthickness=1, highlightbackground=ACCENT,
        )
        self._camera_canvas.pack(fill=tk.BOTH, expand=True)
        self._camera_canvas.bind("<Button-1>", self._on_canvas_click)

        self._canvas_image_id = None

    def _build_control_panel(self) -> None:
        frame = tk.Frame(self._root, bg=BG)
        frame.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)
        frame.columnconfigure(0, weight=1)

        row = 0

        # ── Emergency stop ────────────────────────────────────────────────────
        self._btn_estop = tk.Button(
            frame, text="⊘  EMERGENCY STOP", bg=RED, fg="white",
            font=("Helvetica", 11, "bold"), relief=tk.FLAT,
            command=self._coordinator.emergency_stop,
        )
        self._btn_estop.grid(row=row, column=0, sticky="ew", pady=(0, 6)); row += 1

        # ── State machine panel ───────────────────────────────────────────────
        sm_frame = self._titled_frame(frame, "State Machine", row); row += 1
        self._state_labels: dict[State, tk.Label] = {}
        for s in State:
            lbl = tk.Label(sm_frame, text=f"  {s.name}", bg=PANEL_BG, fg="#888",
                           font=("Consolas", 9), anchor="w")
            lbl.pack(fill=tk.X)
            self._state_labels[s] = lbl
        self._state_banner = tk.Label(
            sm_frame, text="IDLE", bg=PANEL_BG, fg=ACCENT,
            font=("Helvetica", 14, "bold"),
        )
        self._state_banner.pack(pady=4)

        # ── Manual joint control ──────────────────────────────────────────────
        jf = self._titled_frame(frame, "Manual Joint Control", row); row += 1
        self._joint_sliders: list[tk.Scale] = []
        joint_info = [
            ("Base Yaw",  0,   180, 90),
            ("Shoulder",  10,  170, 125),
            ("Elbow",     40,  180, 180),
            ("Gripper",   42,  100, 60),
        ]
        for name, lo, hi, default in joint_info:
            row_f = tk.Frame(jf, bg=PANEL_BG)
            row_f.pack(fill=tk.X, pady=1)
            tk.Label(row_f, text=f"{name}", bg=PANEL_BG, fg=FG,
                     font=("Consolas", 9), width=10, anchor="w").pack(side=tk.LEFT)
            sl = tk.Scale(row_f, from_=lo, to=hi, orient=tk.HORIZONTAL,
                          bg=PANEL_BG, fg=FG, troughcolor=BTN_BG,
                          highlightthickness=0, bd=0, length=160)
            sl.set(default)
            sl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self._joint_sliders.append(sl)
        tk.Button(jf, text="Send All Joints", bg=BTN_BG, fg=FG,
                  font=("Consolas", 9), relief=tk.FLAT,
                  command=self._send_manual_joints).pack(fill=tk.X, pady=2)

        # ── Arduino status ────────────────────────────────────────────────────
        af = self._titled_frame(frame, "Arduino", row); row += 1
        self._lbl_arduino = tk.Label(af, text="Connecting…", bg=PANEL_BG,
                                     fg=AMBER, font=("Consolas", 9))
        self._lbl_arduino.pack(anchor="w")
        tk.Button(af, text="⟳ Reconnect", bg=BTN_BG, fg=FG,
                  font=("Consolas", 9), relief=tk.FLAT,
                  command=self._reconnect_arduino).pack(fill=tk.X, pady=2)

        # ── Tray lock ──────────────────────────────────────────────────────────
        tf = self._titled_frame(frame, "Tray Lock", row); row += 1
        self._lbl_tray = tk.Label(tf, text="SEARCHING… 0%", bg=PANEL_BG,
                                  fg=AMBER, font=("Consolas", 9))
        self._lbl_tray.pack(anchor="w")
        self._tray_progress = ttk.Progressbar(tf, length=200, mode="determinate")
        self._tray_progress.pack(fill=tk.X, pady=2)
        self._lbl_balls = tk.Label(tf, text="Balls in tray: 0", bg=PANEL_BG,
                                   fg=FG, font=("Consolas", 9))
        self._lbl_balls.pack(anchor="w")
        tk.Button(tf, text="⟳ Reset Tray Lock", bg=BTN_BG, fg=FG,
                  font=("Consolas", 9), relief=tk.FLAT,
                  command=self._coordinator.reset_tray).pack(fill=tk.X, pady=2)

        # ── Main controls ─────────────────────────────────────────────────────
        cf = self._titled_frame(frame, "Controls", row); row += 1
        btn_row = tk.Frame(cf, bg=PANEL_BG)
        btn_row.pack(fill=tk.X)
        tk.Button(btn_row, text="▶ START", bg=SUCCESS, fg="white",
                  font=("Helvetica", 10, "bold"), relief=tk.FLAT,
                  command=self._coordinator.start).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        tk.Button(btn_row, text="■ STOP", bg=RED, fg="white",
                  font=("Helvetica", 10, "bold"), relief=tk.FLAT,
                  command=self._coordinator.stop).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        btn_row2 = tk.Frame(cf, bg=PANEL_BG)
        btn_row2.pack(fill=tk.X, pady=2)
        tk.Button(btn_row2, text="⏸ PAUSE",   bg=BTN_BG, fg=FG,
                  font=("Consolas", 9), relief=tk.FLAT,
                  command=lambda: self._log("Pause not yet implemented")).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        tk.Button(btn_row2, text="▶ RESUME", bg=BTN_BG, fg=FG,
                  font=("Consolas", 9), relief=tk.FLAT,
                  command=lambda: self._log("Resume not yet implemented")).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        demo_row = tk.Frame(cf, bg=PANEL_BG)
        demo_row.pack(fill=tk.X, pady=2)
        for label, cmd in [
            ("⊘ Skip State", self._coordinator.skip_state),
            ("↺ Retry",      self._coordinator.retry),
            ("⌂ Home",       lambda: self._arduino.send_home()),
            ("✓ DOF Check",  self._run_dof_check),
        ]:
            tk.Button(demo_row, text=label, bg=BTN_BG, fg=FG,
                      font=("Consolas", 8), relief=tk.FLAT,
                      command=cmd).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        tk.Button(cf, text="✓ OK (Close & Place)", bg=PURPLE, fg="white",
                  font=("Consolas", 9), relief=tk.FLAT,
                  command=self._coordinator.confirm_pick).pack(fill=tk.X, pady=2)

        # ── Calibration ────────────────────────────────────────────────────────
        cal_f = self._titled_frame(frame, "Coordinate Calibration", row); row += 1
        tk.Button(cal_f, text="Begin 4-Point Calibration", bg=BTN_BG, fg=FG,
                  font=("Consolas", 8), relief=tk.FLAT,
                  command=self._begin_calibration).pack(fill=tk.X)
        self._lbl_cal = tk.Label(cal_f, text="Not calibrated", bg=PANEL_BG,
                                 fg=AMBER, font=("Consolas", 8))
        self._lbl_cal.pack(anchor="w")

        # ── Event log ──────────────────────────────────────────────────────────
        lf = self._titled_frame(frame, "Log", row); row += 1
        self._log_text = tk.Text(lf, height=8, bg=PANEL_BG, fg="#aaaaaa",
                                 font=("Consolas", 7), state=tk.DISABLED,
                                 relief=tk.FLAT, wrap=tk.WORD)
        sb = tk.Scrollbar(lf, command=self._log_text.yview)
        self._log_text.config(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._log_text.pack(fill=tk.BOTH, expand=True)
        tk.Button(lf, text="Clear", bg=BTN_BG, fg=FG,
                  font=("Consolas", 7), relief=tk.FLAT,
                  command=self._clear_log).pack(anchor="e")

    def _titled_frame(self, parent: tk.Widget, title: str, row: int) -> tk.Frame:
        outer = tk.Frame(parent, bg=PANEL_BG, bd=1, relief=tk.FLAT)
        outer.grid(row=row, column=0, sticky="ew", pady=2, padx=0)
        outer.columnconfigure(0, weight=1)
        tk.Label(outer, text=f" {title}", bg=PANEL_BG, fg=ACCENT,
                 font=("Helvetica", 8, "bold"), anchor="w").pack(fill=tk.X)
        inner = tk.Frame(outer, bg=PANEL_BG, padx=4, pady=2)
        inner.pack(fill=tk.BOTH, expand=True)
        return inner

    # ─────────────────────────────────────────────────────────────────────────
    # UI refresh (20 Hz)
    # ─────────────────────────────────────────────────────────────────────────

    def _refresh_ui(self) -> None:
        try:
            self._update_camera()
            self._update_state_panel()
            self._update_tray_panel()
        except Exception as exc:
            logger.error(f"UI refresh error: {exc}")
        self._root.after(UI_REFRESH_MS, self._refresh_ui)

    def _update_camera(self) -> None:
        frame = self._vision.get_frame()
        if frame is None:
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        from PIL import Image, ImageTk  # type: ignore
        img = Image.fromarray(frame_rgb)
        img = img.resize((CAMERA_W, CAMERA_H), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self._camera_canvas.config(width=CAMERA_W, height=CAMERA_H)
        if self._canvas_image_id is None:
            self._canvas_image_id = self._camera_canvas.create_image(0, 0, anchor="nw", image=photo)
        else:
            self._camera_canvas.itemconfig(self._canvas_image_id, image=photo)
        self._camera_canvas._photo = photo   # prevent GC

    def _update_state_panel(self) -> None:
        state = self._coordinator.get_state()
        colour = self._coordinator.get_state_colour()
        for s, lbl in self._state_labels.items():
            if s == state:
                lbl.config(fg=colour, font=("Consolas", 9, "bold"))
                lbl.config(text=f"● {s.name}")
            else:
                lbl.config(fg="#555", font=("Consolas", 9))
                lbl.config(text=f"  {s.name}")
        self._state_banner.config(text=state.name, fg=colour)

    def _update_tray_panel(self) -> None:
        tray = self._coordinator.get_locked_tray()
        progress = self._coordinator.get_tray_progress()
        balls = self._coordinator.get_balls_in_tray()
        self._tray_progress["value"] = progress * 100
        if tray:
            pos = f"({tray.x_cm:.1f}, {tray.y_cm:.1f}) cm"
            self._lbl_tray.config(text=f"LOCKED at {pos}", fg=SUCCESS)
        else:
            self._lbl_tray.config(
                text=f"SEARCHING… {int(progress * 100)}%", fg=AMBER)
        self._lbl_balls.config(text=f"Balls in tray: {balls}")

    # ─────────────────────────────────────────────────────────────────────────
    # Callbacks
    # ─────────────────────────────────────────────────────────────────────────

    def _on_arduino_status(self, connected: bool) -> None:
        colour = SUCCESS if connected else RED
        text   = f"Connected on {self._arduino._port}" if connected else "Disconnected"
        def _update():
            self._lbl_arduino.config(text=text, fg=colour)
        self._root.after(0, _update)

    def _on_state_change(self, state: State) -> None:
        self._vision.set_state_label(state.name)

    def _on_tray_lock(self, locked: bool) -> None:
        pass  # tray panel updated by _refresh_ui

    def _on_detection(self, result) -> None:
        self._coordinator.feed_detection(result)
        # Push hints back to vision thread for overlay rendering
        self._vision.set_locked_tray(self._coordinator.get_locked_tray())
        self._vision.set_pick_target(self._coordinator.get_pick_target())
        self._vision.set_stable_metals(self._coordinator.get_stable_metals())

    def _on_canvas_click(self, event: tk.Event) -> None:
        if not self._calibrating:
            return
        # Scale click coordinates to original frame dimensions
        canvas_w = self._camera_canvas.winfo_width()  or CAMERA_W
        canvas_h = self._camera_canvas.winfo_height() or CAMERA_H
        px = event.x * 1280 / canvas_w
        py = event.y * 720  / canvas_h
        self._cal_points.append((px, py))
        idx = len(self._cal_points)
        self._log(f"Cal click {idx}/4 → pixel ({px:.0f},{py:.0f})")
        done = self._mapper.add_pixel_point(px, py)
        if done:
            self._calibrating = False
            self._mapper.save()
            self._lbl_cal.config(text="Calibrated ✓", fg=SUCCESS)
            self._log("Calibration complete and saved.")

    # ─────────────────────────────────────────────────────────────────────────
    # Button actions
    # ─────────────────────────────────────────────────────────────────────────

    def _send_manual_joints(self) -> None:
        angles = [int(sl.get()) for sl in self._joint_sliders]
        self._arduino.send_angles(angles)
        self._log(f"Manual joints sent: {angles}")

    def _reconnect_arduino(self) -> None:
        self._arduino.disconnect()
        threading.Thread(target=self._arduino.connect, daemon=True).start()

    def _run_dof_check(self) -> None:
        from .ik_solver import DOF_CHECK_SEQUENCE, HOME, SAFE_POSE
        def _run():
            for pose in DOF_CHECK_SEQUENCE:
                self._arduino.send_angles(pose)
                time.sleep(2.0)
        threading.Thread(target=_run, daemon=True).start()

    def _begin_calibration(self) -> None:
        # Prompt for real-world point values via simple dialog
        real_pts = [
            (0.0,  10.0),
            (20.0, 10.0),
            (20.0, 25.0),
            (0.0,  25.0),
        ]
        self._mapper.begin_calibration(real_pts)
        self._cal_points = []
        self._calibrating = True
        self._lbl_cal.config(text="Click 4 markers in camera…", fg=AMBER)
        self._log("Calibration mode: click 4 workspace markers in the camera feed.")

    def _load_calibration(self) -> None:
        if self._mapper.load():
            self._lbl_cal.config(text="Calibrated ✓ (loaded)", fg=SUCCESS)
        else:
            self._lbl_cal.config(text="Not calibrated", fg=AMBER)

    def _log(self, msg: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}\n"
        def _insert():
            self._log_text.config(state=tk.NORMAL)
            self._log_text.insert(tk.END, line)
            # Trim to max lines
            lines = int(self._log_text.index(tk.END).split(".")[0])
            if lines > LOG_MAX_LINES:
                self._log_text.delete("1.0", f"{lines - LOG_MAX_LINES}.0")
            self._log_text.see(tk.END)
            self._log_text.config(state=tk.DISABLED)
        self._root.after(0, _insert)

    def _clear_log(self) -> None:
        self._log_text.config(state=tk.NORMAL)
        self._log_text.delete("1.0", tk.END)
        self._log_text.config(state=tk.DISABLED)

    # ─────────────────────────────────────────────────────────────────────────
    # Shutdown
    # ─────────────────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        self._coordinator.emergency_stop()
        self._vision.stop()
        self._arduino.disconnect()
        self._root.destroy()


def launch() -> None:
    root = tk.Tk()
    _app = ArmUI(root)
    root.mainloop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    launch()
