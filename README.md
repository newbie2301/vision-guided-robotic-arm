# Vision-Guided Robotic Arm for Autonomous Metal Object Pick-and-Place

A low-cost, 4-DOF robotic arm that autonomously detects metallic objects using a YOLOv8 deep-learning vision pipeline and performs pick-and-place operations into a designated tray. Built as a prototype for the course **21MEO106T – Introduction to Robotics** at SRM Institute of Science and Technology.

> Submitted by Vedant Tiwari, Fatin Ahmed, J. Yaswanth — under the guidance of Dr. A. Vijaya, Department of Mechanical Engineering.

## Overview

The system integrates three engineering disciplines into a single working prototype:

- **Computer vision** — a custom-trained YOLOv8n model detects `Metal-Object` and `Tray` classes in real time.
- **Inverse kinematics** — closed-form trigonometric IK computes shoulder/elbow joint angles for a 2-DOF planar arm segment, with empirically calibrated servo mappings.
- **Embedded control** — an Arduino Uno executes smooth, interpolated servo motion over a paced, non-blocking serial link.

A homography-based coordinate mapper converts camera pixel coordinates into real-world centimetre positions, and a ten-state finite state machine coordinates the full task lifecycle — including a **permanent tray lock** that survives camera occlusion during pick motions.

## Demo Results

| Metric | Achieved | Target |
|---|---|---|
| mAP@50 | ~0.980 | > 0.97 |
| Recall | ~1.000 | > 0.95 |
| Pick-and-place success rate (40 trials) | 80% | — |
| Coordinate mapping error (centre / edge) | 1.2 cm / 2.5 cm | — |
| Full pick-to-place cycle | 10–15 s | — |

## Repository Structure

```
intelligent_arm/
├── main.py                      # Application entry point (launches UI)
├── requirements.txt
├── configs/
│   └── data.yaml                 # YOLOv8 dataset configuration template
├── src/
│   ├── infer.py                  # VisionThread – YOLOv8 inference + overlay
│   ├── coord_mapper.py           # CoordMapper – 4-point homography calibration
│   ├── ik_solver.py               # Closed-form 2-DOF IK + servo angle mappings
│   ├── arduino_comm.py            # Non-blocking serial communication layer
│   ├── coordinator.py             # 10-state FSM + DetectionStabiliser
│   └── ui.py                      # Tkinter dark-themed control panel
├── scripts/
│   ├── augment_dataset.py         # Data augmentation pipeline (20x per image)
│   ├── train_yolov.py             # YOLOv8 training wrapper with sanity checks
│   ├── fix.py                     # Label format correction & diagnostics
│   ├── bound_viewer.py            # Interactive bounding-box label viewer
│   └── dbtest.py                  # Dataset integrity test suite
├── arduino/
│   └── arm_controller/
│       └── arm_controller.ino     # Servo firmware (smoothMove interpolation)
└── docs/
    └── (architecture diagrams, report)
```

## Hardware

| Component | Spec |
|---|---|
| Arm structure | 3D-printed PLA, FDM, 40% infill, 0.2 mm layer |
| Servos (×4) | SG90 micro servo, 9 g, ~1.8 kg-cm stall torque @ 5 V |
| Controller | Arduino Uno (ATmega328P, 16 MHz) |
| Camera | Smartphone via DroidCam (Wi-Fi MJPEG) or USB webcam |
| Power | 5 V / 2 A external supply for servos, separate from Arduino USB |

**Link lengths:** L1 (shoulder→elbow) = 12.6 cm, L2 (elbow→gripper) = 10.2 cm, base height = 3.2 cm. Reach: 6.0–22.8 cm.

**Pin mapping:** J1→Pin 3, J2→Pin 5, J3→Pin 6, J4→Pin 9.

## Software Stack

Python 3.10+, Ultralytics YOLOv8, OpenCV, PyTorch, NumPy, PySerial, Tkinter, Pillow.

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the detection model (optional — bring your own dataset)

```bash
python scripts/dbtest.py --data configs/data.yaml
python scripts/augment_dataset.py --src datasets/raw --dst datasets/augmented --n 20
python scripts/train_yolov.py train --data configs/data.yaml --epochs 150
```

Place the resulting `best.pt` at `models/best.pt` (referenced by `src/infer.py`).

### 3. Flash the Arduino

Open `arduino/arm_controller/arm_controller.ino` in the Arduino IDE, select **Arduino Uno**, and upload.

### 4. Run the application

```bash
python main.py
```

By default `ArduinoComm` starts in **stub mode** (no physical hardware required) so the full software pipeline can be exercised on a development machine. Set `stub_mode=False` in `src/ui.py` and configure the correct serial port to drive real hardware.

### 5. Calibrate the workspace

In the UI, click **Begin 4-Point Calibration**, then click the four physical reference markers in the live camera feed in order. The homography matrix is saved to `calibration.json` and auto-loads on subsequent runs.

## State Machine

```
IDLE → INIT → WAIT_TRAY → SCANNING → PICKING → WAIT_CONFIRM →
PLACING → VERIFYING → SUCCESS
                ↑___________________________________|
                (loop while objects remain outside tray)
```

`ERROR` is reachable from any state on DOF-check failure, IK failure, scan timeout, or repeated verification failure (max 3 retries).

## Key Engineering Contributions

1. **Permanent tray lock** — decouples placement targeting from continuous tray visibility, eliminating all placement failures caused by arm-body occlusion during testing.
2. **Automated label correction pipeline** — detects and fixes Roboflow's 3-class export quirk and polygon-format annotations before augmentation/training.
3. **Multi-threaded modular architecture** — concurrent vision inference (30 FPS), UI refresh (20 Hz), and paced serial actuation (1.7 s/command) with full `threading.Lock` / `threading.Event` synchronisation.

## Known Limitations & Future Work

- Coordinate mapping error grows near workspace edges (~2.5 cm) due to uncorrected lens distortion.
- Single-plane (sagittal) IK — no wrist rotation for non-symmetric object orientation.
- See Chapter 12 of the full project report for the complete future-improvements roadmap (depth sensing, current-sensing grasp detection, Raspberry Pi port, etc.).

## License

MIT — see [LICENSE](LICENSE).

## References

See the full project report for the complete bibliography, including Ultralytics YOLOv8, Redmon et al. (YOLO), Craig's *Introduction to Robotics*, Hartley & Zisserman's *Multiple View Geometry*, and the OpenCV library.
