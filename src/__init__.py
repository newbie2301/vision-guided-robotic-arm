"""
Vision-Guided Robotic Arm for Autonomous Metal Object Pick-and-Place

Package modules:
    infer          – YOLOv8 vision thread, Detection / FrameResult dataclasses
    coord_mapper   – Homography-based pixel→world coordinate mapping
    ik_solver      – Closed-form 2-DOF inverse kinematics + servo mappings
    arduino_comm   – Non-blocking serial communication with Arduino
    coordinator    – Finite state machine task coordinator + detection stabiliser
    ui             – Tkinter graphical user interface
"""

__version__ = "1.0.0"
