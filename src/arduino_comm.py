"""
arduino_comm.py
Vision-Guided Robotic Arm – Arduino Serial Communication Layer

Provides non-blocking serial communication with the Arduino Uno.
Commands are queued and dispatched by a background worker thread
at a fixed pacing interval, ensuring each smoothMove() completes
before the next command arrives.

Configuration:
    SERIAL_PORT  – COM port (Windows) or /dev/ttyUSB0 (Linux/Mac)
    BAUD_RATE    – must match Arduino firmware (115200)
    SEND_DELAY   – seconds between successive commands (≥ 1.7 s)
    STUB_MODE    – True = no real hardware; useful for software-only testing
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# ── Default configuration ──────────────────────────────────────────────────────
SERIAL_PORT  = "COM3"      # change to /dev/ttyUSB0 on Linux
BAUD_RATE    = 115200
SEND_DELAY   = 1.7         # seconds; covers smoothMove() 80 steps × 20 ms = 1.6 s
QUEUE_CAP    = 5           # bounded queue prevents unbounded accumulation
STUB_MODE    = False       # set True to run without physical Arduino


class ArduinoComm:
    """
    Thread-safe, non-blocking Arduino communication.

    Usage:
        comm = ArduinoComm(port="COM3")
        comm.connect()
        comm.send_angles([90, 125, 180, 60])
        comm.disconnect()
    """

    def __init__(
        self,
        port: str = SERIAL_PORT,
        baud: int = BAUD_RATE,
        send_delay: float = SEND_DELAY,
        stub_mode: bool = STUB_MODE,
        on_status_change: Optional[Callable[[bool], None]] = None,
    ):
        self._port        = port
        self._baud        = baud
        self._delay       = send_delay
        self._stub        = stub_mode
        self._on_status   = on_status_change

        self._serial      = None
        self._connected   = False
        self._lock        = threading.Lock()
        self._queue: queue.Queue[list[int]] = queue.Queue(maxsize=QUEUE_CAP)
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event  = threading.Event()

    # ─────────────────────────────────────────────────────────────────────────
    # Connection management
    # ─────────────────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Open serial port and start background worker thread."""
        if self._stub:
            logger.info("[STUB] ArduinoComm connected in stub mode.")
            self._connected = True
            self._start_worker()
            self._notify_status(True)
            return True

        try:
            import serial  # type: ignore
            self._serial = serial.Serial(self._port, self._baud, timeout=2)
            time.sleep(2)   # Arduino resets on serial open; wait for READY
            # Drain any startup messages
            while self._serial.in_waiting:
                line = self._serial.readline().decode(errors="replace").strip()
                logger.debug(f"Arduino → {line}")
            self._connected = True
            self._start_worker()
            self._notify_status(True)
            logger.info(f"Connected to Arduino on {self._port} @ {self._baud} baud.")
            return True
        except Exception as exc:
            logger.error(f"Arduino connection failed: {exc}")
            self._connected = False
            self._notify_status(False)
            return False

    def disconnect(self) -> None:
        """Stop worker thread and close serial port."""
        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)
        if self._serial and not self._stub:
            try:
                self._serial.close()
            except Exception:
                pass
        self._connected = False
        self._notify_status(False)
        logger.info("ArduinoComm disconnected.")

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ─────────────────────────────────────────────────────────────────────────
    # Command API
    # ─────────────────────────────────────────────────────────────────────────

    def send_angles(self, angles: list[int]) -> bool:
        """
        Enqueue a joint angle command [j1, j2, j3, j4].

        Returns False if the queue is full (command dropped).
        """
        if not self._connected:
            logger.warning("send_angles called while disconnected.")
            return False
        if len(angles) != 4:
            raise ValueError(f"Expected 4 joint angles, got {len(angles)}.")
        try:
            self._queue.put_nowait(list(angles))
            return True
        except queue.Full:
            logger.warning("Command queue full – angle command dropped.")
            return False

    def send_home(self) -> bool:
        return self.send_angles([90, 125, 180, 60])

    def send_safe_pose(self) -> bool:
        return self.send_angles([90, 60, 180, 60])

    def flush_queue(self) -> None:
        """Discard all pending commands (emergency stop support)."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        logger.info("Command queue flushed.")

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    # ─────────────────────────────────────────────────────────────────────────
    # Background worker
    # ─────────────────────────────────────────────────────────────────────────

    def _start_worker(self) -> None:
        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="ArduinoCommWorker",
            daemon=True,
        )
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        """Drain the command queue at the configured pacing rate."""
        while not self._stop_event.is_set():
            try:
                angles = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            self._transmit(angles)
            # Pace: wait for smoothMove() to complete before next command
            time.sleep(self._delay)

    def _transmit(self, angles: list[int]) -> None:
        cmd = ",".join(str(a) for a in angles)
        if self._stub:
            logger.info(f"[STUB] → {cmd}")
            return
        try:
            line = (cmd + "\n").encode()
            with self._lock:
                self._serial.write(line)
                self._serial.flush()
            logger.debug(f"TX → {cmd}")
            # Read acknowledgement
            if self._serial.in_waiting:
                ack = self._serial.readline().decode(errors="replace").strip()
                logger.debug(f"ACK ← {ack}")
        except Exception as exc:
            logger.error(f"Serial transmit error: {exc}")
            self._connected = False
            self._notify_status(False)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _notify_status(self, connected: bool) -> None:
        if self._on_status:
            try:
                self._on_status(connected)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# CLI test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    comm = ArduinoComm(stub_mode=True)
    comm.connect()
    comm.send_home()
    comm.send_angles([90, 80, 150, 100])
    comm.send_angles([45, 60, 140, 42])
    time.sleep(6)
    comm.disconnect()
