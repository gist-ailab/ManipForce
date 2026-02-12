"""
Franka REST + TCP Force Stream API
----------------------------------
* Still works with the existing Flask server (no server‑side change required).
* Keeps the simple REST endpoints for pose / single‑shot force / torque.
* Adds **high‑rate force streaming** via an internal polling thread (default 100 Hz).

Typical usage
-------------
```python
from franka_api import FrankaAPI

api = FrankaAPI("172.27.190.125", rest_port=5000)
api.start_force_stream(rate_hz=100)       # begin high‑rate polling (optional)

pose  = api.get_pose_sync()               # blocking REST call
f_now = api.get_force()                   # latest sample from ring buffer (None until buffer filled)
f_win = api.get_force_window(32)          # (32,6) recent window, zero‑padded if not full

# ... when finished
api.stop()
```
"""

from __future__ import annotations

import threading
import time
import socket
from collections import deque
from typing import Optional, Dict, Any

import numpy as np
import requests

__all__ = ["FrankaAPI"]


class FrankaAPI:
    """Client‑side helper for the existing Flask control server.

    Features
    --------
    1. **REST (blocking)** calls remain unchanged – good for sporadic info (pose, torque, etc.).
    2. **Background polling thread** repeatedly queries `/getforce` at up to ≈100 Hz and stores
       the results in an internal *ring buffer* so camera / policy loops can read the latest window
       without the overhead of an HTTP round‑trip each time.
    3. Thread can be started or stopped on demand. If not started, behaviour is identical to the
       original synchronous version.
    """

    # ---------- construction -------------------------------------------------
    def __init__(
        self,
        server_ip: str,
        *,
        rest_port: int = 5000,
        timeout: float = 1.0,
        buf_len: int = 128,
    ) -> None:
        self.base_url = f"http://{server_ip}:{rest_port}"
        self.timeout = timeout

        # keep‑alive session to reuse TCP connection
        self._session = requests.Session()

        # ring buffer for force samples (6‑D float32)
        self._buf: deque[np.ndarray] = deque(maxlen=buf_len)
        self._lock = threading.Lock()

        # background polling
        self._poll_th: threading.Thread | None = None
        self._stop_evt = threading.Event()

    # ---------- low‑level HTTP helper ---------------------------------------
    def _post_json(self, endpoint: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any] | None:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            resp = self._session.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json() if resp.text else None
        except requests.RequestException as exc:
            print(f"[FrankaAPI] HTTP ERR → {url}: {exc}")
            return None

    # ---------- synchronous REST wrappers -----------------------------------
    def get_pose_sync(self) -> Optional[np.ndarray]:
        data = self._post_json("getpos")
        return np.asarray(data["pose"], np.float32) if data else None

    def get_force_sync(self) -> Optional[np.ndarray]:
        data = self._post_json("getforce")
        return np.asarray(data["force"], np.float32) if data else None

    def get_torque_sync(self) -> Optional[np.ndarray]:
        data = self._post_json("gettorque")
        return np.asarray(data["torque"], np.float32) if data else None

    def get_gripper_sync(self) -> Optional[np.ndarray]:
        data = self._post_json("get_gripper")    
        return np.asarray(data["gripper"], np.float32) if data else None

    def set_gripper_sync(self, position: float) -> bool:
        """Control gripper position. position: 0.0 = closed, 1.0 = open"""
        data = self._post_json("set_gripper", {"position": position})
        return data is not None and data.get("success", False)

    def close_gripper_sync(self) -> bool:
        """Close gripper"""
        return self.set_gripper_sync(0.0)

    def open_gripper_sync(self) -> bool:
        """Open gripper"""
        return self.set_gripper_sync(1.0)

    # ---------- high‑rate polling thread ------------------------------------
    def _poll_loop(self, rate_hz: float) -> None:
        period = 1.0 / rate_hz
        next_t = time.perf_counter()
        while not self._stop_evt.is_set():
            sample = self.get_force_sync()
            if sample is not None:
                with self._lock:
                    self._buf.append(sample)
            next_t += period
            time.sleep(max(0.0, next_t - time.perf_counter()))

    def start_force_stream(self, rate_hz: float = 100.0) -> None:
        """Start background polling at *rate_hz*. Repeat calls safely restart the stream."""
        self.stop()  # ensure previous thread (if any) is closed
        self._stop_evt.clear()
        self._poll_th = threading.Thread(target=self._poll_loop, args=(rate_hz,), daemon=True)
        self._poll_th.start()
        print(f"[FrankaAPI] force stream started ({rate_hz} Hz polling)")

    # ---------- ring buffer read helpers ------------------------------------
    def get_force(self) -> Optional[np.ndarray]:
        """Latest force sample from buffer (None if buffer empty)."""
        with self._lock:
            return None if not self._buf else self._buf[-1].copy()

    def get_force_window(self, length: int) -> np.ndarray:
        """Latest *length* samples, padded with zeros if insufficient."""
        with self._lock:
            data = list(self._buf)[-length:]
        if len(data) < length:
            pad = [np.zeros(6, np.float32)] * (length - len(data))
            data = pad + data
        return np.vstack(data)

    # ---------- graceful shutdown ------------------------------------------
    def stop(self) -> None:
        if self._poll_th and self._poll_th.is_alive():
            self._stop_evt.set()
            self._poll_th.join(timeout=1.0)
            print("[FrankaAPI] force stream stopped")
        self._poll_th = None

    # ---------- context‑manager sugar ---------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        self._session.close()
