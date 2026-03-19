"""GUI thread, camera reading, and keyboard input utilities for eval_robot."""

import sys
import time
import select
import queue
import threading
from typing import Tuple

import cv2
import numpy as np

from utils.eval_display import make_display_image


def read_two_cams(camera, additional_cam) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read from both cameras, return (rgb1, rgb2, bgr1, bgr2).
    Legacy function — kept for compatibility. Prefer camera.get_latest_rgb().
    """
    ok1, res1 = camera.read()
    ok2, res2 = additional_cam.read()
    if not ok1:
        raise RuntimeError("Main camera read failed")
    f1 = res1[0]
    f2 = res2[0] if ok2 else np.zeros_like(f1)
    return cv2.cvtColor(f1, cv2.COLOR_BGR2RGB), cv2.cvtColor(f2, cv2.COLOR_BGR2RGB), f1, f2


class GuiThread:
    """Worker thread: 카메라 링 버퍼에서 ~30Hz로 읽어 cv2.imshow로 표시.
    headless=False일 때만 사용. 키 입력을 key_queue로 전달.
    """

    def __init__(self, camera, additional_cam, cam_lock, key_queue,
                 state, config, ft_collector=None):
        self.camera = camera
        self.additional_cam = additional_cam
        self.cam_lock = cam_lock
        self.key_queue = key_queue
        self.state = state
        self.config = config
        self.ft_collector = ft_collector
        self._stop = threading.Event()
        self._th = None
        self._latest_bgr1 = None
        self._latest_bgr2 = None
        self._frame_lock = threading.Lock()

    def start(self):
        if self._th is not None and self._th.is_alive():
            return
        self._stop.clear()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def stop(self):
        self._stop.set()
        if self._th is not None:
            self._th.join(timeout=2.0)

    def _loop(self):
        while not self._stop.is_set():
            try:
                # Read from camera ring buffers — no cam_lock needed
                latest1 = self.camera.get_latest_rgb()
                latest2 = self.additional_cam.get_latest_rgb()
                if latest1 is None or latest2 is None:
                    time.sleep(0.01)
                    continue
                f1_rgb, f2_rgb = latest1[1], latest2[1]
                fb1 = cv2.cvtColor(f1_rgb, cv2.COLOR_RGB2BGR)
                fb2 = cv2.cvtColor(f2_rgb, cv2.COLOR_RGB2BGR)
                with self._frame_lock:
                    self._latest_bgr1 = fb1
                    self._latest_bgr2 = fb2
                disp = make_display_image(fb1, fb2, self.state.current_mode,
                                          state=self.state, config=self.config,
                                          ft_data=self.state.ft_latest,
                                          ft_collector=self.ft_collector)
                cv2.imshow('Camera 1 | Camera 2', disp)
                self.state.video_recorder.write_frame(disp)
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    mapping = {ord(c): c for c in 'qtpr120sfeo'}
                    ch = mapping.get(key)
                    if ch:
                        self.key_queue.put(ch)
            except Exception as e:
                print(f"\r[GUI] error: {e}", end='', flush=True)
            time.sleep(0.01)

    def get_latest_bgr(self):
        with self._frame_lock:
            return self._latest_bgr1, self._latest_bgr2


def check_keyboard_input(state, key_queue):
    """키 입력 확인: headless면 터미널, 아니면 GUI thread의 key queue에서."""
    if state.headless:
        if select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            if ch in 'qtpr120sfeo':
                return ch
        return None
    try:
        return key_queue.get_nowait()
    except queue.Empty:
        return None
