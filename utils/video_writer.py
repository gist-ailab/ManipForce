"""Video recording helper with lazy initialization and real-time frame timing."""
import os
import time
import threading
import cv2
from datetime import datetime


class VideoWriter:
    """Manages video recording: lazy init, real-time frame pacing, start/stop/delete."""

    def __init__(self, fps=30.0):
        self.fps = fps
        self._writer = None     # None | str (path, lazy) | cv2.VideoWriter
        self._path = None
        self._last_time = None
        self._lock = threading.Lock()
        self._min_interval = 0.5 / fps  # skip frames closer than half a frame period

    @property
    def is_recording(self):
        return self._writer is not None

    @property
    def path(self):
        return self._path

    def start(self, output_dir, prefix=''):
        """Start a new recording. Actual VideoWriter created on first frame."""
        self.stop_and_save()
        os.makedirs(output_dir, exist_ok=True)
        ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        vid_path = os.path.join(output_dir, f"{prefix}_{ts_str}.mp4")
        print(f"\n[녹화] 영상 시작: {vid_path}")
        with self._lock:
            self._writer = vid_path  # lazy init
            self._path = vid_path
            self._last_time = None

    def write_frame(self, frame):
        """Write a frame with real-time pacing (thread-safe)."""
        with self._lock:
            if self._writer is None:
                return
            # Lazy init: first frame creates the actual writer
            if isinstance(self._writer, str):
                self._lazy_init(frame)
            if not isinstance(self._writer, cv2.VideoWriter):
                return
            now_t = time.time()
            if self._last_time is not None:
                elapsed = now_t - self._last_time
                if elapsed < self._min_interval:
                    return  # too soon, skip to avoid duplicate pts
            fh, fw = frame.shape[:2]
            nw = fw if fw % 2 == 0 else fw + 1
            nh = fh if fh % 2 == 0 else fh + 1
            if nw != fw or nh != fh:
                frame = cv2.copyMakeBorder(frame, 0, nh - fh, 0, nw - fw,
                                           cv2.BORDER_CONSTANT, value=[0, 0, 0])
            if self._last_time is None:
                n_frames = 1
            else:
                elapsed = now_t - self._last_time
                n_frames = max(1, int(round(elapsed * self.fps)))
            for _ in range(n_frames):
                self._writer.write(frame)
            self._last_time = now_t

    def stop_and_save(self):
        """Release writer (keep file)."""
        with self._lock:
            if self._writer is not None:
                if isinstance(self._writer, cv2.VideoWriter):
                    self._writer.release()
                self._writer = None
                print("[녹화] 영상 저장 완료")

    def stop_and_delete(self):
        """Release writer and delete file."""
        with self._lock:
            if self._writer is not None:
                if isinstance(self._writer, cv2.VideoWriter):
                    self._writer.release()
                self._writer = None
        if self._path and os.path.exists(self._path):
            os.remove(self._path)
            print(f"[녹화] 영상 삭제: {self._path}")
        self._path = None

    def _lazy_init(self, frame):
        """First frame: convert string path to actual VideoWriter."""
        vid_path_str = self._writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fh, fw = frame.shape[:2]
        fw = fw if fw % 2 == 0 else fw + 1
        fh = fh if fh % 2 == 0 else fh + 1
        self._writer = cv2.VideoWriter(vid_path_str, fourcc, self.fps, (fw, fh))
        self._last_time = time.time()
        if not self._writer.isOpened():
            print(f"[WARNING] VideoWriter 열기 실패: {vid_path_str}")
            self._writer = None
        else:
            print(f"[녹화] VideoWriter 초기화 완료 ({fw}x{fh} @ {self.fps}fps)")
