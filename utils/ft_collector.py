"""FT sensor data collector with background thread and gravity compensation."""
import time
import threading
import numpy as np
from collections import deque


class FTCollector:
    """Collect FT from Aidin UDP sensor with IMU gravity compensation in background thread."""

    def __init__(self, ft_reader, imu_source, gravity_compensator, rate_hz=200, buf_len=200):
        self.ft_reader = ft_reader
        self.imu_source = imu_source
        self.gravity_compensator = gravity_compensator
        self.rate_hz = rate_hz
        self._th = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.buf = deque(maxlen=buf_len)
        self.ts_buf = deque(maxlen=buf_len)
        # Graph display buffer (~10s at 200Hz)
        self.graph_buf = deque(maxlen=2000)
        self.graph_ts_buf = deque(maxlen=2000)
        self.full_ts_list = []
        self.full_ft_list = []
        self.bias_initialized = True  # 학습과 동일: 추가 대기 없이 바로 수집

    def _loop(self):
        period = 1.0 / self.rate_hz
        nxt = time.perf_counter()
        while not self._stop.is_set():
            try:
                self.gravity_compensator.update_imu(self.imu_source)
                try:
                    ts_raw, f_raw, t_raw = self.ft_reader.get_frame(timeout=0.001)
                except Exception:
                    ts_raw, f_raw, t_raw = None, None, None

                if f_raw is not None and t_raw is not None:
                    forces_filt, torques_filt = self.gravity_compensator.process_ft_data(f_raw, t_raw)
                    comp_f, comp_t = self.gravity_compensator.compensate_gravity(
                        forces_filt, torques_filt, gravity_compensation_on=True)

                    ft_vec = np.concatenate([comp_f, comp_t]).astype(np.float32)
                    ts = time.time()
                    with self._lock:
                        self.buf.append(ft_vec)
                        self.ts_buf.append(ts)
                        self.graph_buf.append(ft_vec)
                        self.graph_ts_buf.append(ts)
                        self.full_ts_list.append(ts)
                        self.full_ft_list.append(ft_vec)
            except Exception as e:
                print(f"\rFT data collection error: {e}", end='', flush=True)
            nxt += period
            time.sleep(max(0, nxt - time.perf_counter()))

    def start(self):
        if self._th is None or not self._th.is_alive():
            self._stop.clear()
            self._th = threading.Thread(target=self._loop, daemon=True)
            self._th.start()

    def stop(self):
        self._stop.set()
        if self._th:
            self._th.join(timeout=1)

    def window(self, length: int):
        with self._lock:
            ft_data = list(self.buf)[-length:]
            ts_data = list(self.ts_buf)[-length:]
        if len(ft_data) < length:
            pad_n = length - len(ft_data)
            if ft_data:
                ft_data = [ft_data[-1]] * pad_n + ft_data
                ts_data = [ts_data[-1]] * pad_n + ts_data
            else:
                ft_data = [np.zeros(6, dtype=np.float32)] * pad_n
                ts_data = [0.0] * pad_n
        return np.array(ft_data, dtype=np.float32), np.array(ts_data, dtype=np.float64)

    def graph_data(self):
        """Thread-safe access to graph display buffer."""
        with self._lock:
            return list(self.graph_buf), list(self.graph_ts_buf)
