
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import time
import json
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import scipy.spatial.transform as st
from scipy.spatial.transform import Rotation
import scipy.signal
import quaternion
from typing import Tuple
import yaml
import os

import socket
import sys
import termios
import tty
import select
import requests

import torch
import dill
import hydra.utils
from omegaconf import OmegaConf, DictConfig, ListConfig

from diffusion_policy.real_world.spacemouse_shared_memory import Spacemouse
from diffusion_policy.common.replay_buffer import ReplayBuffer

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from utils.precise_sleep import precise_wait

from diffusion_policy.common.pytorch_util import dict_apply
from utils.real_inference_util import (get_real_obs_resolution,
                                       get_real_gumi_obs_dict,
                                       get_real_gumi_action,
                                       convert_action_10d_to_8d)

from utils.rs_capture import RSCapture
from utils.ft_capture import AidinFTSensorUDP
from utils.gravity_compensation_utils import GravityCompensator
import pyrealsense2 as rs
from PIL import Image
from utils.franka_api import FrankaAPI
from collections import deque
import threading, time, numpy as np

# Load configuration
def load_config(config_path='inference_config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# Global config variable
config = None

# Global buffers
prev_frames_cam1, prev_frames_cam2, prev_timestamps = [], [], []

# Globals for rate display
_last_disp_t = time.time()
_disp_cnt    = 0
_img_accum   = 0.0
_ft_accum    = 0.0

# Image buffers — initialized after config is loaded
prev_frames_cam1, prev_frames_cam2 = None, None
prev_timestamps = None

# Action history buffer
action_history = deque(maxlen=4)
USE_ACTION_HISTORY = False  # True: use action history, False: disabled

# Gripper state tracking globals — initialized after config is loaded
gripper_history = None
current_gripper_state = 'open'
current_gripper_target = 'open'

def close_gripper_http(server_ip, port):
    """Close gripper via HTTP API."""
    try:
        response = requests.post(f"http://{server_ip}:{port}/close_gripper", timeout=1.0)
        return response.status_code == 200
    except Exception as e:
        return False

def open_gripper_http(server_ip, port):
    """Open gripper via HTTP API."""
    try:
        response = requests.post(f"http://{server_ip}:{port}/open_gripper", timeout=1.0)
        return response.status_code == 200
    except Exception as e:
        return False

def smart_gripper_control(predicted_gripper, franka_api, server_ip):
    """
    History-based gripper control logic.
    1.0 = open, 0.0 = close
    - Maintains a history of the last N predictions.
    - Close: 2 or more out of the last N predictions signal close (predicted < 0.1).
    - Open:  3 or more out of the last N predictions signal open  (predicted >= 0.1).
    - If already in the desired state, hold (keep).
    """
    global current_gripper_state, current_gripper_target, gripper_history
    
    # Query actual gripper state
    try:
        actual_gripper = franka_api.get_gripper_sync()
    except Exception as e:
        actual_gripper = 0.025  # default (mid-state)
    
    # Append prediction to history
    gripper_history.append(predicted_gripper)

    # Not enough history yet
    if len(gripper_history) < config['gripper']['history_length']:
        return 'keep'
    
    # Analyse the most recent N predictions
    hist_len = config['gripper']['history_length']
    recent_predictions = list(gripper_history)[-hist_len:]
    close_count = sum(1 for pred in recent_predictions if pred < config['gripper']['close_threshold'])
    open_count = sum(1 for pred in recent_predictions if pred >= config['gripper']['open_threshold'])
    
    # Determine physical state
    is_physically_closed = (actual_gripper < 0.045)
    is_physically_open   = (actual_gripper >= 0.045)
    
    # Infer intent from history
    wants_close = (close_count >= config['gripper']['close_count_threshold'])
    wants_open  = (open_count  >= config['gripper']['open_count_threshold'])
    
    # Control logic
    if wants_close:
        # Already closed — hold
        if is_physically_closed:
            current_gripper_target = 'close'
            return 'keep'
        # Not yet closed — send close command once
        elif current_gripper_target != 'close':
            success = close_gripper_http(server_ip, config['gripper']['http_port'])
            if success:
                current_gripper_target = 'close'
                time.sleep(0.1)
            return 'close'
        else:
            return 'close'
            
    elif wants_open:
        # Already open — hold
        if is_physically_open:
            current_gripper_target = 'open'
            return 'keep'
        # Not yet open — send open command once
        elif current_gripper_target != 'open':
            success = open_gripper_http(server_ip, config['gripper']['http_port'])
            if success:
                current_gripper_target = 'open'
                time.sleep(0.1)
            return 'open'
        else:
            return 'open'
            
    else:
        # No clear intent — hold current state
        return 'keep'

def _read_two_cams() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ok1, res1 = camera.read()
    ok2, res2 = additional_cam.read()
    if not ok1:
        raise RuntimeError("Main camera read failed")
        
    f1 = res1[0]  # (frame, display, acq_time) -> frame
    if not ok2:
        f2 = np.zeros_like(f1)
    else:
        f2 = res2[0]
    
    # BGR → RGB (match training data format)
    f1_rgb = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
    f2_rgb = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)
    
    return f1_rgb, f2_rgb, f1, f2  # RGB (for model), BGR (for visualisation)
class FTCollector:
    """Collect FT from Aidin UDP sensor and apply IMU-based gravity compensation in a background thread."""
    def __init__(self, ft_reader, imu_pipe, gravity_compensator, rate_hz=200, buf_len=200):
        self.ft_reader = ft_reader
        self.imu_pipe = imu_pipe
        self.gravity_compensator = gravity_compensator
        self.rate_hz = rate_hz
        self.dt = 1.0 / max(1, rate_hz)

        # threading
        self._th = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        # ring buffers
        self.buf = deque(maxlen=buf_len)
        self.ts_buf = deque(maxlen=buf_len)

        # full logs (optional)
        self.full_ts_list = []
        self.full_ft_list = []
        
        # Variables for initial bias removal
        self.f_bias_initial = None
        self.t_bias_initial = None
        self.bias_initialized = False

    def _loop(self):
        period = 1.0 / self.rate_hz
        nxt = time.perf_counter()

        while not self._stop.is_set():
            try:
                # Update IMU first for the current cycle
                self.gravity_compensator.update_imu(self.imu_pipe)

                # Read one FT frame (non-blocking-ish)
                try:
                    ts_raw, f_raw, t_raw = self.ft_reader.get_frame(timeout=0.001)
                except Exception:
                    ts_raw, f_raw, t_raw = None, None, None

                if f_raw is not None and t_raw is not None:
                    # Filtering and gravity compensation
                    forces_filt, torques_filt = self.gravity_compensator.process_ft_data(f_raw, t_raw)
                    compensated_force, compensated_torque = self.gravity_compensator.compensate_gravity(
                        forces_filt, torques_filt, gravity_compensation_on=True
                    )
                    
                    # Remove initial bias
                    if not self.bias_initialized:
                        self.f_bias_initial = compensated_force.copy()
                        self.t_bias_initial = compensated_torque.copy()
                        self.bias_initialized = True
                        print(f"FT initial bias set: Force={self.f_bias_initial}, Torque={self.t_bias_initial}")
                    
                    # Subtract initial bias
                    f_final = compensated_force - self.f_bias_initial
                    # Additional fy correction (empirical offset)
                    f_final[1] -= 3.0
                    t_final = compensated_torque - self.t_bias_initial
                    
                    ft_vec = np.concatenate([f_final, t_final]).astype(np.float32)
                    ts = time.time()

                    with self._lock:
                        self.buf.append(ft_vec)
                        self.ts_buf.append(ts)
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
        """Return the most recent 'length' FT samples with padding as needed."""
        with self._lock:
            ft_data = list(self.buf)[-length:]
            ts_data = list(self.ts_buf)[-length:]

        if len(ft_data) < length:
            pad_n = length - len(ft_data)
            if len(ft_data) > 0:
                last_frame = ft_data[-1]
                last_ts = ts_data[-1]
                ft_data = [last_frame] * pad_n + ft_data
                ts_data = [last_ts] * pad_n + ts_data
            else:
                zero = np.zeros(6, dtype=np.float32)
                ft_data = [zero] * pad_n
                ts_data = [0.0] * pad_n

        return np.array(ft_data, dtype=np.float32), np.array(ts_data, dtype=np.float64)

def display_rates(img_freq: float, ft_ts: np.ndarray):
    """
    img_freq : FPS of the most recent frame
    ft_ts    : timestamp array from FTCollector for the current get_obs call
    """
    global _last_disp_t, _disp_cnt, _img_accum, _ft_accum

    _disp_cnt   += 1
    _img_accum  += img_freq
    if len(ft_ts) >= 2:
        time_diffs = np.diff(ft_ts[-8:])
        if np.all(time_diffs > 0):  # verify timestamps are increasing
            ft_hz = 1.0 / np.mean(time_diffs)  # estimate FT Hz from last 8 samples
            _ft_accum += ft_hz
        else:
            ft_hz = 0.0
    else:
        ft_hz = 0.0

    now = time.time()
    if now - _last_disp_t >= 1.0:  # print average every second
        avg_img = _img_accum / _disp_cnt
        avg_ft  = _ft_accum  / _disp_cnt
        sys.stdout.write(
            f"\r[Rate] IMG {avg_img:5.1f} FPS | FT {avg_ft:5.1f} Hz")
        sys.stdout.flush()

        # Reset accumulators
        _last_disp_t = now
        _disp_cnt = _img_accum = _ft_accum = 0.0

def quaternion_multiply(q1, q2):
    """Hamilton product, quaternion in [w, x, y, z] order."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])

def transform_pose(pose, current_ee_pose, ft_data):
    R_mat_pos = np.array(config['coordinate_transform']['R_mat_pos'])
    R_mat_rot = np.array(config['coordinate_transform']['R_mat_rot'])
    # Per-axis rotation scale — use individual keys if present, otherwise fall back to xy_scale
    if 'rotation_x_scale' in config['action'] and 'rotation_y_scale' in config['action']:
        rotation_x_scale = config['action']['rotation_x_scale']
        rotation_y_scale = config['action']['rotation_y_scale']
    else:
        rotation_xy_scale = config['action']['rotation_xy_scale']
        rotation_x_scale = rotation_xy_scale
        rotation_y_scale = rotation_xy_scale
    
    rotation_z_scale = config['action']['rotation_z_scale']
    
    rel_pos = R_mat_pos @ pose[:3]
    # Per-axis position scale — use individual keys if present, otherwise fall back to xy_scale
    if 'position_x_scale' in config['action'] and 'position_y_scale' in config['action']:
        position_x_scale = config['action']['position_x_scale']
        position_y_scale = config['action']['position_y_scale']
    else:
        position_xy_scale = config['action']['position_xy_scale']
        position_x_scale = position_xy_scale
        position_y_scale = position_xy_scale
    
    position_z_scale = config['action']['position_z_scale']
    rel_pos_x = rel_pos[0] * position_x_scale
    rel_pos_y = rel_pos[1] * position_y_scale
    rel_pos_z = rel_pos[2] * position_z_scale
    rel_pos = [rel_pos_x, rel_pos_y, rel_pos_z]
    
    # Rotation handling — apply per-axis scale
    delta_quat = np.array(pose[3:7])  # [x, y, z, w]
    
    # Convert quaternion to rotation vector for per-axis scaling
    R_orig = Rotation.from_quat(delta_quat)
    rotvec_orig = R_orig.as_rotvec()
        
    # Apply per-axis scale
    rotvec_scaled = rotvec_orig.copy()
    rotvec_scaled[0] *= rotation_x_scale  # x-axis
    rotvec_scaled[1] *= rotation_y_scale  # y-axis
    rotvec_scaled[2] *= rotation_z_scale  # z-axis
    
    # Convert back to rotation matrix
    R_scaled = Rotation.from_rotvec(rotvec_scaled).as_matrix()
    R_new = R_mat_rot @ R_scaled @ R_mat_rot.T
    rel_quat = Rotation.from_matrix(R_new).as_quat()
    
    # Apply current EE orientation
    current_quat = current_ee_pose[3:]   # [w, x, y, z]
    current_quat = np.roll(current_quat, -1)  # → [x, y, z, w]
    current_rot = Rotation.from_quat(current_quat)
    world_rel_pos = current_rot.inv().apply(rel_pos)
    
    return np.concatenate([world_rel_pos, rel_quat])

def nudge_y_if_stuck(current_ee_pose: np.ndarray,
                     last_target_pose: np.ndarray,
                     pos_thresh: float = 1e-4,
                     nudge_amount: float = 0.002) -> Tuple[np.ndarray, bool]:
    """
    When the EE pose barely differs from the previous target (robot is stuck),
    apply a small y-axis perturbation to break the deadlock.

    returns: (nudged_target_pose, did_nudge)
    """
    try:
        pos_delta = np.linalg.norm((current_ee_pose[:3] - last_target_pose[:3]).astype(np.float64))
    except Exception:
        pos_delta = np.linalg.norm(current_ee_pose[:3] - last_target_pose[:3])

    if pos_delta < pos_thresh:
        nudged = last_target_pose.copy()
        nudged[1] += nudge_amount * _y_nudge_dir  # small y-axis perturbation
        _y_nudge_dir *= -1  # toggle direction to minimise drift
        return nudged, True
    return last_target_pose, False

def connect_to_server(host='localhost', port=4999, timeout=5):
    while True:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        client_socket.settimeout(timeout)
        try:
            client_socket.connect((host, port))
            return client_socket
        except (ConnectionRefusedError, socket.timeout) as e:
            print(f"Failed to connect to {host}:{port} ({e}). Retrying in 1s...")
            time.sleep(1)
            continue

def check_keyboard_input():
    """Non-blocking keyboard input check via OpenCV window."""
    key = cv2.waitKey(1) & 0xFF
    
    if key != 255:  # 키가 눌렸으면
        if key == ord('q'):
            return 'q'
        elif key == ord('t'):
            return 't'
        elif key == ord('p'):
            return 'p'
        elif key == ord('r'):
            return 'r'
        elif key == ord('1'):
            return '1'
        elif key == ord('2'):
            return '2'
        elif key == ord('0'):
            return '0'
    
    return None

def resize_with_padding(img, target_size=224):
    """
    Resize an image to a square while preserving aspect ratio by padding.
    img: (H, W, C) uint8
    return: (target_size, target_size, C) uint8
    """
    h, w = img.shape[:2]
    aspect = w / h
    
    if aspect > 1:  # landscape (e.g. 1280×800)
        new_w = target_size
        new_h = int(target_size / aspect)
    else:  # portrait (e.g. 640×480)
        new_h = target_size
        new_w = int(target_size * aspect)
    
    pil_img = Image.fromarray(img)
    resized = pil_img.resize((new_w, new_h))
    
    # Pad to square
    new_img = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    new_img.paste(resized, (paste_x, paste_y))
    
    return np.array(new_img)

def get_obs(ft_collector, last_executed_action=None):
    TARGET_HW = tuple(config['image']['target_resolution'])  # observation 크기 유지
    
    # ------ 0. 버퍼 초기화 (프로그램 첫 2프레임) ------
    while len(prev_frames_cam1) < config['image']['history_length']:
        f1_rgb, f2_rgb, f1_bgr, f2_bgr = _read_two_cams()
        # 데이터로더와 동일하게, 리사이즈는 이후 단계에서 수행
        prev_frames_cam1.append(f1_rgb)
        prev_frames_cam2.append(f2_rgb)
        prev_timestamps.append(time.time())
        time.sleep(1/30)

    # ------ 1. 새 카메라 프레임 ------
    now_img = time.time()
    f1_rgb, f2_rgb, f1_bgr, f2_bgr = _read_two_cams()
    # 데이터로더와 동일하게, 원본 RGB를 버퍼에 저장 (리사이즈는 이후)
    prev_frames_cam1.append(f1_rgb)
    prev_frames_cam2.append(f2_rgb)
    prev_timestamps.append(now_img)

    # ------ 2. sampler.py와 동일한 FT 타임스탬프 매핑 ------
    ft_buf, ts_buf = ft_collector.window(config['ft_sensor']['buffer_length'])
    # prev_timestamps에서 최근 history만 꺼내고, 그중 "두 프레임" 구간만 사용
    img_ts_hist = np.array(list(prev_timestamps)[-config['image']['history_length']:], dtype=np.float64)

    # 최근 2프레임 혹은 down_sample_steps 간격을 반영
    if img_ts_hist.size >= 2:
        ds = int(config['image'].get('down_sample_steps', 1))
        if img_ts_hist.size >= ds + 1:
            t0 = img_ts_hist[-(ds + 1)]   # 이전 프레임 (downsample 간격)
            t1 = img_ts_hist[-1]          # 최신 프레임
        else:
            # history가 충분치 않으면 그냥 최근 2프레임
            t0 = img_ts_hist[-2]
            t1 = img_ts_hist[-1]
    elif img_ts_hist.size == 1:
        # 프레임이 1개뿐이면 아주 짧은 구간으로 처리
        t0 = img_ts_hist[0]
        t1 = img_ts_hist[0] + 1e-3
    else:
        # 타임스탬프가 전혀 없으면 현재 시각 기준으로 최소 구간
        now = time.time()
        t0, t1 = now - 1e-3, now

    # sampler.py와 동일한 타임스탬프 변환 (마이크로초 단위)
    img_ts_int = np.array([int(t0 * 1e6), int(t1 * 1e6)], dtype=np.int64)
    ft_ts_int  = np.array([int(ts * 1e6) for ts in ts_buf], dtype=np.int64)

    # [t0, t1] 사이의 FT 슬라이스만 취함
    left_idx  = np.searchsorted(ft_ts_int, img_ts_int[0], side='left')
    right_idx = np.searchsorted(ft_ts_int, img_ts_int[1], side='right')
    left_idx  = max(0, left_idx)
    right_idx = min(len(ft_ts_int), right_idx)

    ft_slice    = ft_buf[left_idx:right_idx]
    ft_ts_slice = ts_buf[left_idx:right_idx]
    n = len(ft_slice)

    # ------ sampler.py와 동일한 패딩/리샘플링 로직 ------
    # FT 프레임 수 계산 (sampler.py와 동일)
    img_hz = config['camera']['main_camera']['fps']  # 30Hz
    ft_hz = config['ft_sensor']['rate_hz']  # 200Hz
    target_ft_frames = int(round(ft_hz / img_hz)) + 1  # round(200/30) + 1 = 7 + 1 = 8
    target_ft_frames = max(3, min(20, target_ft_frames))  # 3~20 범위로 제한
    
    # config의 obs_horizon 값 사용 (8)
    target_ft_frames = config['ft_sensor']['obs_horizon']
    
    try:
        # 1) 너무 짧으면 마지막 프레임을 반복해서 패딩
        if n < target_ft_frames:
            if n == 0:
                # 완전히 비어있는 경우: 영벡터로 target_ft_frames개 프레임 생성
                ft_sel = np.zeros((target_ft_frames, 6), dtype=np.float32)
                ts_sel = np.zeros(target_ft_frames, dtype=np.float64)
            else:
                # 일부 데이터는 있는 경우: 마지막 프레임으로 패딩
                pad_n = target_ft_frames - n
                last_frame = ft_slice[-1][None, ...]  # shape (1,6)
                last_ts = ft_ts_slice[-1][None]       # shape (1,)
                ft_sel = np.concatenate([ft_slice, np.repeat(last_frame, pad_n, axis=0)], axis=0)
                ts_sel = np.concatenate([ft_ts_slice, np.repeat(last_ts, pad_n)], axis=0)
        elif n > target_ft_frames:
            # 균등 샘플링: indices 0, 1/3, 2/3, 1.0 구간에 대응
            idxs = np.linspace(0, n-1, target_ft_frames).round().astype(int)
            ft_sel = ft_slice[idxs]
            ts_sel = ft_ts_slice[idxs]
        else:
            # 정확히 맞는 경우
            ft_sel = ft_slice
            ts_sel = ft_ts_slice
            
        assert ft_sel.shape[0] == target_ft_frames, f"FT shape mismatch: {ft_sel.shape[0]} != {target_ft_frames}"
            
    except (ValueError, TypeError) as e:
        # 타임스탬프 변환 오류 시 기본값 사용
        print(f"FT 타임스탬프 매핑 오류: {e}, 기본값 사용")
        ft_sel = np.zeros((target_ft_frames, 6), dtype=np.float32)
        ts_sel = np.zeros(target_ft_frames, dtype=np.float64)

    # ------ 3. obs dict ------
    # FT 데이터에 축 변환 및 토크 스케일링 적용
    ft_sel_converted = convert_torque_scale(ft_sel)
    obs = build_obs_np(f1_rgb, f2_rgb, ft_sel_converted, ts_sel, last_executed_action)
    
    return obs, f1_rgb, f2_rgb, ft_sel, f1_bgr, f2_bgr, ts_sel

def convert_torque_scale(ft):
    """Robot FT 축을 GUMI 축으로 변환 (실제 측정 기반)"""
    ft = np.asarray(ft)
    
    # 거리 비율에 따른 토크 스케일링 (GUMI: 12cm, Panda: 18cm)
    torque_scale = 12.0 / 18.0  # 0.667
    
    if ft.ndim == 1:
        # 1차원 배열인 경우
        # 토크 스케일링 적용
        ft[3] *= torque_scale  # Tx
        ft[4] *= torque_scale  # Ty  
        ft[5] *= torque_scale  # Tz
        
        return np.array([ft[0], ft[1], ft[2], ft[3], ft[4], ft[5]], dtype=ft.dtype)
    else:
        # 2차원 배열인 경우
        # 토크 스케일링 적용
        ft[:,3] *= torque_scale  # Tx
        ft[:,4] *= torque_scale  # Ty  
        ft[:,5] *= torque_scale  # Tz
        
        # Force와 Torque 모두 같은 변환 적용
        return np.stack([ft[:,0], ft[:,1], ft[:,2], ft[:,3], ft[:,4], ft[:,5]], axis=-1)
# FT 데이터 정규화 후 사용
def build_obs_np(img1: np.ndarray,
                 img2: np.ndarray,
                 ft_sel: np.ndarray,
                 ft_timestamps: np.ndarray = None,  # FT 타임스탬프 추가
                 last_executed_action: np.ndarray = None  # 이전 스텝에서 실행한 action
                 ) -> dict:
    
    global action_history

    """
    img1, img2 : (224,224,3) uint8 0-255
    ft_sel     : (N,6) float32 - 이미 정규화된 FT 데이터
    ft_timestamps : (N,) float64 - FT 타임스탬프
    last_executed_action : (8,) float32 - 이전 스텝에서 실제 실행한 action
    """
    # 데이터 로더와 완전히 동일한 처리
    # 1. 이미지를 0~1로 정규화 (THWC)
    cam1 = np.stack(prev_frames_cam1, axis=0).astype(np.float32) / 255.0  # (T, H, W, C)
    cam2 = np.stack(prev_frames_cam2, axis=0).astype(np.float32) / 255.0  # (T, H, W, C)

    # 2. THWC → TCHW 변환
    cam1_tchw = np.moveaxis(cam1, -1, 1)  # (T, C, H, W)
    cam2_tchw = np.moveaxis(cam2, -1, 1)  # (T, C, H, W)

    # 3. 필요 시 리사이즈(패딩 포함) 적용 - 데이터로더와 동일 동작
    expected_shape = tuple(config['image']['target_resolution'])  # (H, W)처럼 쓰되, 아래에서 H 사용
    target_size = expected_shape[0]
    if cam1_tchw.shape[2] != target_size or cam1_tchw.shape[3] != target_size:
        resized = []
        for t in range(cam1_tchw.shape[0]):
            curr_img = np.moveaxis(cam1_tchw[t], 0, -1)  # CHW -> HWC
            curr_img_uint8 = (curr_img * 255).astype(np.uint8)
            resized_pil = resize_with_padding(curr_img_uint8, target_size=target_size)
            resized_np = resized_pil.astype(np.float32) / 255.0
            resized_np = np.moveaxis(resized_np, -1, 0)  # HWC -> CHW
            resized.append(resized_np)
        cam1_tchw = np.stack(resized, axis=0)
    if cam2_tchw.shape[2] != target_size or cam2_tchw.shape[3] != target_size:
        resized = []
        for t in range(cam2_tchw.shape[0]):
            curr_img = np.moveaxis(cam2_tchw[t], 0, -1)
            curr_img_uint8 = (curr_img * 255).astype(np.uint8)
            resized_pil = resize_with_padding(curr_img_uint8, target_size=target_size)
            resized_np = resized_pil.astype(np.float32) / 255.0
            resized_np = np.moveaxis(resized_np, -1, 0)
            resized.append(resized_np)
        cam2_tchw = np.stack(resized, axis=0)

    obs_np = {
        'handeye_cam_1': cam1_tchw,  # (T, C, H, W) - 데이터 로더와 동일
        'handeye_cam_2': cam2_tchw,  # (T, C, H, W) - 데이터 로더와 동일
        'ft_data': ft_sel,           # (N, 6) - 이미 정규화됨
    }
    
    # FT 타임스탬프가 있으면 추가
    if ft_timestamps is not None:
        obs_np['ft_timestamps'] = ft_timestamps.astype(np.float32)
    
    return obs_np

def save_ft_data(ft_collector, filename="ft_trace.csv"):
    """
    FTCollector.buf 안의 (timestamp, [fx,fy,fz,tx,ty,tz]) 전체를 CSV로 저장
    """
    # 1) collector 정지
    ft_collector.stop()

    # 2) 뮤텍트 잡고 버퍼 전체 복사
    with ft_collector._lock:
        data = list(ft_collector.buf)

    # 3) 빈 경우 처리
    if not data:
        return

    # 4) timestamp 와 force–torque 분리
    ts, ft = zip(*data)                     # ts: tuple of floats, ft: tuple of length-6 arrays
    arr_ts = np.array(ts, dtype=np.float64).reshape(-1, 1)   # (N,1)
    arr_ft = np.vstack(ft).astype(np.float32)               # (N,6)

    # 5) 합치기
    csv_data = np.hstack([arr_ts, arr_ft])  # (N,7)

    # 6) 헤더 정의
    header = "timestamp,fx,fy,fz,tx,ty,tz"

    # 7) 저장
    np.savetxt(filename,
               csv_data,
               delimiter=",",
               header=header,
               comments="")  # comments="" 로 '#' 제거

@click.command()
@click.option('--config_path', '-c', default='inference_config.yaml', type=str, help="설정 파일 경로")
@click.option('--model_checkpoint_path', '-mcp', default='', type=str, help="모델 체크포인트 경로")
def main(config_path, model_checkpoint_path):
    global config, prev_frames_cam1, prev_frames_cam2, prev_timestamps, gripper_history
    
    # Load configuration
    config = load_config(config_path)
    
    # Initialize buffers with config values
    prev_frames_cam1 = deque(maxlen=config['image']['history_length'])
    prev_frames_cam2 = deque(maxlen=config['image']['history_length'])
    prev_timestamps = deque(maxlen=config['image']['history_length'])
    gripper_history = deque(maxlen=config['gripper']['history_length'])
    
    # Extract values from config
    server_ip = config['robot']['server_ip']
    server_port = config['robot']['server_port']
    frequency = config['control']['frequency']
    output = config['paths']['output_video']
    
    franka_api = FrankaAPI(server_ip)

    # Initialize Aidin FT reader (UDP)
    ft_ip = config.get('ft_sensor', {}).get('ip', '172.27.190.4')
    ft_port = int(config.get('ft_sensor', {}).get('port', 8890))
    ft_reader = AidinFTSensorUDP(ft_ip, ft_port)
    ft_reader.start()

    # Gravity compensator (configurable, with sensible defaults)
    gravity_compensator = GravityCompensator(
        mass_for_x=config['gravity_compensator']['mass_for_x'],
        mass_for_y=config['gravity_compensator']['mass_for_y'],
        mass_for_z=config['gravity_compensator']['mass_for_z'],
        com_ft=np.array(config['gravity_compensator']['com_ft']),
        g_const=config['gravity_compensator']['g_const']
    )
    
    # === 현재 로봇 자세 출력 ===
    try:
        current_pose = franka_api.get_pose_sync()
        current_gripper = franka_api.get_gripper_sync()
    except Exception as e:
        pass
    
    # 시작시 로봇 초기 자세 출력
    print("\n=== 초기 로봇 자세 (5-4-3) ===")
    print(f"Initial pose: {[current_pose[0], current_pose[1], current_pose[2], current_pose[3], current_pose[4], current_pose[5], current_pose[6]]}")

    dt = 1 / frequency
    max_pos_speed = config['robot']['max_pos_speed']  # m/s
    max_rot_speed = config['robot']['max_rot_speed']  # rad/s
    
    # 사전 계산된 상수
    pos_scale = max_pos_speed * dt
    rot_scale = max_rot_speed * dt
    
    initial_pose = np.array(config['robot']['initial_pose'])

    ### ADD Random pose ###
    # initial pose에서 랜덤 자세를 추가 (position만)
    random_pos_range = config['robot']['random_pos_range']
    pos_noise = np.random.uniform(-random_pos_range, random_pos_range, size=3)
    
    cfg_basename = os.path.basename(config_path).lower() if isinstance(config_path, str) else ''
    
    
    # if battery_assemb task
    if 'battery_assemb' in cfg_basename:
        initial_pose[0] = initial_pose[0] + pos_noise[0]
        initial_pose[2] = initial_pose[2] + pos_noise[2]
    else:
        initial_pose[:3] = initial_pose[:3] + pos_noise 

    initial_tilt_deg = float(config['robot'].get('initial_tilt_deg', 0.0))
    
    # One-sided tilt only for jetson_flipping configs; otherwise symmetric
    if 'jetson_flipping' in cfg_basename:
        additional_y_angle = np.random.uniform(0, initial_tilt_deg)  # -20° to 0° additional
    else:
        additional_y_angle = np.random.uniform(10, initial_tilt_deg)  # ±20° additional
    additional_y_angle = initial_tilt_deg
    existing_quat = [1, 0, 0, 0]
    existing_rot = st.Rotation.from_quat(existing_quat)
    half_angle = np.deg2rad(additional_y_angle) / 2
    c, s = np.cos(half_angle), np.sin(half_angle)
    additional_y_quat = np.array([0, s, 0, c])  # [x, y, z, w] format
    combined_rot = existing_rot * st.Rotation.from_quat(additional_y_quat)
    initial_pose[3:] = combined_rot.as_quat()  # [x, y, z, w] format

    # Set camera
    global camera  # 전역 변수로 선언
    print(f"카메라 초기화 중 (Serial: {config['camera']['main_camera']['serial_number']})...")
    camera = RSCapture(
        name=config['camera']['main_camera']['name'],
        serial_number=config['camera']['main_camera']['serial_number'],
        dim=tuple(config['camera']['main_camera']['resolution']),
        fps=config['camera']['main_camera']['fps'],
        depth=False
    )

    global additional_cam
    print(f"추가 카메라 초기화 중 (Serial: {config['camera']['additional_camera']['serial_number']})...")
    additional_cam = RSCapture(
        name=config['camera']['additional_camera']['name'],
        serial_number=config['camera']['additional_camera']['serial_number'],
        dim=tuple(config['camera']['additional_camera']['resolution']),
        fps=config['camera']['additional_camera']['fps'],
        depth=False
    )

    # Initialize IMU pipeline (RealSense)
    print("IMU 파이프라인 초기화 중...")
    imu_pipe = rs.pipeline()
    imu_cfg = rs.config()
    # imu_cfg.enable_device(config['camera']['main_camera']['serial_number']) # 특정 장치 지정 시도
    imu_cfg.enable_stream(rs.stream.accel)
    imu_cfg.enable_stream(rs.stream.gyro)
    try:
        imu_pipe.start(imu_cfg)
    except Exception as e:
        print(f"IMU 초기화 실패 (무시하고 진행): {e}")
        imu_pipe = None

    # FT collector using Aidin + IMU gravity compensation
    ft_collector = FTCollector(
        ft_reader=ft_reader,
        imu_pipe=imu_pipe,
        gravity_compensator=gravity_compensator,
        rate_hz=config['ft_sensor']['rate_hz'],
        buf_len=config['ft_sensor']['buffer_length']
    )

    # 서버 연결
    client_socket = connect_to_server(server_ip, server_port)
    if client_socket is None:
        return

    with SharedMemoryManager() as shm_manager:
        # OpenCV GUI 백엔드를 먼저 초기화 (SpaceMouse와의 충돌 방지)
        print("[DEBUG] OpenCV GUI 백엔드 사전 초기화 중...")
        try:
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imshow('Initializing...', dummy_img)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # 창이 완전히 닫힐 때까지 대기
            print("[DEBUG] OpenCV GUI 백엔드 초기화 완료")
        except Exception as e:
            print(f"[WARNING] OpenCV GUI 사전 초기화 실패: {e}")
        
        with Spacemouse(shm_manager=shm_manager) as sm:
            # 터미널 설정 저장
            old_settings = termios.tcgetattr(sys.stdin)

            # load checkpoint (CLI overrides config)
            if not model_checkpoint_path:
                ckpt_path = (config.get('paths', {}).get('model_checkpoint')
                             or config.get('model_checkpoint_path', ''))
            else:
                ckpt_path = model_checkpoint_path
            if not ckpt_path:
                raise ValueError("모델 체크포인트 경로가 없습니다. '--model_checkpoint_path'를 지정하거나 config의 'paths.model_checkpoint' 또는 'model_checkpoint_path'를 설정하세요.")
            payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
            cfg = payload['cfg']
            cfg.policy.obs_encoder.pretrained = False # pretrained 설정 추가
            
            def patch_config(config):
                if isinstance(config, (dict, DictConfig)):
                    # Use keys() to safely iterate and modify
                    for k in config.keys():
                        v = config[k]
                        if k == '_target_' and isinstance(v, str):
                            new_v = None
                            if 'train_diffusion_unet_image_workspace' in v:
                                new_v = 'diffusion_policy.workspace.train_manipforce_workspace.TrainManipForceWorkspace'
                            elif 'timm_obs_encoder' in v or 'TimmObsEncoder' in v:
                                new_v = 'diffusion_policy.model.vision.fmt_obs_encoder.FMTObsEncoder'
                            elif 'diffusion_transformer_timm_policy' in v:
                                new_v = 'diffusion_policy.policy.diffusion_transformer_timm_policy.DiffusionTransformerTimmPolicy'
                            
                            if new_v and new_v != v:
                                print(f"[Patch] Changing {v} -> {new_v}")
                                config[k] = new_v
                        else:
                            patch_config(v)
                elif isinstance(config, (list, ListConfig)):
                    for item in config:
                        patch_config(item)

            # Apply patching to the entire config
            patch_config(cfg)
            
            try:
                cls = hydra.utils.get_class(cfg._target_)
            except (ImportError, AttributeError):
                from diffusion_policy.workspace.train_manipforce_workspace import TrainManipForceWorkspace
                cls = TrainManipForceWorkspace
            
            workspace = cls(cfg)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)
            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model
            policy.num_inference_steps = config['model']['num_inference_steps'] # DDIM inference iterations
            obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
            action_pose_repr = cfg.task.pose_repr.action_pose_repr
    
            try:
                target_pose = initial_pose.copy()
                current_pos = initial_pose[:3].copy()
                current_rot = st.Rotation.from_quat(initial_pose[3:])
                
                # 프로그램 시작 시 자동으로 initial pose로 이동
                print("초기 위치로 이동 중...")
                try:
                    data = {
                        'target_pose': initial_pose.tolist(),
                        'timestamp': time.time(),
                        'gripper_command': 'keep',
                        'reset': True
                    }
                    message = json.dumps(data, separators=(',', ':'))
                    client_socket.sendall(message.encode('utf-8') + b'\n')
                    time.sleep(2.0)  # 2초 대기하여 로봇이 이동할 시간 확보
                    print("초기 위치 이동 완료")
                except (socket.error, BrokenPipeError) as e:
                    print(f"초기 위치 이동 실패: {e}")
                
                # Initial pose에서 IMU warmup 및 중력 보상 캘리브레이션
                print("IMU warmup 및 중력 보상 캘리브레이션 시작...")
                gravity_compensator.calibrate_baseline(
                    imu_pipe, ft_reader,
                    warmup_sec=float(config['gravity_compensator']['warmup_sec'])
                )
                print("IMU warmup 및 중력 보상 캘리브레이션 완료")
                
                # FT 보정을 위해 FT collector 시작
                print("FT 보정 시작...")
                ft_collector.start()
                time.sleep(3.0)  # 3초간 FT 데이터 수집하여 bias 설정
                print("FT 보정 완료")
                
                # 그리퍼 상태 추적
                gripper_state = 'keep' #open, close, keep
                last_btn0_state = False
                last_command_time = time.monotonic()
                
                # 메뉴 표시 및 초기 모드 선택 (일반 터미널 모드에서)
                menu = """
                === 제어 모드 ===
                t: Teleop 모드
                p: Policy 모드
                q: 프로그램 종료
                r: 로봇 위치 리셋
                """
                print(menu)
                print("시작할 모드를 선택하세요 (t/p): ", end='', flush=True)
                mode = input().strip().lower()
                while mode not in ['t', 'p']:
                    print("잘못된 입력입니다. 't' 또는 'p'를 입력하세요: ", end='', flush=True)
                    mode = input().strip().lower()
                
                mode = 'teleop' if mode == 't' else 'policy'
                print(f"\n{mode} 모드를 시작합니다...")
                print("[INFO] 키 입력은 OpenCV 창에 포커스를 맞춘 후 사용하세요 (t/p/r/q)")
                
                # OpenCV GUI 백엔드 초기화
                try:
                    cv2.startWindowThread()
                    print("[DEBUG] cv2.startWindowThread() 호출 성공")
                except Exception as e:
                    print(f"[WARNING] cv2.startWindowThread() 실패: {e}")
                
                # OpenCV 창은 첫 imshow 호출 시 자동 생성됨
                # Raw 모드는 cv2.waitKey와 충돌하므로 사용하지 않음
                
                iter_idx = 0
                
                # 이전 스텝에서 실행한 action 추적 (policy 모드용)
                last_executed_action = np.zeros(8, dtype=np.float32)
                last_executed_action[7] = 1.0  # 그리퍼 기본값 (open)
                
                while True:
                    cycle_start = time.monotonic()
                    
                    # 키보드 입력 체크
                    key = check_keyboard_input()
                    if key:
                        if key == 'q':                    
                            # save ft data
                            # save_ft_data(ft_collector, filename="real_ft_data.csv")        
                            print("\n프로그램 종료")
                            break
                        elif key == 't':
                            mode = 'teleop'
                            print("\n텔레오퍼레이션 모드로 전환")
                            if hasattr(policy, '_init_done'):
                                delattr(policy, '_init_done')  # policy 초기화 플래그 제거
                            # 초기 상태 설정
                            prev_target_pose = initial_pose.copy()  # 초기 포즈로 리셋
                            continue
                        elif key == 'p':
                            mode = 'policy'
                            print("\nPolicy 모드로 전환")
                            policy.reset()  # policy 초기화
                            # FT collector는 이미 시작되어 있으므로 중복 시작하지 않음
                            continue
                        elif key == 'r':
                            print("\n초기 위치로 리셋")
                            current_pos = initial_pose[:3].copy()
                            current_rot = st.Rotation.from_quat(initial_pose[3:])
                            target_pose[:3] = current_pos
                            target_pose[3:] = current_rot.as_quat()
                            # 리셋 명령 전송
                            try:
                                data = {
                                    'target_pose': target_pose.tolist(),
                                    'timestamp': cycle_start,
                                    'gripper_command': gripper_state,
                                    'reset': True
                                }
                                message = json.dumps(data, separators=(',', ':'))
                                client_socket.sendall(message.encode('utf-8') + b'\n')
                                time.sleep(0.1)
                            except (socket.error, BrokenPipeError) as e:
                                print(f"\n리셋 명령 전송 실패: {e}")
                            continue
                                        
                    if mode == 'teleop':    
                        # ========= human control loop ==========
                        sm_state = sm.get_motion_state_transformed()
                        btn0 = sm.is_button_pressed(0)
                        btn1 = sm.is_button_pressed(1)
                        
                        # 그리퍼 토글 처리
                        if btn0 and not last_btn0_state:
                            gripper_state = 'close' if gripper_state == 'open' else 'open'
                            last_command_time = cycle_start
                        last_btn0_state = btn0

                        # 입력 변환 및 스케일링
                        dpos = sm_state[:3] * pos_scale
                        drot_xyz = sm_state[3:] * rot_scale
                        current_pos += dpos
                        
                        # 회전 업데이트
                        if np.any(drot_xyz != 0):
                            drot = st.Rotation.from_euler('xyz', drot_xyz)
                            current_rot = drot * current_rot

                        # 타겟 포즈 업데이트
                        target_pose[:3] = current_pos
                        target_pose[3:] = current_rot.as_quat()
                        
                        # 카메라 영상 가져오기 및 시각화
                        f1_rgb, f2_rgb, f1_bgr, f2_bgr = _read_two_cams()
                        if iter_idx % 50 == 0:
                            print(f"\r[Teleop] Frame received. f1 shape: {f1_bgr.shape}", end='', flush=True)
       
                        
                        # 원본 크기 유지하되 패딩으로 높이를 맞춤
                        h1, w1 = f1_bgr.shape[:2]
                        h2, w2 = f2_bgr.shape[:2]
                        
                        # 더 큰 높이에 맞춰서 패딩
                        target_height = max(h1, h2)
                        
                        # f1 패딩
                        if h1 < target_height:
                            pad_top = (target_height - h1) // 2
                            pad_bottom = target_height - h1 - pad_top
                            f1_padded = cv2.copyMakeBorder(f1_bgr, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                        else:
                            f1_padded = f1_bgr
                            
                        # f2 패딩
                        if h2 < target_height:
                            pad_top = (target_height - h2) // 2
                            pad_bottom = target_height - h2 - pad_top
                            f2_padded = cv2.copyMakeBorder(f2_bgr, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                        else:
                            f2_padded = f2_bgr
                        
                        # 두 이미지를 가로로 나란히 배치
                        vis_img = np.hstack([f1_padded, f2_padded])
                        
                        # 화면 크기 조정 (비율 유지하면서 크기 줄이기)
                        display_scale = 0.6  # 60% 크기로 줄이기
                        h, w = vis_img.shape[:2]
                        new_h, new_w = int(h * display_scale), int(w * display_scale)
                        vis_img_resized = cv2.resize(vis_img, (new_w, new_h))
                        
                        # 현재 모드와 키 안내 텍스트 추가
                        text_img = np.zeros((100, new_w, 3), dtype=np.uint8)
                        cv2.putText(text_img, f"Mode: {mode.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(text_img, "Keys: t(teleop) p(policy) r(reset) q(quit)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                        
                        # 이미지와 텍스트 합치기
                        combined_img = np.vstack([vis_img_resized, text_img])

                        
                        # 이미지 제목과 함께 표시
                        win_name = 'Camera 1 | Camera 2'
                        try:
                            cv2.imshow(win_name, combined_img)
                        except Exception as e:
                            if iter_idx == 0:
                                print(f"[ERROR] cv2.imshow failed: {e}")
                                print("[INFO] 시각화 없이 계속합니다...")
                        # cv2.waitKey는 check_keyboard_input()에서 호출됨

                        # 데이터 전송
                        try:
                            data = {
                                'target_pose': target_pose.tolist(),
                                'timestamp': cycle_start,
                                'gripper_command': gripper_state,
                                'reset': False
                            }
                            message = json.dumps(data, separators=(',', ':'))
                            client_socket.sendall(message.encode('utf-8') + b'\n')
                        except (socket.error, BrokenPipeError) as e:
                            print(f"서버 연결 오류: {e}. 재연결 시도...")
                            client_socket.close()
                            client_socket = connect_to_server(server_ip, server_port)
                            if client_socket is None:
                                print("재연결 실패. 루프 유지.")
                                time.sleep(0.1)
                                continue

                    if mode == 'policy':
                        device = torch.device('cuda')
                        policy.eval().to(device)
                        
                        # policy 모드로 처음 전환될 때만 초기화
                        if not hasattr(policy, '_init_done'):
                            policy._init_done = True
                            prev_target_pose = initial_pose.copy()
                            policy.reset()
                            
                            print("Warming up policy inference")
                            obs_np, img1, img2, ft_now, img1_view, img2_view, ft_ts = get_obs(ft_collector, last_executed_action)
                            with torch.no_grad():
                                obs_dict_np = get_real_gumi_obs_dict(
                                    env_obs=obs_np, shape_meta=cfg.task.shape_meta, 
                                    obs_pose_repr=obs_pose_rep)
                                obs_dict = dict_apply(obs_dict_np, 
                                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                                result = policy.predict_action(obs_dict)
                                action = result['action_pred'][0].detach().to('cpu').numpy()
                                action_8d = convert_action_10d_to_8d(action[-1])
                        
                        # policy.reset()
                        obs_np, img1, img2, ft_now, img1_view, img2_view, ft_ts = get_obs(ft_collector, last_executed_action)   # img1/img2: BGR uint8
                        
                        # Policy 실행 중에도 키 입력 체크 (리셋 응답성 향상)
                        key_during_policy = check_keyboard_input()
                        if key_during_policy:
                            if key_during_policy == 'q':
                                print("\n프로그램 종료")
                                break
                            elif key_during_policy == 't':
                                mode = 'teleop'
                                print("\n텔레오퍼레이션 모드로 전환")
                                if hasattr(policy, '_init_done'):
                                    delattr(policy, '_init_done')
                                prev_target_pose = initial_pose.copy()
                                continue
                            elif key_during_policy == 'r':
                                print("\n초기 위치로 리셋")
                                prev_target_pose = initial_pose.copy()
                                # 리셋 명령 전송
                                try:
                                    data = {
                                        'target_pose': initial_pose.tolist(),
                                        'timestamp': time.time(),
                                        'gripper_command': 'keep',
                                        'reset': True
                                    }
                                    message = json.dumps(data, separators=(',', ':'))
                                    client_socket.sendall(message.encode('utf-8') + b'\n')
                                    time.sleep(0.5)
                                except (socket.error, BrokenPipeError) as e:
                                    print(f"\n리셋 명령 전송 실패: {e}")
                                # policy 재초기화
                                if hasattr(policy, '_init_done'):
                                    delattr(policy, '_init_done')
                                continue
                        
                        # 현재 로봇 상태 가져오기
                        current_ee_pose = franka_api.get_pose_sync()
                        
                        with torch.no_grad():
                            s = time.time()  # 시작 시간 측정
                            # torch.cuda.empty_cache()
                            obs_dict_np = get_real_gumi_obs_dict(
                                env_obs=obs_np, shape_meta=cfg.task.shape_meta, 
                                obs_pose_repr=obs_pose_rep)
                            
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            
                            result = policy.predict_action(obs_dict)
                            action = result['action_pred'][0].detach().to('cpu').numpy()
                            # 잔차 처리 추가
                            if 'delta_p' in result:
                                delta_p = result['delta_p'][0].detach().to('cpu').numpy()  # (H, 3)
                                delta_p = delta_p[0]  # (3,)
                                action[-1, :3] += delta_p*0.01
                            
                            # ========= 시간 기반 액션 필터링 =========
                            # convert policy action to env actions
                            this_target_poses = action
                            
                            # deal with timing - 관측 타임스탬프 기준으로 스케줄링
                            # the same step actions are always the target for
                            obs_timestamp = ft_ts[-1]  # 관측의 마지막 타임스탬프 사용
                            action_timestamps = (np.arange(len(action), dtype=np.float64)) * dt + obs_timestamp
                            action_exec_latency = config['control']['action_exec_latency']
                            curr_time = time.time()
                            is_new = action_timestamps > (curr_time + action_exec_latency)
                            
                            if np.sum(is_new) == 0:
                                # exceeded time budget, still do something
                                this_target_poses = this_target_poses[[-1]]
                                # schedule on next available step
                                next_step_idx = int(np.ceil((curr_time - obs_timestamp) / dt))
                                action_timestamp = obs_timestamp + (next_step_idx) * dt
                                # print('Over budget', action_timestamp - curr_time)
                                action_timestamps = np.array([action_timestamp])
                            else:
                                this_target_poses = this_target_poses[is_new]
                                action_timestamps = action_timestamps[is_new]

                            # 안전 가드: 비어있는 경우 마지막 액션으로 대체
                            if this_target_poses.shape[0] == 0:
                                this_target_poses = action[[-1]]
                                action_timestamps = np.array([obs_timestamp + dt])
                            
                            # 항상 첫 번째 액션을 사용하여 일관성 유지
                            selected_action = this_target_poses[0]
                            selected_timestamp = action_timestamps[0]
                            
                            # 필터링된 액션의 첫 번째 사용
                            action_8d = convert_action_10d_to_8d(selected_action)
                            
                            # 이전 액션 저장
                            policy._last_action = action_8d.copy()
                            
                            transformed_action = transform_pose(action_8d, current_ee_pose, ft_now)
                            
                            # gripper는 항상 keep으로 고정
                            gripper_state = 'keep'

                            # position update
                            this_target_pose = prev_target_pose.copy()
                            this_target_pose[:3] += transformed_action[:3]
                            transformed_rot = Rotation.from_quat(transformed_action[3:])
                            current_rot = Rotation.from_quat(prev_target_pose[3:])
                            
                            # 보간없이 바로 사용
                            new_rot = current_rot * transformed_rot
                            this_target_pose[3:] = new_rot.as_quat()

                            # change to euler
                            target_euler = Rotation.from_quat(this_target_pose[3:]).as_euler('xyz', degrees=True)
                            if target_euler[1] < -35:
                                target_euler[1] = -35
                            # 다시 quaternion으로 변환하고 반영
                            new_quat = Rotation.from_euler('xyz', target_euler, degrees=True).as_quat()
                            # normalize to be safe
                            new_quat = new_quat / np.linalg.norm(new_quat)
                            this_target_pose[3:] = new_quat

                            # 마지막 포즈 전송 후에만 짧게 대기
                            time.sleep(0.01)
                            
                            # prev_target_pose 업데이트
                            prev_target_pose = this_target_pose.copy()
                            last_executed_action = action_8d.copy()

                            # # # target pose의 z값 이 0.045 이하일 경우 0.045로 설정
                            # if this_target_pose[2] < 0.06:
                            #     this_target_pose[2] = 0.06

                            # Battery disassem w tool
                            # if this_target_pose[2] < 0.036:
                            #     this_target_pose[2] = 0.036
                          
                            # target pose의 z값 이 0.045 이하일 경우 0.045로 설정
                            # if this_target_pose[2] < 0.015:
                            #     this_target_pose[2] = 0.015
                            
                            # Open lid
                            # if this_target_pose[2] < 0.07:
                            #     this_target_pose[2] = 0.07
                            
                            # this_target_pose[1] = initial_pose[1]
                            

                            e = time.time()  # 종료 시간 측정
                            cycle_time = e - s  # 한 사이클에 걸린 시간
                            hz = 1.0 / cycle_time  # Hz 계산
                            
                            # 액션 Hz 출력 (1초마다)
                            if iter_idx % int(frequency) == 0:  # 1초마다 출력
                                print(f"\r[Action Rate] {hz:.1f} Hz | Cycle: {cycle_time*1000:.1f}ms", end='', flush=True)
                            
                            try:
                                # 데이터 전송 최적화 - 타임스탬프 포함
                                data = {        
                                    'target_pose': this_target_pose.tolist(),
                                    'gripper_command': gripper_state,
                                    'reset': False,
                                    'timestamp': selected_timestamp,  # 액션 실행 타임스탬프 추가
                                    'action_timestamps': action_timestamps.tolist()  # 전체 액션 시퀀스 타임스탬프
                                }
                                # JSON 직렬화 최적화
                                message = json.dumps(data, separators=(',', ':'))
                                client_socket.sendall(message.encode('utf-8') + b'\n')
                            except (socket.error, BrokenPipeError) as e:
                                print(f"\r서버 연결 오류: {e}", end='', flush=True)
                                try:
                                    client_socket.close()
                                    client_socket = connect_to_server(server_ip, server_port)
                                except:
                                    pass
                                continue

                            # 두 이미지를 가로로 나란히 배치 (원본 크기, 패딩으로 높이 맞춤)
                            h1, w1 = img1_view.shape[:2]
                            h2, w2 = img2_view.shape[:2]
                            
                            # 더 큰 높이에 맞춰서 패딩
                            target_height = max(h1, h2)
                            
                            # img1 패딩
                            if h1 < target_height:
                                pad_top = (target_height - h1) // 2
                                pad_bottom = target_height - h1 - pad_top
                                img1_padded = cv2.copyMakeBorder(img1_view, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                            else:
                                img1_padded = img1_view
                                
                            # img2 패딩
                            if h2 < target_height:
                                pad_top = (target_height - h2) // 2
                                pad_bottom = target_height - h2 - pad_top
                                img2_padded = cv2.copyMakeBorder(img2_view, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                            else:
                                img2_padded = img2_view
                            
                            vis_img = np.hstack([img1_padded, img2_padded])
                            
                            # 화면 크기 조정 (비율 유지하면서 크기 줄이기)
                            display_scale = 0.6  # 60% 크기로 줄이기
                            h, w = vis_img.shape[:2]
                            new_h, new_w = int(h * display_scale), int(w * display_scale)
                            vis_img_resized = cv2.resize(vis_img, (new_w, new_h))
                            
                            # 현재 모드와 키 안내 텍스트 추가
                            text_img = np.zeros((100, new_w, 3), dtype=np.uint8)
                            cv2.putText(text_img, f"Mode: {mode.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            cv2.putText(text_img, "Keys: t(teleop) p(policy) r(reset) q(quit)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                            
                            # 이미지와 텍스트 합치기
                            combined_img = np.vstack([vis_img_resized, text_img])

                            # 이미지 제목과 함께 표시
                            win_name = 'Camera 1 | Camera 2'
                            cv2.imshow(win_name, combined_img)
                            cv2.waitKey(1)

                    cycle_end = cycle_start + dt
                    sleep_time = cycle_end - time.monotonic()
                    if sleep_time > 0:
                        precise_wait(cycle_end)
                    
                    iter_idx += 1

            finally:
                # 터미널 설정 복원 (raw 모드 사용 안 함)
                # termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                try:
                    ft_collector.stop()
                except Exception:
                    pass
                try:
                    ft_reader.stop()
                except Exception:
                    pass
                try:
                    imu_pipe.stop()
                except Exception:
                    pass
                client_socket.close()
                cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기
                print("\n서버 연결을 종료합니다.")
            
            
            
if __name__ == "__main__":
    main()