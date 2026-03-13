
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

import socket
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
import threading


def load_config(config_path='inference_config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

config = None

prev_frames_cam1, prev_frames_cam2, prev_timestamps = None, None, None

_last_disp_t = time.time()
_disp_cnt = 0
_img_accum = 0.0
_ft_accum = 0.0

action_history = deque(maxlen=4)
USE_ACTION_HISTORY = False

gripper_history = None
current_gripper_state = 'open'
current_gripper_target = 'open'


def close_gripper_http(server_ip, port):
    """Close gripper via HTTP API."""
    try:
        response = requests.post(f"http://{server_ip}:{port}/close_gripper", timeout=1.0)
        return response.status_code == 200
    except Exception:
        return False


def open_gripper_http(server_ip, port):
    """Open gripper via HTTP API."""
    try:
        response = requests.post(f"http://{server_ip}:{port}/open_gripper", timeout=1.0)
        return response.status_code == 200
    except Exception:
        return False


def smart_gripper_control(predicted_gripper, franka_api, server_ip):
    """
    History-based gripper control logic.
    1.0 = open, 0.0 = close.
    Uses a sliding window of recent predictions to determine intent,
    with separate thresholds for opening and closing.
    """
    global current_gripper_state, current_gripper_target, gripper_history

    try:
        actual_gripper = franka_api.get_gripper_sync()
    except Exception:
        actual_gripper = 0.025

    gripper_history.append(predicted_gripper)

    if len(gripper_history) < config['gripper']['history_length']:
        return 'keep'

    hist_len = config['gripper']['history_length']
    recent_predictions = list(gripper_history)[-hist_len:]
    close_count = sum(1 for pred in recent_predictions if pred < config['gripper']['close_threshold'])
    open_count = sum(1 for pred in recent_predictions if pred >= config['gripper']['open_threshold'])

    is_physically_closed = (actual_gripper < 0.045)
    is_physically_open = (actual_gripper >= 0.045)

    wants_close = (close_count >= config['gripper']['close_count_threshold'])
    wants_open = (open_count >= config['gripper']['open_count_threshold'])

    if wants_close:
        if is_physically_closed:
            current_gripper_target = 'close'
            return 'keep'
        elif current_gripper_target != 'close':
            success = close_gripper_http(server_ip, config['gripper']['http_port'])
            if success:
                current_gripper_target = 'close'
                time.sleep(0.1)
            return 'close'
        else:
            return 'close'

    elif wants_open:
        if is_physically_open:
            current_gripper_target = 'open'
            return 'keep'
        elif current_gripper_target != 'open':
            success = open_gripper_http(server_ip, config['gripper']['http_port'])
            if success:
                current_gripper_target = 'open'
                time.sleep(0.1)
            return 'open'
        else:
            return 'open'

    else:
        return 'keep'


def _read_two_cams() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read frames from both cameras. Returns (rgb1, rgb2, bgr1, bgr2)."""
    ok1, res1 = camera.read()
    ok2, res2 = additional_cam.read()
    if not ok1:
        raise RuntimeError("Main camera read failed")

    f1 = res1[0]
    if not ok2:
        f2 = np.zeros_like(f1)
    else:
        f2 = res2[0]

    f1_rgb = cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)
    f2_rgb = cv2.cvtColor(f2, cv2.COLOR_BGR2RGB)

    return f1_rgb, f2_rgb, f1, f2


class FTCollector:
    """Collect F/T data from Aidin UDP sensor with IMU-based gravity compensation in a background thread."""

    def __init__(self, ft_reader, imu_pipe, gravity_compensator, rate_hz=200, buf_len=200):
        self.ft_reader = ft_reader
        self.imu_pipe = imu_pipe
        self.gravity_compensator = gravity_compensator
        self.rate_hz = rate_hz
        self.dt = 1.0 / max(1, rate_hz)

        self._th = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self.buf = deque(maxlen=buf_len)
        self.ts_buf = deque(maxlen=buf_len)

        self.full_ts_list = []
        self.full_ft_list = []

        self.f_bias_initial = None
        self.t_bias_initial = None
        self.bias_initialized = False

    def _loop(self):
        period = 1.0 / self.rate_hz
        nxt = time.perf_counter()

        while not self._stop.is_set():
            try:
                self.gravity_compensator.update_imu(self.imu_pipe)

                try:
                    ts_raw, f_raw, t_raw = self.ft_reader.get_frame(timeout=0.001)
                except Exception:
                    ts_raw, f_raw, t_raw = None, None, None

                if f_raw is not None and t_raw is not None:
                    forces_filt, torques_filt = self.gravity_compensator.process_ft_data(f_raw, t_raw)
                    compensated_force, compensated_torque = self.gravity_compensator.compensate_gravity(
                        forces_filt, torques_filt, gravity_compensation_on=True
                    )

                    if not self.bias_initialized:
                        self.f_bias_initial = compensated_force.copy()
                        self.t_bias_initial = compensated_torque.copy()
                        self.bias_initialized = True
                        print(f"FT initial bias set: Force={self.f_bias_initial}, Torque={self.t_bias_initial}")

                    f_final = compensated_force - self.f_bias_initial
                    f_final[1] -= 3.0  # empirical fy offset
                    t_final = compensated_torque - self.t_bias_initial

                    ft_vec = np.concatenate([f_final, t_final]).astype(np.float32)
                    ts = time.time()

                    with self._lock:
                        self.buf.append(ft_vec)
                        self.ts_buf.append(ts)
                        self.full_ts_list.append(ts)
                        self.full_ft_list.append(ft_vec)
            except Exception as e:
                print(f"\rFT collection error: {e}", end='', flush=True)

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
        """Return the most recent `length` F/T samples, padding if needed."""
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
    """Print rolling average of image FPS and F/T Hz every second."""
    global _last_disp_t, _disp_cnt, _img_accum, _ft_accum

    _disp_cnt += 1
    _img_accum += img_freq
    if len(ft_ts) >= 2:
        time_diffs = np.diff(ft_ts[-8:])
        if np.all(time_diffs > 0):
            ft_hz = 1.0 / np.mean(time_diffs)
            _ft_accum += ft_hz
        else:
            ft_hz = 0.0
    else:
        ft_hz = 0.0

    now = time.time()
    if now - _last_disp_t >= 1.0:
        avg_img = _img_accum / _disp_cnt
        avg_ft = _ft_accum / _disp_cnt
        sys.stdout.write(
            f"\r[Rate] IMG {avg_img:5.1f} FPS | FT {avg_ft:5.1f} Hz")
        sys.stdout.flush()

        _last_disp_t = now
        _disp_cnt = _img_accum = _ft_accum = 0.0


def quaternion_multiply(q1, q2):
    """Hamilton product for quaternions in [w, x, y, z] order."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.array([w, x, y, z])


def transform_pose(pose, current_ee_pose, ft_data):
    """Transform predicted delta action from policy frame to world frame."""
    R_mat_pos = np.array(config['coordinate_transform']['R_mat_pos'])
    R_mat_rot = np.array(config['coordinate_transform']['R_mat_rot'])

    if 'rotation_x_scale' in config['action'] and 'rotation_y_scale' in config['action']:
        rotation_x_scale = config['action']['rotation_x_scale']
        rotation_y_scale = config['action']['rotation_y_scale']
    else:
        rotation_xy_scale = config['action']['rotation_xy_scale']
        rotation_x_scale = rotation_xy_scale
        rotation_y_scale = rotation_xy_scale

    rotation_z_scale = config['action']['rotation_z_scale']

    rel_pos = R_mat_pos @ pose[:3]

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

    delta_quat = np.array(pose[3:7])  # [x, y, z, w]

    R_orig = Rotation.from_quat(delta_quat)
    rotvec_orig = R_orig.as_rotvec()

    rotvec_scaled = rotvec_orig.copy()
    rotvec_scaled[0] *= rotation_x_scale
    rotvec_scaled[1] *= rotation_y_scale
    rotvec_scaled[2] *= rotation_z_scale

    R_scaled = Rotation.from_rotvec(rotvec_scaled).as_matrix()
    R_new = R_mat_rot @ R_scaled @ R_mat_rot.T
    rel_quat = Rotation.from_matrix(R_new).as_quat()

    current_quat = current_ee_pose[3:]   # [w, x, y, z]
    current_quat = np.roll(current_quat, -1)  # -> [x, y, z, w]
    current_rot = Rotation.from_quat(current_quat)
    world_rel_pos = current_rot.inv().apply(rel_pos)

    return np.concatenate([world_rel_pos, rel_quat])


def nudge_y_if_stuck(current_ee_pose: np.ndarray,
                     last_target_pose: np.ndarray,
                     pos_thresh: float = 1e-4,
                     nudge_amount: float = 0.002) -> Tuple[np.ndarray, bool]:
    """
    Apply a small y-axis perturbation when the EE is stuck
    (position barely changed from last target).
    """
    try:
        pos_delta = np.linalg.norm((current_ee_pose[:3] - last_target_pose[:3]).astype(np.float64))
    except Exception:
        pos_delta = np.linalg.norm(current_ee_pose[:3] - last_target_pose[:3])

    if pos_delta < pos_thresh:
        nudged = last_target_pose.copy()
        nudged[1] += nudge_amount * _y_nudge_dir
        _y_nudge_dir *= -1
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
    """Non-blocking keyboard input via OpenCV window."""
    key = cv2.waitKey(1) & 0xFF

    if key != 255:
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
    """Resize image to a square while preserving aspect ratio via padding."""
    h, w = img.shape[:2]
    aspect = w / h

    if aspect > 1:
        new_w = target_size
        new_h = int(target_size / aspect)
    else:
        new_h = target_size
        new_w = int(target_size * aspect)

    pil_img = Image.fromarray(img)
    resized = pil_img.resize((new_w, new_h))

    new_img = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    new_img.paste(resized, (paste_x, paste_y))

    return np.array(new_img)


def get_obs(ft_collector, last_executed_action=None):
    """Collect a single observation: camera frames + F/T data aligned by timestamp."""
    TARGET_HW = tuple(config['image']['target_resolution'])

    # Initialize frame buffers on first call
    while len(prev_frames_cam1) < config['image']['history_length']:
        f1_rgb, f2_rgb, f1_bgr, f2_bgr = _read_two_cams()
        prev_frames_cam1.append(f1_rgb)
        prev_frames_cam2.append(f2_rgb)
        prev_timestamps.append(time.time())
        time.sleep(1/30)

    # Read new camera frame
    now_img = time.time()
    f1_rgb, f2_rgb, f1_bgr, f2_bgr = _read_two_cams()
    prev_frames_cam1.append(f1_rgb)
    prev_frames_cam2.append(f2_rgb)
    prev_timestamps.append(now_img)

    # Align F/T samples to image timestamps (same logic as sampler.py)
    ft_buf, ts_buf = ft_collector.window(config['ft_sensor']['buffer_length'])
    img_ts_hist = np.array(list(prev_timestamps)[-config['image']['history_length']:], dtype=np.float64)

    if img_ts_hist.size >= 2:
        ds = int(config['image'].get('down_sample_steps', 1))
        if img_ts_hist.size >= ds + 1:
            t0 = img_ts_hist[-(ds + 1)]
            t1 = img_ts_hist[-1]
        else:
            t0 = img_ts_hist[-2]
            t1 = img_ts_hist[-1]
    elif img_ts_hist.size == 1:
        t0 = img_ts_hist[0]
        t1 = img_ts_hist[0] + 1e-3
    else:
        now = time.time()
        t0, t1 = now - 1e-3, now

    img_ts_int = np.array([int(t0 * 1e6), int(t1 * 1e6)], dtype=np.int64)
    ft_ts_int = np.array([int(ts * 1e6) for ts in ts_buf], dtype=np.int64)

    left_idx = np.searchsorted(ft_ts_int, img_ts_int[0], side='left')
    right_idx = np.searchsorted(ft_ts_int, img_ts_int[1], side='right')
    left_idx = max(0, left_idx)
    right_idx = min(len(ft_ts_int), right_idx)

    ft_slice = ft_buf[left_idx:right_idx]
    ft_ts_slice = ts_buf[left_idx:right_idx]
    n = len(ft_slice)

    target_ft_frames = config['ft_sensor']['obs_horizon']

    try:
        if n < target_ft_frames:
            if n == 0:
                ft_sel = np.zeros((target_ft_frames, 6), dtype=np.float32)
                ts_sel = np.zeros(target_ft_frames, dtype=np.float64)
            else:
                pad_n = target_ft_frames - n
                last_frame = ft_slice[-1][None, ...]
                last_ts = ft_ts_slice[-1][None]
                ft_sel = np.concatenate([ft_slice, np.repeat(last_frame, pad_n, axis=0)], axis=0)
                ts_sel = np.concatenate([ft_ts_slice, np.repeat(last_ts, pad_n)], axis=0)
        elif n > target_ft_frames:
            idxs = np.linspace(0, n-1, target_ft_frames).round().astype(int)
            ft_sel = ft_slice[idxs]
            ts_sel = ft_ts_slice[idxs]
        else:
            ft_sel = ft_slice
            ts_sel = ft_ts_slice

        assert ft_sel.shape[0] == target_ft_frames, f"FT shape mismatch: {ft_sel.shape[0]} != {target_ft_frames}"

    except (ValueError, TypeError) as e:
        print(f"FT timestamp mapping error: {e}, using default")
        ft_sel = np.zeros((target_ft_frames, 6), dtype=np.float32)
        ts_sel = np.zeros(target_ft_frames, dtype=np.float64)

    ft_sel_converted = convert_torque_scale(ft_sel)
    obs = build_obs_np(f1_rgb, f2_rgb, ft_sel_converted, ts_sel, last_executed_action)

    return obs, f1_rgb, f2_rgb, ft_sel, f1_bgr, f2_bgr, ts_sel


def convert_torque_scale(ft):
    """Scale torques by lever-arm ratio (sensor: 12cm, Panda flange: 18cm)."""
    ft = np.asarray(ft)
    torque_scale = 12.0 / 18.0

    if ft.ndim == 1:
        ft[3] *= torque_scale
        ft[4] *= torque_scale
        ft[5] *= torque_scale
        return np.array([ft[0], ft[1], ft[2], ft[3], ft[4], ft[5]], dtype=ft.dtype)
    else:
        ft[:, 3] *= torque_scale
        ft[:, 4] *= torque_scale
        ft[:, 5] *= torque_scale
        return np.stack([ft[:, 0], ft[:, 1], ft[:, 2], ft[:, 3], ft[:, 4], ft[:, 5]], axis=-1)


def build_obs_np(img1: np.ndarray,
                 img2: np.ndarray,
                 ft_sel: np.ndarray,
                 ft_timestamps: np.ndarray = None,
                 last_executed_action: np.ndarray = None
                 ) -> dict:
    """
    Build observation dict matching training data format.
    img1, img2: (H, W, 3) uint8 RGB
    ft_sel: (N, 6) float32
    """
    global action_history

    # Normalize images to [0, 1] in TCHW format
    cam1 = np.stack(prev_frames_cam1, axis=0).astype(np.float32) / 255.0
    cam2 = np.stack(prev_frames_cam2, axis=0).astype(np.float32) / 255.0
    cam1_tchw = np.moveaxis(cam1, -1, 1)
    cam2_tchw = np.moveaxis(cam2, -1, 1)

    # Resize with padding if needed
    expected_shape = tuple(config['image']['target_resolution'])
    target_size = expected_shape[0]
    if cam1_tchw.shape[2] != target_size or cam1_tchw.shape[3] != target_size:
        resized = []
        for t in range(cam1_tchw.shape[0]):
            curr_img = np.moveaxis(cam1_tchw[t], 0, -1)
            curr_img_uint8 = (curr_img * 255).astype(np.uint8)
            resized_pil = resize_with_padding(curr_img_uint8, target_size=target_size)
            resized_np = resized_pil.astype(np.float32) / 255.0
            resized_np = np.moveaxis(resized_np, -1, 0)
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
        'handeye_cam_1': cam1_tchw,
        'handeye_cam_2': cam2_tchw,
        'ft_data': ft_sel,
    }

    if ft_timestamps is not None:
        obs_np['ft_timestamps'] = ft_timestamps.astype(np.float32)

    return obs_np


def save_ft_data(ft_collector, filename="ft_trace.csv"):
    """Save the full F/T trace from FTCollector to a CSV file."""
    ft_collector.stop()

    with ft_collector._lock:
        data = list(ft_collector.buf)

    if not data:
        return

    ts, ft = zip(*data)
    arr_ts = np.array(ts, dtype=np.float64).reshape(-1, 1)
    arr_ft = np.vstack(ft).astype(np.float32)
    csv_data = np.hstack([arr_ts, arr_ft])
    header = "timestamp,fx,fy,fz,tx,ty,tz"

    np.savetxt(filename, csv_data, delimiter=",", header=header, comments="")


@click.command()
@click.option('--config_path', '-c', default='inference_config.yaml', type=str, help="Path to inference config")
@click.option('--model_checkpoint_path', '-mcp', default='', type=str, help="Path to model checkpoint")
def main(config_path, model_checkpoint_path):
    global config, prev_frames_cam1, prev_frames_cam2, prev_timestamps, gripper_history

    config = load_config(config_path)

    prev_frames_cam1 = deque(maxlen=config['image']['history_length'])
    prev_frames_cam2 = deque(maxlen=config['image']['history_length'])
    prev_timestamps = deque(maxlen=config['image']['history_length'])
    gripper_history = deque(maxlen=config['gripper']['history_length'])

    server_ip = config['robot']['server_ip']
    server_port = config['robot']['server_port']
    frequency = config['control']['frequency']
    output = config['paths']['output_video']

    franka_api = FrankaAPI(server_ip)

    # Initialize Aidin F/T sensor (UDP)
    ft_ip = config.get('ft_sensor', {}).get('ip', '172.27.190.4')
    ft_port = int(config.get('ft_sensor', {}).get('port', 8890))
    ft_reader = AidinFTSensorUDP(ft_ip, ft_port)
    ft_reader.start()

    gravity_compensator = GravityCompensator(
        mass_for_x=config['gravity_compensator']['mass_for_x'],
        mass_for_y=config['gravity_compensator']['mass_for_y'],
        mass_for_z=config['gravity_compensator']['mass_for_z'],
        com_ft=np.array(config['gravity_compensator']['com_ft']),
        g_const=config['gravity_compensator']['g_const']
    )

    try:
        current_pose = franka_api.get_pose_sync()
        current_gripper = franka_api.get_gripper_sync()
    except Exception:
        pass

    print("\n=== Initial Robot Pose ===")
    print(f"Pose: {[current_pose[i] for i in range(7)]}")

    dt = 1 / frequency
    max_pos_speed = config['robot']['max_pos_speed']
    max_rot_speed = config['robot']['max_rot_speed']

    pos_scale = max_pos_speed * dt
    rot_scale = max_rot_speed * dt

    initial_pose = np.array(config['robot']['initial_pose'])

    # Add random position perturbation to initial pose
    random_pos_range = config['robot']['random_pos_range']
    pos_noise = np.random.uniform(-random_pos_range, random_pos_range, size=3)

    cfg_basename = os.path.basename(config_path).lower() if isinstance(config_path, str) else ''

    if 'battery_assemb' in cfg_basename:
        initial_pose[0] = initial_pose[0] + pos_noise[0]
        initial_pose[2] = initial_pose[2] + pos_noise[2]
    else:
        initial_pose[:3] = initial_pose[:3] + pos_noise

    initial_tilt_deg = float(config['robot'].get('initial_tilt_deg', 0.0))

    if 'jetson_flipping' in cfg_basename:
        additional_y_angle = np.random.uniform(0, initial_tilt_deg)
    else:
        additional_y_angle = np.random.uniform(10, initial_tilt_deg)
    additional_y_angle = initial_tilt_deg
    existing_quat = [1, 0, 0, 0]
    existing_rot = st.Rotation.from_quat(existing_quat)
    half_angle = np.deg2rad(additional_y_angle) / 2
    c, s = np.cos(half_angle), np.sin(half_angle)
    additional_y_quat = np.array([0, s, 0, c])  # [x, y, z, w]
    combined_rot = existing_rot * st.Rotation.from_quat(additional_y_quat)
    initial_pose[3:] = combined_rot.as_quat()

    # Initialize cameras
    global camera
    print(f"Initializing main camera (Serial: {config['camera']['main_camera']['serial_number']})...")
    camera = RSCapture(
        name=config['camera']['main_camera']['name'],
        serial_number=config['camera']['main_camera']['serial_number'],
        dim=tuple(config['camera']['main_camera']['resolution']),
        fps=config['camera']['main_camera']['fps'],
        depth=False
    )

    global additional_cam
    print(f"Initializing additional camera (Serial: {config['camera']['additional_camera']['serial_number']})...")
    additional_cam = RSCapture(
        name=config['camera']['additional_camera']['name'],
        serial_number=config['camera']['additional_camera']['serial_number'],
        dim=tuple(config['camera']['additional_camera']['resolution']),
        fps=config['camera']['additional_camera']['fps'],
        depth=False
    )

    # Initialize IMU pipeline (RealSense)
    print("Initializing IMU pipeline...")
    imu_pipe = rs.pipeline()
    imu_cfg = rs.config()
    imu_cfg.enable_stream(rs.stream.accel)
    imu_cfg.enable_stream(rs.stream.gyro)
    try:
        imu_pipe.start(imu_cfg)
    except Exception as e:
        print(f"IMU init failed (continuing without): {e}")
        imu_pipe = None

    ft_collector = FTCollector(
        ft_reader=ft_reader,
        imu_pipe=imu_pipe,
        gravity_compensator=gravity_compensator,
        rate_hz=config['ft_sensor']['rate_hz'],
        buf_len=config['ft_sensor']['buffer_length']
    )

    client_socket = connect_to_server(server_ip, server_port)
    if client_socket is None:
        return

    with SharedMemoryManager() as shm_manager:
        # Pre-initialize OpenCV GUI backend to avoid conflicts with SpaceMouse
        try:
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imshow('Initializing...', dummy_img)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception as e:
            print(f"[WARNING] OpenCV GUI pre-init failed: {e}")

        with Spacemouse(shm_manager=shm_manager) as sm:
            old_settings = termios.tcgetattr(sys.stdin)

            # Load checkpoint (CLI flag overrides config)
            if not model_checkpoint_path:
                ckpt_path = (config.get('paths', {}).get('model_checkpoint')
                             or config.get('model_checkpoint_path', ''))
            else:
                ckpt_path = model_checkpoint_path
            if not ckpt_path:
                raise ValueError("No model checkpoint path provided. "
                                 "Use '--model_checkpoint_path' or set 'paths.model_checkpoint' in config.")
            payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
            cfg = payload['cfg']
            cfg.policy.obs_encoder.pretrained = False

            def patch_config(config):
                if isinstance(config, (dict, DictConfig)):
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
                                print(f"[Patch] {v} -> {new_v}")
                                config[k] = new_v
                        else:
                            patch_config(v)
                elif isinstance(config, (list, ListConfig)):
                    for item in config:
                        patch_config(item)

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
            policy.num_inference_steps = config['model']['num_inference_steps']
            obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
            action_pose_repr = cfg.task.pose_repr.action_pose_repr

            try:
                target_pose = initial_pose.copy()
                current_pos = initial_pose[:3].copy()
                current_rot = st.Rotation.from_quat(initial_pose[3:])

                # Move to initial pose on startup
                print("Moving to initial pose...")
                try:
                    data = {
                        'target_pose': initial_pose.tolist(),
                        'timestamp': time.time(),
                        'gripper_command': 'keep',
                        'reset': True
                    }
                    message = json.dumps(data, separators=(',', ':'))
                    client_socket.sendall(message.encode('utf-8') + b'\n')
                    time.sleep(2.0)
                    print("Initial pose reached")
                except (socket.error, BrokenPipeError) as e:
                    print(f"Failed to send initial pose: {e}")

                # IMU warmup and gravity compensation calibration
                print("Running IMU warmup and gravity compensation calibration...")
                gravity_compensator.calibrate_baseline(
                    imu_pipe, ft_reader,
                    warmup_sec=float(config['gravity_compensator']['warmup_sec'])
                )
                print("Calibration complete")

                # Start F/T collector and wait for bias to settle
                print("Starting F/T bias calibration...")
                ft_collector.start()
                time.sleep(3.0)
                print("F/T calibration complete")

                gripper_state = 'keep'
                last_btn0_state = False
                last_command_time = time.monotonic()

                menu = """
                === Control Modes ===
                t: Teleop mode
                p: Policy mode
                q: Quit
                r: Reset robot pose
                """
                print(menu)
                print("Select starting mode (t/p): ", end='', flush=True)
                mode = input().strip().lower()
                while mode not in ['t', 'p']:
                    print("Invalid input. Enter 't' or 'p': ", end='', flush=True)
                    mode = input().strip().lower()

                mode = 'teleop' if mode == 't' else 'policy'
                print(f"\nStarting {mode} mode...")
                print("[INFO] Focus the OpenCV window for key input (t/p/r/q)")

                try:
                    cv2.startWindowThread()
                except Exception:
                    pass

                iter_idx = 0

                last_executed_action = np.zeros(8, dtype=np.float32)
                last_executed_action[7] = 1.0  # gripper default: open

                while True:
                    cycle_start = time.monotonic()

                    key = check_keyboard_input()
                    if key:
                        if key == 'q':
                            print("\nQuitting...")
                            break
                        elif key == 't':
                            mode = 'teleop'
                            print("\nSwitched to teleop mode")
                            if hasattr(policy, '_init_done'):
                                delattr(policy, '_init_done')
                            prev_target_pose = initial_pose.copy()
                            continue
                        elif key == 'p':
                            mode = 'policy'
                            print("\nSwitched to policy mode")
                            policy.reset()
                            continue
                        elif key == 'r':
                            print("\nResetting to initial pose")
                            current_pos = initial_pose[:3].copy()
                            current_rot = st.Rotation.from_quat(initial_pose[3:])
                            target_pose[:3] = current_pos
                            target_pose[3:] = current_rot.as_quat()
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
                                print(f"\nReset command failed: {e}")
                            continue

                    if mode == 'teleop':
                        # === Teleop control loop ===
                        sm_state = sm.get_motion_state_transformed()
                        btn0 = sm.is_button_pressed(0)
                        btn1 = sm.is_button_pressed(1)

                        if btn0 and not last_btn0_state:
                            gripper_state = 'close' if gripper_state == 'open' else 'open'
                            last_command_time = cycle_start
                        last_btn0_state = btn0

                        dpos = sm_state[:3] * pos_scale
                        drot_xyz = sm_state[3:] * rot_scale
                        current_pos += dpos

                        if np.any(drot_xyz != 0):
                            drot = st.Rotation.from_euler('xyz', drot_xyz)
                            current_rot = drot * current_rot

                        target_pose[:3] = current_pos
                        target_pose[3:] = current_rot.as_quat()

                        f1_rgb, f2_rgb, f1_bgr, f2_bgr = _read_two_cams()

                        # Visualize side-by-side camera feeds
                        h1, w1 = f1_bgr.shape[:2]
                        h2, w2 = f2_bgr.shape[:2]
                        target_height = max(h1, h2)

                        if h1 < target_height:
                            pad_top = (target_height - h1) // 2
                            pad_bottom = target_height - h1 - pad_top
                            f1_padded = cv2.copyMakeBorder(f1_bgr, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                        else:
                            f1_padded = f1_bgr

                        if h2 < target_height:
                            pad_top = (target_height - h2) // 2
                            pad_bottom = target_height - h2 - pad_top
                            f2_padded = cv2.copyMakeBorder(f2_bgr, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                        else:
                            f2_padded = f2_bgr

                        vis_img = np.hstack([f1_padded, f2_padded])

                        display_scale = 0.6
                        h, w = vis_img.shape[:2]
                        new_h, new_w = int(h * display_scale), int(w * display_scale)
                        vis_img_resized = cv2.resize(vis_img, (new_w, new_h))

                        text_img = np.zeros((100, new_w, 3), dtype=np.uint8)
                        cv2.putText(text_img, f"Mode: {mode.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(text_img, "Keys: t(teleop) p(policy) r(reset) q(quit)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                        combined_img = np.vstack([vis_img_resized, text_img])

                        win_name = 'Camera 1 | Camera 2'
                        try:
                            cv2.imshow(win_name, combined_img)
                        except Exception as e:
                            if iter_idx == 0:
                                print(f"[ERROR] cv2.imshow failed: {e}")

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
                            print(f"Connection error: {e}. Reconnecting...")
                            client_socket.close()
                            client_socket = connect_to_server(server_ip, server_port)
                            if client_socket is None:
                                print("Reconnection failed.")
                                time.sleep(0.1)
                                continue

                    if mode == 'policy':
                        # === Policy inference loop ===
                        device = torch.device('cuda')
                        policy.eval().to(device)

                        # One-time initialization on first policy entry
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

                        obs_np, img1, img2, ft_now, img1_view, img2_view, ft_ts = get_obs(ft_collector, last_executed_action)

                        # Check for mode-switch keys during policy execution
                        key_during_policy = check_keyboard_input()
                        if key_during_policy:
                            if key_during_policy == 'q':
                                print("\nQuitting...")
                                break
                            elif key_during_policy == 't':
                                mode = 'teleop'
                                print("\nSwitched to teleop mode")
                                if hasattr(policy, '_init_done'):
                                    delattr(policy, '_init_done')
                                prev_target_pose = initial_pose.copy()
                                continue
                            elif key_during_policy == 'r':
                                print("\nResetting to initial pose")
                                prev_target_pose = initial_pose.copy()
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
                                    print(f"\nReset command failed: {e}")
                                if hasattr(policy, '_init_done'):
                                    delattr(policy, '_init_done')
                                continue

                        current_ee_pose = franka_api.get_pose_sync()

                        with torch.no_grad():
                            s = time.time()

                            obs_dict_np = get_real_gumi_obs_dict(
                                env_obs=obs_np, shape_meta=cfg.task.shape_meta,
                                obs_pose_repr=obs_pose_rep)

                            obs_dict = dict_apply(obs_dict_np,
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))

                            result = policy.predict_action(obs_dict)
                            action = result['action_pred'][0].detach().to('cpu').numpy()

                            if 'delta_p' in result:
                                delta_p = result['delta_p'][0].detach().to('cpu').numpy()
                                delta_p = delta_p[0]
                                action[-1, :3] += delta_p * 0.01

                            # Time-based action filtering
                            this_target_poses = action

                            obs_timestamp = ft_ts[-1]
                            action_timestamps = (np.arange(len(action), dtype=np.float64)) * dt + obs_timestamp
                            action_exec_latency = config['control']['action_exec_latency']
                            curr_time = time.time()
                            is_new = action_timestamps > (curr_time + action_exec_latency)

                            if np.sum(is_new) == 0:
                                # Exceeded time budget; use the last action
                                this_target_poses = this_target_poses[[-1]]
                                next_step_idx = int(np.ceil((curr_time - obs_timestamp) / dt))
                                action_timestamp = obs_timestamp + (next_step_idx) * dt
                                action_timestamps = np.array([action_timestamp])
                            else:
                                this_target_poses = this_target_poses[is_new]
                                action_timestamps = action_timestamps[is_new]

                            if this_target_poses.shape[0] == 0:
                                this_target_poses = action[[-1]]
                                action_timestamps = np.array([obs_timestamp + dt])

                            selected_action = this_target_poses[0]
                            selected_timestamp = action_timestamps[0]

                            action_8d = convert_action_10d_to_8d(selected_action)
                            policy._last_action = action_8d.copy()

                            transformed_action = transform_pose(action_8d, current_ee_pose, ft_now)

                            gripper_state = 'keep'

                            # Accumulate pose: position += delta, rotation *= delta
                            this_target_pose = prev_target_pose.copy()
                            this_target_pose[:3] += transformed_action[:3]
                            transformed_rot = Rotation.from_quat(transformed_action[3:])
                            current_rot = Rotation.from_quat(prev_target_pose[3:])

                            new_rot = current_rot * transformed_rot
                            this_target_pose[3:] = new_rot.as_quat()

                            # Clamp pitch to prevent excessive tilt
                            target_euler = Rotation.from_quat(this_target_pose[3:]).as_euler('xyz', degrees=True)
                            if target_euler[1] < -35:
                                target_euler[1] = -35
                            new_quat = Rotation.from_euler('xyz', target_euler, degrees=True).as_quat()
                            new_quat = new_quat / np.linalg.norm(new_quat)
                            this_target_pose[3:] = new_quat

                            time.sleep(0.01)

                            prev_target_pose = this_target_pose.copy()
                            last_executed_action = action_8d.copy()

                            # Safety: minimum height guard
                            if this_target_pose[2] < 0.02:
                                this_target_pose[2] = 0.02

                            e = time.time()
                            cycle_time = e - s
                            hz = 1.0 / cycle_time

                            if iter_idx % int(frequency) == 0:
                                print(f"\r[Action Rate] {hz:.1f} Hz | Cycle: {cycle_time*1000:.1f}ms", end='', flush=True)

                            try:
                                data = {
                                    'target_pose': this_target_pose.tolist(),
                                    'gripper_command': gripper_state,
                                    'reset': False,
                                    'timestamp': selected_timestamp,
                                    'action_timestamps': action_timestamps.tolist()
                                }
                                message = json.dumps(data, separators=(',', ':'))
                                client_socket.sendall(message.encode('utf-8') + b'\n')
                            except (socket.error, BrokenPipeError) as e:
                                print(f"\rConnection error: {e}", end='', flush=True)
                                try:
                                    client_socket.close()
                                    client_socket = connect_to_server(server_ip, server_port)
                                except Exception:
                                    pass
                                continue

                            # Visualize side-by-side camera feeds
                            h1, w1 = img1_view.shape[:2]
                            h2, w2 = img2_view.shape[:2]
                            target_height = max(h1, h2)

                            if h1 < target_height:
                                pad_top = (target_height - h1) // 2
                                pad_bottom = target_height - h1 - pad_top
                                img1_padded = cv2.copyMakeBorder(img1_view, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                            else:
                                img1_padded = img1_view

                            if h2 < target_height:
                                pad_top = (target_height - h2) // 2
                                pad_bottom = target_height - h2 - pad_top
                                img2_padded = cv2.copyMakeBorder(img2_view, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                            else:
                                img2_padded = img2_view

                            vis_img = np.hstack([img1_padded, img2_padded])

                            display_scale = 0.6
                            h, w = vis_img.shape[:2]
                            new_h, new_w = int(h * display_scale), int(w * display_scale)
                            vis_img_resized = cv2.resize(vis_img, (new_w, new_h))

                            text_img = np.zeros((100, new_w, 3), dtype=np.uint8)
                            cv2.putText(text_img, f"Mode: {mode.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            cv2.putText(text_img, "Keys: t(teleop) p(policy) r(reset) q(quit)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                            combined_img = np.vstack([vis_img_resized, text_img])

                            win_name = 'Camera 1 | Camera 2'
                            cv2.imshow(win_name, combined_img)
                            cv2.waitKey(1)

                    cycle_end = cycle_start + dt
                    sleep_time = cycle_end - time.monotonic()
                    if sleep_time > 0:
                        precise_wait(cycle_end)

                    iter_idx += 1

            finally:
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
                cv2.destroyAllWindows()
                print("\nConnection closed.")


if __name__ == "__main__":
    main()
