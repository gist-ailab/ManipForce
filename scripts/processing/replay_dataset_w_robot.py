
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import time
import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
import dill
import hydra.utils
import os
import json
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation  # added
import socket
import click
from typing import Optional, Dict, Any
from collections import deque
import threading

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from utils.real_inference_util import (get_real_gumi_obs_dict,
                                       get_real_gumi_action)
from utils.franka_api import FrankaAPI

# Global variables
prev_frames = []
prev_timestamps = []
current_img_idx = 0
BASE_DIR = None


def transform_pose(pos, quat, current_ee_pose):
    """
    Transform position and quaternion into a new coordinate frame.
    pose: [px, py, pz, qw, qx, qy, qz]
    return: transformed [px, py, pz, qw, qx, qy, qz]
    """

    R_mat_pos = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1]
        ])

    R_mat_rot = np.array([
            [0, 1, 0],
            [-1, 0, 0],
            [0, 0, 1]
        ])
    
    position_scale = 1
    rotation_scale = 1
    
    rel_pos = R_mat_pos @ pos * position_scale
    R_orig = Rotation.from_quat(quat).as_matrix()
    R_new  = R_mat_rot @ R_orig @ R_mat_rot.T
    rel_quat = Rotation.from_matrix(R_new).as_quat()

    # Apply the current EE orientation
    current_quat = current_ee_pose[3:]  # [w, x, y, z]
    current_quat = np.roll(current_quat, -1)  # convert to [x, y, z, w]
    current_rot = Rotation.from_quat(current_quat)
    world_rel_pos = current_rot.inv().apply(rel_pos)

    return np.concatenate([world_rel_pos, rel_quat])


def convert_action_10d_to_8d(action_10d):
    position = action_10d[..., :3]
    rotation_6d = action_10d[..., 3:9]
    gripper = action_10d[..., 9:10]

    first_col = rotation_6d[..., :3]
    second_col = rotation_6d[..., 3:6]

    first_col = first_col /  np.linalg.norm(first_col, axis=-1, keepdims=True)
    second_col = second_col - np.sum(first_col * second_col, axis=-1, keepdims=True) * first_col
    second_col = second_col / np.linalg.norm(second_col, axis=-1, keepdims=True)
    third_col = np.cross(first_col, second_col)

    rotation_matrix = np.stack([first_col, second_col, third_col], axis=-1)
    from scipy.spatial.transform import Rotation as R
    quat = R.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w] order
    # Combine into an 8D action vector
    action_8d = np.concatenate([
        position,    # position (3)
        quat,        # quaternion (4)
        gripper      # gripper (1)
    ], axis=-1)
    
    return action_8d


def get_obs():
    global prev_frames, prev_timestamps, current_img_idx, BASE_DIR
    n_obs_steps = 1
    if BASE_DIR is None:
        raise RuntimeError("BASE_DIR is not set. Pass --base_dir to main().")
    base_dir = BASE_DIR
    image_dir = os.path.join(base_dir, 'images/handeye')
    pose_tracking_dir = os.path.join(base_dir, 'images/pose_tracking')
    
    # Load GT data
    if not hasattr(get_obs, 'action_gt_data'):
        with open(os.path.join(base_dir, 'pose_data.json'), 'r') as f:
            get_obs.action_gt_data = json.load(f)
            print(f"Total GT data entries: {len(get_obs.action_gt_data)}")

    # Get list of image files
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    pose_tracking_files = sorted([
        f for f in os.listdir(pose_tracking_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if current_img_idx >= len(image_files):
        return None, None, None
    
    # Initialise buffer if empty
    if len(prev_frames) == 0:
        for _ in range(n_obs_steps):
            if current_img_idx >= len(image_files):
                return None, None, None

            img_path = os.path.join(image_dir, image_files[current_img_idx])
            frame = cv2.imread(img_path)
            if frame is None:
                raise RuntimeError(f"Cannot read image: {img_path}")
            
            prev_frames.append(frame)
            prev_timestamps.append(time.time())
            current_img_idx += 1
    
    # Load current image
    img_path = os.path.join(image_dir, image_files[current_img_idx])
    frame = cv2.imread(img_path)

    # Load pose-tracking image
    pose_tracking_path = os.path.join(pose_tracking_dir, pose_tracking_files[current_img_idx])
    pose_tracking_frame = cv2.imread(pose_tracking_path)
    if pose_tracking_frame is None:
        raise RuntimeError(f"Cannot read pose-tracking image: {pose_tracking_path}")

    # Find matching GT action
    current_img_file = image_files[current_img_idx]
    current_gt = None
    for data in get_obs.action_gt_data:
        if data['image_file'] == current_img_file:
            current_gt = data['action']
            break

    # Update ring buffer
    prev_frames.append(frame)
    prev_timestamps.append(time.time())
    prev_frames.pop(0)
    prev_timestamps.pop(0)

    # Advance image index
    current_img_idx += 1
    
    obs = {
        'img': np.stack(prev_frames, axis=0),
        'agent_pos': np.zeros((n_obs_steps, 7)),
        'timestamp': np.array(prev_timestamps)
    }
    
    return obs, current_gt, pose_tracking_frame



def connect_to_server(ip, port):
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((ip, port))
        print(f"Connected to server: {ip}:{port}")
        return client_socket
    except Exception as e:
        print(f"Server connection failed: {e}")
        return None

def measure_zero_force(api, num_samples=100, delay=0.01):
    """Read num_samples FT samples and return their mean as the zero-force baseline."""
    print(f"[ZeroForce] Measuring... ({num_samples} samples)")
    ft_samples = []
    for _ in range(num_samples):
        f = api.get_force_sync()
        t = api.get_torque_sync()
        if f is not None and t is not None:
            ft_samples.append(np.concatenate([f, t]))
        time.sleep(delay)
    zero_force = np.mean(ft_samples, axis=0)
    print(f"[ZeroForce] Done! zero_force = {zero_force}")
    return zero_force

# FTCollector class
class FTCollector:
    """Continuously polls force/torque vectors from FrankaAPI at a desired Hz and stores them."""
    def __init__(self, api, rate_hz: float = 100., zero_force=None):
        self.api = api
        self.rate = rate_hz
        self.full_ts = []  # all timestamps
        self.full_ft = []  # all FT data
        self.zero_force = zero_force  # zero-force offset supplied externally
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._th = None

    def _loop(self):
        period = 1.0 / self.rate
        nxt = time.perf_counter()
        while not self._stop.is_set():
            ts = time.time()
            f = self.api.get_force_sync()
            t = self.api.get_torque_sync()
            if f is not None and t is not None:
                ft_vec = np.concatenate([f, t])
                # Apply zero-force correction
                if self.zero_force is not None:
                    ft_vec = ft_vec - self.zero_force
                with self._lock:
                    self.full_ts.append(ts)
                    self.full_ft.append(ft_vec)
            nxt += period
            time.sleep(max(0, nxt - time.perf_counter()))

    def start(self):
        if self._th is None or not self._th.is_alive():
            self._stop.clear()
            self._th = threading.Thread(target=self._loop, daemon=True)
            self._th.start()
            print(f"[FTCollector] started {self.rate} Hz polling")

    def stop(self):
        self._stop.set()
        if self._th: self._th.join(timeout=1)

def save_ft_data(ft_collector, filename="ft_trace.csv"):
    """Save all (timestamp, [fx,fy,fz,tx,ty,tz]) entries collected by FTCollector to a CSV file."""
    ft_collector.stop()
    
    with ft_collector._lock:
        data = list(zip(ft_collector.full_ts, ft_collector.full_ft))
    
    if not data:
        print("No FT data collected.")
        return
    
    ts, ft = zip(*data)
    arr_ts = np.array(ts, dtype=np.float64).reshape(-1, 1)
    arr_ft = np.vstack(ft).astype(np.float32)


    csv_data = np.hstack([arr_ts, arr_ft])
    header = "timestamp,fx,fy,fz,tx,ty,tz"
    
    np.savetxt(filename,
               csv_data,
               delimiter=",",
               header=header,
               comments="")
    
    print(f"Saved -> {filename} ({csv_data.shape[0]} samples)")

@click.command()
@click.option('--server_ip', '-si', default='172.27.190.125', help="Server IP address")
@click.option('--server_port', '-sp', default=5001, type=int, help="Server port number")
@click.option('--operator', '-op', is_flag=True, help="Manual mode (advance one step at a time with Enter)")
@click.option('--base_dir', '-bd', default='/home/ailab-2204/Workspace/gail-umi/data/Test_Pose/episode_1', help='Dataset episode path (containing images/*)')
def main(server_ip, server_port, operator, base_dir):
    # Inject configuration path
    global BASE_DIR
    BASE_DIR = base_dir

    # Connect to server
    client_socket = connect_to_server(server_ip, server_port)
    if client_socket is None:
        print("Initial connection failed. Exiting.")
        return
    franka_api = FrankaAPI(server_ip)
    
    # === 1) Measure zero-force baseline ===
    zero_force = measure_zero_force(franka_api, num_samples=100, delay=0.01)

    # === 2) Pass zero_force to FTCollector ===
    ft_collector = FTCollector(franka_api, rate_hz=200, zero_force=zero_force)
    ft_collector.start()

    pose = franka_api.get_pose_sync()
    print(f"Current pose: {pose}")

    # Set initial pose
    initial_pose = np.array([0.56, -0.0,  0.17,  0.99959016, -0.01183819,  0.02564274, -0.00467247])

    # Move to initial pose first
    print("Moving to initial pose. Press Enter to begin.")
    input()
    
    try:
        # Send initial pose command
        data = {
            'target_pose': initial_pose.tolist(),
            'timestamp': time.time(),
            'gripper_command': False,
            'reset': False
        }
        message = json.dumps(data, separators=(',', ':'))
        client_socket.sendall(message.encode('utf-8') + b'\n')
        print(f"Moving to initial pose: {initial_pose[:3].round(3)}")
        time.sleep(2.0)

        print("Reached initial pose. Focus the image window and press Enter to continue.")
        cv2.namedWindow('Pose Tracking')
        input()  # Press Enter here to start playback
        
        prev_target_pose = initial_pose.copy()
        current_idx = 0
        
        while True:
            obs, gt_action, pose_tracking_frame = get_obs()
            
            if obs is None:
                print("\nDataset playback complete.")
                # Save FT data
                save_ft_data(ft_collector, filename="replay_ft_data.csv")
                break
            
            if gt_action is not None:
                # Get current end-effector pose
                current_ee_pose = franka_api.get_pose_sync()

                # GT actions for current and next frames
                current_action = gt_action
                next_action = get_obs.action_gt_data[current_idx + 1]['action'] if current_idx + 1 < len(get_obs.action_gt_data) else None
                
                if next_action is not None:
                    # Relative pose for the current frame
                    current_rel_pos = np.array(current_action['relative_position'])
                    current_rel_quat = np.array(current_action['relative_orientation'])
                    current_transformed = transform_pose(current_rel_pos, current_rel_quat, current_ee_pose)

                    # Relative pose for the next frame
                    next_rel_pos = np.array(next_action['relative_position'])
                    next_rel_quat = np.array(next_action['relative_orientation'])
                    next_transformed = transform_pose(next_rel_pos, next_rel_quat, current_ee_pose)

                    # Execute the current frame
                    this_target_pose = prev_target_pose.copy()
                    this_target_pose[:3] += current_transformed[:3]
                    transformed_rot = Rotation.from_quat(current_transformed[3:])
                    current_rot = Rotation.from_quat(prev_target_pose[3:])
                    new_rot = current_rot * transformed_rot
                    this_target_pose[3:] = new_rot.as_quat()
                    
                    # Send command to server
                    data = {
                        'target_pose': this_target_pose.tolist(),
                        'timestamp': time.time(),
                        'gripper_command': False,
                        'reset': False
                    }
                    message = json.dumps(data, separators=(',', ':'))
                    client_socket.sendall(message.encode('utf-8') + b'\n')
                    
                    # Print status and display image
                    print(f"Replay Step {current_idx} | Pos: {this_target_pose[:3].round(3)} | Quat: {this_target_pose[3:].round(3)}")
                    cv2.imshow('Pose Tracking', cv2.resize(pose_tracking_frame, (0,0), fx=0.5, fy=0.5))
                    
                    should_quit = False
                    
                    if operator:
                        # Manual mode: wait for Enter key (step by step)
                        print("Press Enter for the next step, or 'q' to quit.")
                        user_input = input().strip().lower()
                        if user_input == 'q':
                            print("\nUser requested quit.")
                            should_quit = True
                    else:
                        # Auto mode: play GT actions as-is (send to server)
                        # Update pose-tracking image
                        cv2.imshow('Pose Tracking', cv2.resize(pose_tracking_frame, (0,0), fx=0.5, fy=0.5))
                        # Speed control: play at 10 Hz (0.1s intervals)
                        time.sleep(0.1)
                        key = cv2.waitKey(1) & 0xFF  # wait 1ms
                        if key == ord('q'):
                            print("\nUser requested quit.")
                            should_quit = True
                    
                    if should_quit:
                        break
                    
                    prev_target_pose = this_target_pose.copy()
                    current_idx += 1

    finally:
        # Stop FT collection and save data
        save_ft_data(ft_collector, filename="replay_ft_data.csv")
        cv2.destroyAllWindows()
        if client_socket:
            client_socket.close()

if __name__ == "__main__":
    main()