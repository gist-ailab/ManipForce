
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
import socket
from utils.spacemouse_device import SpacemouseDevice as Spacemouse
from utils.precise_sleep import precise_wait
import sys
import termios
import tty
import select
import threading
from collections import deque
from utils.franka_api import FrankaAPI


class FTCollector:
    """Continuously polls force/torque vectors from FrankaAPI at a desired Hz and stores them in a ring buffer."""
    def __init__(self, api, rate_hz: float = 100., buf_len: int = 256, enable_ros=False):
        self.api = api
        self.rate = rate_hz
        # Sliding window buffer
        self.buf = deque(maxlen=buf_len)      # [(timestamp, np.array(6)), ...]
        # Full-session log — raw data
        self.full_ts = []
        self.full_ft_raw = []         # raw FT data
        self.full_ft_normalized = []  # EMA-normalized FT data
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._th = None
        self.zero_force = np.zeros(6, dtype=np.float32)  # zero-force calibration offset

        # Variables for FT filtering
        self.filter_alpha = 0.3  # Low-pass filter coefficient (0.1–0.5 recommended; smaller = smoother)
        self.filtered_ft = None  # Currently filtered FT data

        # ROS-related variables
        self.enable_ros = enable_ros
        self.ft_pub = None
        self.ft_filtered_pub = None
        if enable_ros:
            import rospy
            from std_msgs.msg import Float64MultiArray
            self.ft_pub = rospy.Publisher('/ft_sensor/raw', Float64MultiArray, queue_size=10)
            self.ft_filtered_pub = rospy.Publisher('/ft_sensor/filtered', Float64MultiArray, queue_size=10)
        
        # EMA normalizer (same parameters as used in the zarr pipeline)
        self.normalizer = RunningEMANormalizer(
            beta=0.95,          # same as zarr code
            warmup_steps=10,    # same as zarr code
            tanh_c=1.5,         # same as zarr code
            min_scale=1e-3      # same as zarr code
        )

    def _loop(self):
        period = 1.0 / self.rate
        nxt = time.perf_counter()
        last_normalize_time = time.time()
        normalize_interval = 1.0  # run EMA normalization every 1 second
        
        while not self._stop.is_set():
            ts = time.time()
            f = self.api.get_force_sync()      # REST API call
            t = self.api.get_torque_sync()
            if f is not None and t is not None:
                # Combine FT data
                ft_vec = np.concatenate([f, t])
                # Apply zero-force offset correction
                ft_vec = ft_vec - self.zero_force
                # Apply coordinate frame transformation
                ft_vec = convert_ft_axis(ft_vec)

                # Apply low-pass filter
                if self.filtered_ft is None:
                    self.filtered_ft = ft_vec.copy()
                else:
                    self.filtered_ft = self.filter_alpha * ft_vec + (1 - self.filter_alpha) * self.filtered_ft
                
                # Publish to ROS (raw data)
                if self.enable_ros and self.ft_pub:
                    try:
                        raw_msg = Float64MultiArray()
                        raw_msg.data = ft_vec.tolist()
                        self.ft_pub.publish(raw_msg)
                    except Exception as e:
                        pass  # Ignore ROS publish errors

                # Publish to ROS (filtered data)
                if self.enable_ros and self.ft_filtered_pub:
                    try:
                        filtered_msg = Float64MultiArray()
                        filtered_msg.data = self.filtered_ft.tolist()
                        self.ft_filtered_pub.publish(filtered_msg)
                    except Exception as e:
                        pass  # Ignore ROS publish errors

                with self._lock:
                    self.buf.append((ts, self.filtered_ft))  # store filtered data in ring buffer
                    # also append to full session log
                    self.full_ts.append(ts)
                    self.full_ft_raw.append(ft_vec)  # raw data stored separately

                # Periodically run EMA normalization (every 1 second)
                if ts - last_normalize_time >= normalize_interval:
                    with self._lock:
                        if len(self.full_ft_raw) > 0:
                            # Apply EMA normalization to the accumulated raw data
                            raw_data = np.array(self.full_ft_raw)
                            try:
                                normalized_data = self.normalizer.normalize(raw_data)
                                # Overwrite with the newly normalized data
                                self.full_ft_normalized = normalized_data.tolist()
                            except Exception as e:
                                # Continue on normalization failure
                                pass
                    last_normalize_time = ts
            
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
    
    def set_filter_alpha(self, alpha):
        """Set the low-pass filter strength (0.1–0.5 recommended; smaller = smoother)."""
        self.filter_alpha = np.clip(alpha, 0.01, 0.9)
        print(f"FT filter alpha set to: {self.filter_alpha:.2f}")

    def reset_filter(self):
        """Reset the filter state."""
        self.filtered_ft = None
        print("FT filter state reset.")

class RunningEMANormalizer:
    def __init__(self, dim=6, beta=0.95, eps=1e-6, warmup_steps=10, tanh_c=1.5, min_scale=1e-3):
        self.beta = beta
        self.eps = eps
        self.warmup_steps = warmup_steps
        self.tanh_c = tanh_c
        self.min_scale = min_scale
        
        self.ema_scale = np.zeros(dim) + eps
        self.f0 = None
        self.step_count = 0

    def normalize(self, ft_data):
        """Apply the same EMA normalization as used in the zarr pipeline."""
        if self.f0 is None:
            self.f0 = ft_data[0].copy()
        
        N = len(ft_data)
        normalized = np.zeros_like(ft_data)
        
        for t in range(N):
            f_c = ft_data[t] - self.f0
            r = np.abs(f_c)
            
            if self.step_count + t < self.warmup_steps:
                if self.step_count + t == 0:
                    self.ema_scale = r + self.eps
                else:
                    self.ema_scale = np.maximum(self.ema_scale, r)
            else:
                self.ema_scale = self.beta * self.ema_scale + (1 - self.beta) * r
            
            self.ema_scale = np.maximum(self.ema_scale, self.min_scale)
            r_norm = r / self.ema_scale
            mag = np.tanh(np.log1p(r_norm) / self.tanh_c)
            normalized[t] = np.sign(f_c) * mag
        
        self.step_count += N
        return normalized

def convert_ft_axis(ft):
    # ft: shape (N, 6) or (6,)
    # Coordinate transform: [x,y,z] -> [-y, x, -z]
    # Force:  [Fx, Fy, Fz]  -> [-Fy, Fx, -Fz]
    # Torque: [Tx, Ty, Tz]  -> [-Ty, Tx, -Tz]
    ft = np.asarray(ft)
    if ft.ndim == 1:
        return np.array([-ft[1], ft[0], -ft[2], -ft[4], ft[3], -ft[5]], dtype=ft.dtype)
    else:
        return np.stack([-ft[:,1], ft[:,0], -ft[:,2], -ft[:,4], ft[:,3], -ft[:,5]], axis=-1)

def save_ft_data(ft_collector, filename="teleop_ft_data.csv", save_normalized=True):
    """Save the FT data collected by FTCollector to CSV files."""
    ft_collector.stop()
    
    with ft_collector._lock:
        timestamps = ft_collector.full_ts.copy()
        raw_ft_data = ft_collector.full_ft_raw.copy()
        normalized_ft_data = ft_collector.full_ft_normalized.copy()
    
    if not timestamps or not raw_ft_data:
        print("No FT data collected.")
        return
    
    # Save raw data
    arr_ts = np.array(timestamps, dtype=np.float64).reshape(-1, 1)
    arr_ft_raw = np.array(raw_ft_data, dtype=np.float32)

    # Raw data filename
    base_name = filename.replace('.csv', '')
    raw_filename = f"{base_name}_raw.csv"
    
    csv_data_raw = np.hstack([arr_ts, arr_ft_raw])
    header = "timestamp,fx,fy,fz,tx,ty,tz"
    
    np.savetxt(raw_filename,
               csv_data_raw,
               delimiter=",",
               header=header,
               comments="")
    
    print(f"Raw data saved -> {raw_filename} ({csv_data_raw.shape[0]} samples)")

    # Save normalized data (optional)
    if save_normalized and normalized_ft_data:
        # Final normalization pass (to capture the most recent data)
        try:
            final_normalized = ft_collector.normalizer.normalize(arr_ft_raw)
            arr_ft_normalized = final_normalized.astype(np.float32)
            
            normalized_filename = f"{base_name}_normalized.csv"
            csv_data_normalized = np.hstack([arr_ts, arr_ft_normalized])
            
            np.savetxt(normalized_filename,
                       csv_data_normalized,
                       delimiter=",",
                       header=header,
                       comments="")
            
            print(f"Normalized data saved -> {normalized_filename} ({csv_data_normalized.shape[0]} samples)")
        except Exception as e:
            print(f"Failed to save normalized data: {e}")
    
    # Save normalized data under the base filename (for training)
    if save_normalized and normalized_ft_data:
        try:
            np.savetxt(filename,
                       csv_data_normalized,
                       delimiter=",",
                       header=header,
                       comments="")
            print(f"Training data saved -> {filename} (normalized)")
        except Exception as e:
            print(f"Failed to save training data: {e}")

def calibrate_ft_zero(ft_collector, num_samples=100):
    """Calibrate the FT sensor zero-force (residual force) offset."""
    print(f"[FT Calibration] Collecting {num_samples} samples for zero-force calibration... (ensure no external force is applied)")
    samples = []
    for i in range(num_samples):
        f = ft_collector.api.get_force_sync()
        t = ft_collector.api.get_torque_sync()
        if f is not None and t is not None:
            samples.append(np.concatenate([f, t]))
        time.sleep(0.01)
    if len(samples) == 0:
        print("[FT Calibration] Failed: no samples collected.")
        return
    zero_force = np.mean(samples, axis=0)
    ft_collector.zero_force = zero_force
    print(f"[FT Calibration] Done! zero-force: {zero_force.round(4)}")

def connect_to_server(host='localhost', port=5001, timeout=5):
    while True:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        client_socket.settimeout(timeout)
        try:
            client_socket.connect((host, port))
            print(f"Connected to server {host}:{port}.")
            return client_socket
        except (ConnectionRefusedError, socket.timeout):
            print(f"Connection to {host}:{port} failed. Retrying in 1s...")
            time.sleep(1)
            return None

def check_keyboard_input():
    """Check keyboard input in a non-blocking manner."""
    if select.select([sys.stdin], [], [], 0)[0]:
        key = sys.stdin.read(1)
        return key
    return None

@click.command()
@click.option('--frequency', '-f', default=50, type=float, help="Control frequency (Hz)")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Command latency (seconds)")
# @click.option('--server_ip', '-si', default='172.27.190.125', help="Server IP address")
@click.option('--server_ip', '-si', default='172.27.190.241', help="Server IP address")
@click.option('--server_port', '-sp', default=5001, type=int, help="Server port number")
@click.option('--save_ft', '-ft', is_flag=True, help="Collect and save FT sensor data")
@click.option('--enable_ros', '-ros', is_flag=True, help="Publish FT data to ROS")
@click.option('--ft_filter_alpha', '-fa', default=0.3, type=float, help="FT filter strength (0.1–0.5 recommended; smaller = smoother)")
def main(frequency, command_latency, server_ip, server_port, save_ft, enable_ros, ft_filter_alpha):
    dt = 1 / frequency
    max_pos_speed = 0.1  # m/s
    max_rot_speed = 0.3  # rad/s

    # Precomputed constants
    pos_scale = max_pos_speed * dt
    rot_scale = max_rot_speed * dt

    # Initial pose
    initial_pose = np.array([0.5, 0.0162628, 0.16830516, 0.9986493, 0.01434065, 0.04985237, -0.00294379])

    # === Initialize FT sensor (optional) ===
    franka_api = None
    ft_collector = None
    if save_ft or enable_ros:
        print("=" * 60)
        print("🤖 Initializing FT sensor...")
        try:
            franka_api = FrankaAPI(server_ip)
            ft_collector = FTCollector(franka_api, rate_hz=100, enable_ros=enable_ros)
            
            # Set FT filter strength
            ft_collector.set_filter_alpha(ft_filter_alpha)
            
            # Check current robot state
            current_pose = franka_api.get_pose_sync()
            current_gripper = franka_api.get_gripper_sync()
            print(f"current_pose: {current_pose}")
            print(f"current_gripper: {current_gripper:.3f}")
            
            # FT zero-force calibration
            print("\n⚠️  Starting FT sensor calibration.")
            print("Ensure no external force is applied to the robot, then press Enter...")
            input()
            calibrate_ft_zero(ft_collector, num_samples=100)
            
            # Start FT data collection
            ft_collector.start()
            print("✅ FT data collection started.")
            if enable_ros:
                print("📡 Publishing FT data to ROS topics:")
                print("   - /ft_sensor/raw:      raw FT data")
                print("   - /ft_sensor/filtered: low-pass filtered FT data")
            
        except Exception as e:
            print(f"❌ Error initializing FT sensor: {e}")
            print("Continuing without FT data collection.")
            save_ft = False
        print("=" * 60)
    
    # Connect to server
    client_socket = connect_to_server(server_ip, server_port)
    if client_socket is None:
        print("Initial connection failed. Exiting.")
        # Cleanup FT collector
        if ft_collector:
            ft_collector.stop()
        return

    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager) as sm:
            print("🎮 SpaceMouse teleoperation ready!")
            if save_ft:
                print("📊 FT data collection enabled — data will be saved to 'teleop_ft_data.csv' on exit.")
            if enable_ros:
                print("📡 ROS FT data publishing enabled.")
            if not save_ft and not enable_ros:
                print("💡 To collect FT data, use '--save_ft' or '-ft'.")
                print("💡 To publish via ROS, use '--enable_ros' or '-ros'.")

            print("\n=== Controls ===")
            print("SpaceMouse: 6-DOF position / orientation control")
            print("Button 0:   Toggle gripper (open / close)")
            print("Button 1:   Allow Z-axis movement")
            print("Keyboard q: Quit")
            print("Keyboard r: Reset to initial pose")
            if save_ft or enable_ros:
                print("Keyboard 1: Increase FT filter strength (smoother)")
                print("Keyboard 2: Decrease FT filter strength (more responsive)")
                print("Keyboard 0: Reset FT filter")
            print("=" * 60)
            
            target_pose = initial_pose.copy()
            current_pos = initial_pose[:3].copy()
            current_rot = st.Rotation.from_quat(initial_pose[3:])

            # Gripper state tracking
            gripper_state = 'open'
            last_btn0_state = False
            last_command_time = time.monotonic()

            t_start = time.monotonic()
            iter_idx = 0

            # Save terminal settings
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                # Switch terminal to raw mode
                tty.setraw(sys.stdin.fileno())
                
                while True:
                    cycle_start = time.monotonic()
                    
                    # Read SpaceMouse state
                    sm_state = sm.get_motion_state_transformed()
                    btn0 = sm.is_button_pressed(0)
                    btn1 = sm.is_button_pressed(1)

                    # Toggle gripper on button-press edge
                    if btn0 and not last_btn0_state:
                        gripper_state = 'close' if gripper_state == 'open' else 'open'
                        last_command_time = cycle_start
                    last_btn0_state = btn0

                    # Convert and scale inputs
                    dpos = sm_state[:3] * pos_scale
                    drot_xyz = sm_state[3:] * rot_scale

                    # Check keyboard input
                    key = check_keyboard_input()
                    if key:
                        if key == 'q':
                            print("\nExiting.")
                            # Save FT data
                            if save_ft and ft_collector:
                                print("Saving FT data...")
                                save_ft_data(ft_collector, filename="teleop_ft_data.csv")
                            break
                        elif key == 'r':
                            print("\nResetting to initial pose.")
                            
                            current_pos = initial_pose[:3].copy()
                            current_rot = st.Rotation.from_quat(initial_pose[3:])
                            
                            # Target pose [x, y, z, qx, qy, qz, qw] [1,0,0,0]
                            target_pose[:3] = current_pos
                            target_pose[3:] = current_rot.as_quat()
                            
                            
                            # Send reset command to server
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
                                print(f"\nFailed to send reset command: {e}")
                            continue
                        elif key == '1' and (save_ft or enable_ros):
                            # Increase FT filter alpha (smoother)
                            new_alpha = max(0.01, ft_collector.filter_alpha - 0.05)
                            ft_collector.set_filter_alpha(new_alpha)
                            continue
                        elif key == '2' and (save_ft or enable_ros):
                            # Decrease FT filter alpha (more responsive)
                            new_alpha = min(0.9, ft_collector.filter_alpha + 0.05)
                            ft_collector.set_filter_alpha(new_alpha)
                            continue
                        elif key == '0' and (save_ft or enable_ros):
                            # Reset FT filter
                            ft_collector.reset_filter()
                            continue
                    
                    current_pos += dpos
                    # Always apply rotation update
                    if np.any(drot_xyz != 0):
                        drot = st.Rotation.from_euler('xyz', drot_xyz)
                        current_rot = drot * current_rot

                    # Update target pose
                    target_pose[:3] = current_pos
                    target_pose[3:] = current_rot.as_quat()

                    # Send command to server
                    try:
                        data = {
                            'target_pose': target_pose.tolist(),
                            'timestamp': cycle_start,
                            'gripper_command': gripper_state,
                            'reset': False  # standard command
                        }
                        message = json.dumps(data, separators=(',', ':'))
                        client_socket.sendall(message.encode('utf-8') + b'\n')
                    except (socket.error, BrokenPipeError) as e:
                        print(f"Server connection error: {e}. Reconnecting...")
                        client_socket.close()
                        client_socket = connect_to_server(server_ip, server_port)
                        if client_socket is None:
                            print("Reconnection failed. Keeping loop alive.")
                            time.sleep(0.1)
                            continue

                    # Check quit condition (every 10 iterations)
                    if iter_idx % 10 == 0:
                        if cv2.waitKey(1) == ord('q'):
                            break

                    # Frequency debug print (every 100 iterations)
                    if iter_idx % 100 == 0 and iter_idx > 0:
                        elapsed = time.monotonic() - t_start
                        status_msg = f"Actual frequency: {iter_idx / elapsed:.2f} Hz"
                        if (save_ft or enable_ros) and ft_collector:
                            with ft_collector._lock:
                                ft_samples = len(ft_collector.full_ft_raw)
                                normalized_samples = len(ft_collector.full_ft_normalized)
                            status_msg += f" | FT samples: {ft_samples} (normalized: {normalized_samples})"
                            if enable_ros:
                                status_msg += f" | Filter alpha: {ft_collector.filter_alpha:.2f}"
                        print(status_msg)

                    # Precise timing control
                    cycle_end = cycle_start + dt
                    sleep_time = cycle_end - time.monotonic()
                    if sleep_time > 0:
                        precise_wait(cycle_end)

                    iter_idx += 1

            finally:
                # Final FT data save and cleanup
                if save_ft and ft_collector:
                    print("\nSaving final FT data...")
                    save_ft_data(ft_collector, filename="teleop_ft_data.csv")

                # Stop FT collector
                if ft_collector:
                    ft_collector.stop()

                # Restore terminal settings
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                client_socket.close()
                print("\nServer connection closed.")

if __name__ == "__main__":
    main()