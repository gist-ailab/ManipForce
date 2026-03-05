import os
import sys

# Suppress Qt and OpenCV warnings
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts=false;*.debug=false;driver.usb=false"
os.environ["QT_QPA_PLATFORM_NODEBUG"] = "1"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import cv2
import os
from datetime import datetime
import time
import numpy as np
import queue
import threading
from collections import deque
import pyrealsense2 as rs
from ahrs.filters import Madgwick
from tqdm import trange, tqdm
import argparse
import sys

from utils.rs_capture import RSCapture, DJICapture
from utils.ft_capture import AidinFTSensorUDP
from utils.gravity_compensation_utils import GravityCompensator

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data', help='Base directory path for saving data')
parser.add_argument('--add_cam', action='store_true', help='Enable additional camera stream')
parser.add_argument('--dji_device', type=str, default=None, help='/dev/video path for the DJI action camera')
parser.add_argument('--ft_ip', type=str, default='172.27.190.4', help='F/T sensor UDP IP address')
parser.add_argument('--ft_port', type=int, default=50000, help='F/T sensor UDP port number')
args = parser.parse_args()
base_save_path = args.data_path
saving_images = False
episode_num = 1
handeye_save_path = None
pose_tracking_save_path = None
additional_cam_save_path = None
ft_data_save_path = None

use_additional_cam = args.add_cam

# Thread control flags
stop_ft_thread = False
stop_image_save_thread = False
ft_thread = None
image_save_thread = None
ft_csv_file = None

# Image save queue
image_queue = queue.Queue(maxsize=300)

# Dummy frames used when cameras are unavailable
DUMMY_HAND_FRAME = np.zeros((800, 1280, 3), dtype=np.uint8)
DUMMY_ADDITIONAL_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)

# Frequency monitoring variables
MONITOR_INTERVAL = 10.0  # seconds
image_count = 0
last_image_time = None
ft_count = 0
last_ft_time = None
ft_count_lock = threading.Lock()
image_capture_count = 0
image_save_count = 0
capture_count_lock = threading.Lock()

# Episode progress tracking
episode_start_time = None
episode_image_count = 0
episode_ft_count = 0
episode_progress_bar = None
progress_lock = threading.Lock()

# Image capture rate
FRAME_RATE = 30  # Hz
FRAME_INTERVAL = 1.0 / FRAME_RATE

# F/T data collection rate
FT_RATE = 200  # Hz
FT_INTERVAL = 1.0 / FT_RATE
ft_dummy_mode = False

os.makedirs(base_save_path, exist_ok=True)

def find_last_episode_number():
    """Find and return the last episode number from existing episode directories."""
    if not os.path.exists(base_save_path):
        return 0
    
    episode_dirs = []
    for item in os.listdir(base_save_path):
        if os.path.isdir(os.path.join(base_save_path, item)) and item.startswith('episode_'):
            try:
                episode_num = int(item.split('_')[1])
                episode_dirs.append(episode_num)
            except (ValueError, IndexError):
                continue
    
    if not episode_dirs:
        return 0
    
    return max(episode_dirs)

episode_num = find_last_episode_number() + 1
print(f"Last episode: {episode_num - 1}. New episode starts from {episode_num}.")

# Initialize gravity compensator
gravity_compensator = GravityCompensator(
    mass_for_x=0.58,
    mass_for_y=0.53, 
    mass_for_z=0.7,
    com_ft=np.array([0.01, 0.01, 0.03]),
    g_const=9.81
)

def update_progress_bar():
    """Update the episode progress bar display."""
    global episode_progress_bar, episode_start_time, episode_image_count, episode_ft_count
    
    if episode_progress_bar is None or episode_start_time is None:
        return
    
    current_time = time.time()
    elapsed_time = current_time - episode_start_time
    
    progress_str = f"📸 {episode_image_count} | 🔧 {episode_ft_count} | ⏱️ {elapsed_time:.1f}s"
    if ft_dummy_mode:
        progress_str += " (FT:DUMMY)"
    
    target_images = 1000
    progress = min(episode_image_count, target_images)
    
    episode_progress_bar.total = target_images
    episode_progress_bar.n = progress
    episode_progress_bar.set_description(progress_str)
    episode_progress_bar.refresh()

def print_episode_summary():
    """Print a summary of collected data after each episode."""
    global episode_start_time, episode_image_count, episode_ft_count, ft_dummy_mode
    
    if episode_start_time is None:
        return
    
    total_time = time.time() - episode_start_time
    image_hz = episode_image_count / total_time if total_time > 0 else 0
    ft_hz = episode_ft_count / total_time if total_time > 0 else 0
    
    # Count actual saved files
    save_counts = {}
    current_ep_dir = f"{base_save_path}/episode_{episode_num}"
    for cam in ['handeye', 'pose_tracking', 'additional_cam']:
        path = os.path.join(current_ep_dir, "images", cam)
        if os.path.exists(path):
            save_counts[cam] = len([f for f in os.listdir(path) if f.endswith('.jpg')])
        else:
            save_counts[cam] = 0

    print("\n" + "="*60)
    print(f"📊 Episode {episode_num} complete! {'[FT DUMMY]' if ft_dummy_mode else ''}")
    print("="*60)
    print(f"⏱️  Total collection time: {total_time:.1f}s")
    print(f"📸 Images captured (base): {episode_image_count} ({image_hz:.1f} Hz)")
    print(f"💾 Files saved:")
    print(f"   - Handeye: {save_counts['handeye']}")
    print(f"   - DJI (Pose): {save_counts['pose_tracking']}")
    if use_additional_cam:
        print(f"   - Additional: {save_counts['additional_cam']}")
    print(f"🔧 F/T samples: {episode_ft_count} ({ft_hz:.1f} Hz)")
    print(f"📁 Saved to: {current_ep_dir}")
    print("="*60 + "\n")

def start_new_episode():
    """Create a new episode directory and initialize data files."""
    global handeye_save_path, pose_tracking_save_path, additional_cam_save_path, ft_data_save_path, episode_num, ft_csv_file
    global image_count, last_image_time, ft_count, last_ft_time, image_capture_count, image_save_count
    global episode_start_time, episode_image_count, episode_ft_count, episode_progress_bar
    
    image_count = 0
    last_image_time = time.time()
    ft_count = 0
    last_ft_time = time.time()
    image_capture_count = 0
    image_save_count = 0
    
    episode_start_time = time.time()
    episode_image_count = 0
    episode_ft_count = 0
    
    if episode_progress_bar is not None:
        episode_progress_bar.close()
    
    desc = "Collecting data"
    if ft_dummy_mode:
        desc += " (FT:DUMMY)"
        
    episode_progress_bar = tqdm(total=1800, desc=desc, 
                               bar_format='{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]', 
                               ncols=85, leave=True)
    
    episode_dir = os.path.join(base_save_path, f'episode_{episode_num}')
    
    if os.path.exists(episode_dir):
        print(f"Warning: episode_{episode_num} directory already exists. Overwriting.")
        import shutil
        shutil.rmtree(episode_dir)
    
    handeye_save_path = os.path.join(episode_dir, 'images', 'handeye')
    pose_tracking_save_path = os.path.join(episode_dir, 'images', 'pose_tracking')
    additional_cam_save_path = os.path.join(episode_dir, 'images', 'additional_cam')
    ft_data_save_path = os.path.join(episode_dir, 'ft_data')
    
    os.makedirs(handeye_save_path, exist_ok=True)
    os.makedirs(pose_tracking_save_path, exist_ok=True)
    os.makedirs(additional_cam_save_path, exist_ok=True)
    os.makedirs(ft_data_save_path, exist_ok=True)
    
    ft_csv_path = os.path.join(episode_dir, f'ft_data_episode_{episode_num}.csv')
    ft_csv_file = open(ft_csv_path, 'w')
    ft_csv_file.write("timestamp,force_x,force_y,force_z,torque_x,torque_y,torque_z\n")
    
    print(f"Episode save paths: handeye={handeye_save_path}, pose_tracking={pose_tracking_save_path}, additional_cam={additional_cam_save_path}")
    print(f"F/T CSV: {ft_csv_path}")

def read_latest_imu(pipe):
    """
    Drain all buffered frames from a RealSense pipeline and return
    the latest accel + gyro motion frame pair as numpy vectors.
    Returns None if no new frame is available.
    """
    frames = pipe.poll_for_frames()
    if not frames:
        return None

    while True:
        more = pipe.poll_for_frames()
        if not more:
            break
        frames = more  

    accel = frames.first_or_default(rs.stream.accel)
    gyro  = frames.first_or_default(rs.stream.gyro)
    if accel is None or gyro is None:
        return None 

    a = accel.as_motion_frame().get_motion_data()
    g = gyro .as_motion_frame().get_motion_data()
    acc_vec  = np.array([a.x, a.y, a.z], dtype=float)
    gyro_vec = np.array([g.x, g.y, g.z], dtype=float)
    return acc_vec, gyro_vec

def check_ft_frequency():
    """Check and print the F/T data collection frequency."""
    global ft_count, last_ft_time
    
    current_time = time.time()
    elapsed = current_time - last_ft_time
    
    if elapsed >= MONITOR_INTERVAL:
        with ft_count_lock:
            local_ft_count = ft_count
            ft_count = 0
        
        ft_frequency = local_ft_count / elapsed
        print(f"F/T data rate: {ft_frequency:.2f} Hz (target: {FT_RATE} Hz) — {local_ft_count} samples / {elapsed:.2f}s")
        last_ft_time = current_time
        return True
    
    return False

def check_image_frequency():
    """Check and print the image capture frequency."""
    global image_count, last_image_time, image_capture_count, image_save_count
    
    current_time = time.time()
    elapsed = current_time - last_image_time
    
    if elapsed >= MONITOR_INTERVAL:
        local_image_count = image_count
        image_count = 0
        
        with capture_count_lock:
            local_capture_count = image_capture_count
            local_save_count = image_save_count
        
        image_frequency = local_image_count / elapsed
        print(f"Image rate: {image_frequency:.2f} Hz (target: {FRAME_RATE} Hz) — {local_image_count} frames / {elapsed:.2f}s")
        print(f"Capture/Save status: captured={local_capture_count}, saved={local_save_count}, queue={image_queue.qsize()}")
        last_image_time = current_time
        return True
    
    return False

def format_timestamp_from_float(timestamp_float):
    """Convert a float timestamp (time.time()) to a save-friendly string."""
    dt = datetime.fromtimestamp(timestamp_float)
    return dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]

def image_save_thread_func():
    """Thread function that saves individual frames to disk."""
    global stop_image_save_thread, image_save_count, episode_image_count
    
    print("Image save thread started")
    total_save_count = 0
    start_time = time.time()
    
    jpeg_params = {
        'handeye': [cv2.IMWRITE_JPEG_QUALITY, 90],
        'pose_tracking': [cv2.IMWRITE_JPEG_QUALITY, 100],
        'additional_cam': [cv2.IMWRITE_JPEG_QUALITY, 90]
    }
    
    while not stop_image_save_thread or not image_queue.empty():
        try:
            item = image_queue.get(block=True, timeout=0.1)
            if item is None:
                image_queue.task_done()
                break
                
            subfolder_name, timestamp_str, frame = item
            
            if subfolder_name == 'handeye':
                save_dir = handeye_save_path
            elif subfolder_name == 'pose_tracking':
                save_dir = pose_tracking_save_path
            else:
                save_dir = additional_cam_save_path

            if save_dir is None:
                image_queue.task_done()
                continue

            save_path = os.path.join(save_dir, f"{timestamp_str}.jpg")
            cv2.imwrite(save_path, frame, jpeg_params.get(subfolder_name, [cv2.IMWRITE_JPEG_QUALITY, 90]))
            
            with capture_count_lock:
                image_save_count += 1
            if subfolder_name == 'handeye':  # Use handeye as reference for progress
                with progress_lock:
                    episode_image_count += 1
                    update_progress_bar()
            
            total_save_count += 1
            image_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Image save error: {e}")
    
    elapsed = time.time() - start_time
    print(f"Image save thread done: {total_save_count} saved, avg {total_save_count/elapsed:.1f} Hz")

def ft_collection_thread(ft_reader, imu_pipe, gravity_compensator):
    """Thread function that independently collects F/T sensor data."""
    global stop_ft_thread, ft_csv_file, ft_count, episode_ft_count, ft_dummy_mode
    
    if ft_dummy_mode:
        print("F/T dummy mode active — saving zeros.")
    else:
        print("F/T collection thread started (target: 200 Hz)")
    
    # Flush stale data from the sensor buffer
    if not ft_dummy_mode:
        print("Flushing F/T sensor buffer...")
        for _ in range(50):
            try:
                ft_reader.get_frame(timeout=0.001)
            except:
                break
    
    total_ft_count = 0
    start_collection_time = time.time()
    
    try:
        while not stop_ft_thread:
            if ft_dummy_mode:
                time.sleep(FT_INTERVAL)
                compensated_force = np.zeros(3)
                compensated_torque = np.zeros(3)
            else:
                if imu_pipe is not None:
                    gravity_compensator.update_imu(imu_pipe)

                try:
                    ts, f_raw, t_raw = ft_reader.get_frame(timeout=0.001)
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"F/T read error: {str(e)}")
                    continue

                forces_filt, torques_filt = gravity_compensator.process_ft_data(f_raw, t_raw)
                compensated_force, compensated_torque = gravity_compensator.compensate_gravity(
                    forces_filt, torques_filt, gravity_compensation_on=True
                )

            ft_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            if ft_csv_file is not None:
                ft_csv_file.write(f"{ft_timestamp}," +
                                  f"{compensated_force[0]},{compensated_force[1]},{compensated_force[2]}," +
                                  f"{compensated_torque[0]},{compensated_torque[1]},{compensated_torque[2]}\n")
                ft_csv_file.flush()

            with ft_count_lock:
                ft_count += 1
                total_ft_count += 1
            
            with progress_lock:
                episode_ft_count += 1
                update_progress_bar()

    finally:
        if not ft_dummy_mode:
            print("F/T thread shutting down — draining remaining data")
            while True:
                try:
                    ts, f_raw, t_raw = ft_reader.get_frame(timeout=0.001)
                    
                    forces_filt, torques_filt = gravity_compensator.process_ft_data(f_raw, t_raw)
                    compensated_force, compensated_torque = gravity_compensator.compensate_gravity(
                        forces_filt, torques_filt, gravity_compensation_on=True
                    )

                    ft_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    if ft_csv_file is not None:
                        ft_csv_file.write(f"{ft_timestamp}," +
                                          f"{compensated_force[0]},{compensated_force[1]},{compensated_force[2]}," +
                                          f"{compensated_torque[0]},{compensated_torque[1]},{compensated_torque[2]}\n")
                        ft_csv_file.flush()

                    with ft_count_lock:
                        ft_count += 1
                        total_ft_count += 1
                    
                    with progress_lock:
                        episode_ft_count += 1
                        update_progress_bar()
                except:
                    break
        print(f"F/T thread done — {total_ft_count} samples saved")
    
    total_time = time.time() - start_collection_time
    avg_frequency = total_ft_count / total_time if total_time > 0 else 0
    print(f"F/T thread exit: {total_ft_count} samples, avg {avg_frequency:.2f} Hz")

print("Initializing cameras and sensors...")

# Realsense D455i — handeye camera
handeye_cam = None
try:
    handeye_cam = RSCapture(
        name='wrist_1',
        serial_number='241122306040',
        dim=(1280, 800),
        fps=30,
        depth=False
    )
except Exception as e:
    print(f"[WARNING] Handeye camera init failed, using dummy frames: {e}")

# DJI Action Cam — pose tracking camera
dji_cam = None
try:
    dji_cam = DJICapture(
        name='pose_tracking_cam',
        device=args.dji_device,
        dim=(1280, 720),
        fps=30,
        zero_config=False
    )
except Exception as e:
    print(f"\n[ERROR] DJI Action Cam init failed: {e}")
    print("DJI camera may not be connected or is in use by another process.")
    print("Continuing with dummy frames.\n")

# Optional additional RealSense camera
additional_cam = None
if use_additional_cam:
    additional_cam = RSCapture(
        name='additional_camera',
        serial_number='427622273372',
        dim=(640, 480),
        fps=30,
        depth=False
    )

# Start F/T sensor (UDP)
ft_reader = AidinFTSensorUDP(args.ft_ip, args.ft_port)
ft_reader.start()

# Test F/T sensor connectivity
print("Testing F/T sensor connection...")
try:
    ft_reader.get_frame(timeout=1.0)
    print("F/T sensor connected!")
    ft_dummy_mode = False
except queue.Empty:
    print("\n" + "!"*40)
    print("[WARNING] No data received from F/T sensor.")
    print("!"*40)
    user_choice = input("F/T sensor not connected. Continue in dummy mode? (y/n): ").strip().lower()
    if user_choice == 'y':
        print(">> Proceeding in F/T dummy mode (data will be zeros).")
        ft_dummy_mode = True
    else:
        print(">> Exiting.")
        sys.exit(1)

# IMU setup
imu_pipe = None
try:
    imu_pipe = rs.pipeline()
    imu_cfg = rs.config()
    imu_cfg.enable_stream(rs.stream.accel)
    imu_cfg.enable_stream(rs.stream.gyro)
    imu_pipe.start(imu_cfg)
    print("IMU initialized")
except Exception as e:
    print(f"\n[ERROR] RealSense IMU init failed: {e}")
    imu_pipe = None

# Gravity compensator baseline calibration
if not ft_dummy_mode:
    if imu_pipe is not None:
        try:
            gravity_compensator.calibrate_baseline(imu_pipe, ft_reader, warmup_sec=5.0)
        except Exception as e:
            print(f"[ERROR] Baseline calibration failed: {e}")
            user_choice = input("Calibration failed. Switch to dummy mode? (y/n): ").strip().lower()
            if user_choice == 'y':
                ft_dummy_mode = True
            else:
                sys.exit(1)
    else:
        print("[WARNING] IMU unavailable — skipping baseline calibration.")
else:
    print("[INFO] F/T dummy mode — skipping baseline calibration.")

frame_count = 0
start_time = None
total_image_count = 0
program_start_time = time.time()

last_acq_saved = {'handeye': 0, 'pose_tracking': 0, 'additional_cam': 0}

DUMMY_DISP_640x480 = np.zeros((480, 640, 3), dtype=np.uint8)
DUMMY_DISP_640x360 = np.zeros((360, 640, 3), dtype=np.uint8)

while True:
    # 1. Read camera frames (threaded, non-blocking)
    res1, res2, res3 = None, None, None

    if handeye_cam is not None:
        ok1, res1 = handeye_cam.read()
    if dji_cam is not None:
        ok2, res2 = dji_cam.read()
    if use_additional_cam and additional_cam is not None:
        ok3, res3 = additional_cam.read()

    # 2. Composite display
    disp1 = res1[1] if res1 else DUMMY_DISP_640x480
    disp2 = res2[1] if res2 else DUMMY_DISP_640x360
    
    combined = np.vstack([disp1, disp2])
    cv2.imshow("RAW Capture - q:exit, s:save, x:stop", combined)
    
    key = cv2.waitKey(1) & 0xFF

    # 3. Handle keyboard input
    if key == ord('s') and not saving_images:
        saving_images = True
        start_new_episode()
        last_acq_saved = {'handeye': 0, 'pose_tracking': 0, 'additional_cam': 0}
        
        while not image_queue.empty():
            try: image_queue.get_nowait(); image_queue.task_done()
            except: break
        
        stop_image_save_thread = False
        image_save_thread = threading.Thread(target=image_save_thread_func, daemon=True)
        image_save_thread.start()
        
        stop_ft_thread = False
        ft_thread = threading.Thread(target=ft_collection_thread, args=(ft_reader, imu_pipe, gravity_compensator), daemon=True)
        ft_thread.start()
        print(f"Raw Capture started (Episode {episode_num})")

    elif (key == ord('x') or key == ord('q')) and saving_images:
        print("Stopping data collection... (draining queue)")
        saving_images = False
        
        print(f"Waiting for image queue to drain... ({image_queue.qsize()} remaining)")
        image_queue.join()
        print("All images saved.")

        stop_ft_thread = True
        if ft_thread is not None: ft_thread.join(timeout=2.0)
        
        stop_image_save_thread = True
        if image_save_thread is not None: image_save_thread.join(timeout=2.0)
        
        print_episode_summary()
        if episode_progress_bar is not None: episode_progress_bar.close()
        if ft_csv_file is not None: ft_csv_file.close()
        episode_num = find_last_episode_number() + 1
        if key == ord('q'): break

    elif key == ord('q'):
        break

    # 4. Save unique frames independently
    if saving_images:
        # Handeye
        if res1 and res1[0] is not None and res1[2] > last_acq_saved['handeye']:
            ts_str = format_timestamp_from_float(res1[2])
            image_queue.put(('handeye', ts_str, res1[0].copy()))
            last_acq_saved['handeye'] = res1[2]
            with capture_count_lock: image_capture_count += 1
            total_image_count += 1

        # DJI (Pose Tracking)
        if res2 and res2[0] is not None and res2[2] > last_acq_saved['pose_tracking']:
            ts_str = format_timestamp_from_float(res2[2])
            image_queue.put(('pose_tracking', ts_str, res2[0].copy()))
            last_acq_saved['pose_tracking'] = res2[2]
            with capture_count_lock: image_capture_count += 1

        # Additional camera
        if use_additional_cam and res3 and res3[0] is not None and res3[2] > last_acq_saved['additional_cam']:
            ts_str = format_timestamp_from_float(res3[2])
            image_queue.put(('additional_cam', ts_str, res3[0].copy()))
            last_acq_saved['additional_cam'] = res3[2]
            with capture_count_lock: image_capture_count += 1

# Cleanup
stop_image_save_thread = True
stop_ft_thread = True

if episode_progress_bar is not None:
    episode_progress_bar.close()
    episode_progress_bar = None

if image_save_thread is not None:
    image_save_thread.join(timeout=5.0)
    
if ft_thread is not None:
    ft_thread.join(timeout=2.0)
    
if ft_csv_file is not None:
    ft_csv_file.close()
    
if handeye_cam is not None:
    handeye_cam.close()
dji_cam.close()
if 'additional_cam' in globals() and additional_cam is not None:
    additional_cam.close()
ft_reader.stop()
imu_pipe.stop()
cv2.destroyAllWindows()