
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import cv2
import cv2.aruco as aruco
import numpy as np
from scipy.spatial.transform import Rotation as R
import os
import json
import time
from datetime import datetime
import argparse
from filterpy.kalman import KalmanFilter
from collections import deque
from tqdm import tqdm
from utils.rs_capture import AzureImageCapture

# ArUco marker dictionary
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)

# Globals for initial pose tracking
T_cam_to_cube_initial = None
initial_tcp_pos   = None
initial_tcp_quat  = None

# Pose tracking globals
last_valid_pose = None
pose_buffer = deque(maxlen=5)
velocity_buffer = deque(maxlen=3)
MAX_VELOCITY = 0.5       # m/s
MAX_ANGULAR_VELOCITY = 1.0  # rad/s

dist_buffer = deque(maxlen=7)
vote_buffer = deque(maxlen=7)
ref_dist = None
state = 'open'
open_thres  = 0.008  # smaller threshold to switch back to open
close_thres = 0.015  # larger threshold to switch to close

state_buffer = deque(maxlen=5)

# Gripper marker pixel history for velocity-based state detection
marker_pixel_history = {}
pixel_velocity_history = deque(maxlen=5)

# Flag: whether the initial gripper state has been determined
initial_state_determined = False

def build_T(position, quaternion):
    """Build a 4x4 homogeneous transform from translation + quaternion."""
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quaternion).as_matrix()
    T[:3,  3] = position
    return T

def transform_aruco_to_tcp(aruco_pose, tcp_offset=None, R_aruco_to_tcp=None):
    """
    Transform an ArUco cube-frame pose to TCP frame.
    
    Args:
        aruco_pose (dict): Pose in ArUco frame {'position': [...], 'orientation': [...]}
        tcp_offset (ndarray, optional): TCP offset from ArUco origin [x, y, z].
            Default: [0, -0.136, 0.135] (from hand-eye calibration).
        R_aruco_to_tcp (ndarray, optional): Rotation matrix from ArUco to TCP.
            Default: identity (no additional rotation).
    
    Returns:
        dict: TCP pose {'position': [...], 'orientation': [...]}
    """
    if tcp_offset is None:
        tcp_offset = np.array([0.0, -0.136, 0.135])
    if R_aruco_to_tcp is None:
        R_aruco_to_tcp = np.eye(3)
    
    aruco_position = np.array(aruco_pose['position'])
    aruco_orientation = np.array(aruco_pose['orientation'])
    
    if len(aruco_orientation) == 4:  # quaternion [x, y, z, w] (scipy convention)
        R_aruco = R.from_quat(aruco_orientation).as_matrix()
    elif len(aruco_orientation) == 3:  # Euler angles [rx, ry, rz]
        R_aruco = R.from_euler('xyz', aruco_orientation).as_matrix()
    else:
        raise ValueError(f"Unsupported orientation format: {aruco_orientation}")
    
    tcp_position = aruco_position + R_aruco @ tcp_offset
    R_tcp = R_aruco @ R_aruco_to_tcp
    tcp_orientation = R.from_matrix(R_tcp).as_quat()
    
    return {
        'position': tcp_position.tolist(),
        'orientation': tcp_orientation.tolist()
    }

def normalize_quaternion_sign(curr_quat, ref_quat):
    """Flip the sign of curr_quat if it disagrees with ref_quat (to ensure consistent hemisphere)."""
    if np.dot(curr_quat, ref_quat) < 0:
        return -np.array(curr_quat)
    return np.array(curr_quat)

def load_config():
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: config.json not found")
        return None

def load_dji_intrinsics(json_path: str = "calibration/calibration_dji_intrinsics.json"):
    """
    Load DJI action camera intrinsics (K, dist) from a JSON file
    produced by the checkerboard calibration script.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    K = np.array(data["K"], dtype=np.float32)
    dist = np.array(data["dist"], dtype=np.float32)
    return K, dist


def transform_to_cube_center(marker_id, marker_pose):
    """
    Transform a single face marker's pose to the cube center coordinate frame.
    Each marker's x-axis points to the right when attached to the cube face.
    """
    cube_size = 0.058  # 58mm cube
    half_size = cube_size / 2
    
    # Per-face: translation from marker to cube center, and rotation into cube frame
    transforms = {
        1: {  # Front face
            'translation': np.array([0, 0, -half_size]),
            'rotation': np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
        },
        3: {  # Left face (+90° rotation around Y)
            'translation': np.array([0, 0, -half_size]),
            'rotation': np.array([
                [0, 0, 1],
                [0, 1, 0],
                [-1, 0, 0]
            ])
        },
        0: {  # Right face (-90° rotation around Y)
            'translation': np.array([0, 0, -half_size]),
            'rotation': np.array([
                [0, 0, -1],
                [0, 1, 0],
                [1, 0, 0]
            ])
        },
        2: {  # Back face (180° rotation around Y)
            'translation': np.array([0, 0, -half_size]),
            'rotation': np.array([
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, -1]
            ])
        },
        4: {  # Top face (+90° rotation around X)
            'translation': np.array([0, 0, -half_size]),
            'rotation': np.array([
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0]
            ])
        }
    }
    
    # Gripper markers (IDs 6, 7) are not cube markers
    if marker_id in [6, 7]:
        return None
    
    if marker_id not in transforms:
        print(f"Warning: Unknown marker ID {marker_id}, skipping...")
        return None
    
    transform = transforms[marker_id]

    marker_pos = np.array(marker_pose['position'])
    marker_rot = R.from_quat(marker_pose['orientation']).as_matrix()
    
    cube_rot = marker_rot @ transform['rotation']
    cube_pos = marker_pos + (marker_rot @ transform['translation'])
    
    return {
        'position': cube_pos,
        'orientation': R.from_matrix(cube_rot).as_quat()
    }

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray

def init_kalman_filter():
    """
    State: [x, y, z, vx, vy, vz, qw, qx, qy, qz]
    Measurement: [x, y, z, qw, qx, qy, qz]
    """
    kf = KalmanFilter(dim_x=10, dim_z=7)
    dt = 1.0/30.0  # assuming 30 fps
    
    kf.F = np.eye(10)
    kf.F[:3, 3:6] = np.eye(3) * dt  # position update from velocity
    
    kf.H = np.zeros((7, 10))
    kf.H[:3, :3] = np.eye(3)  # position measurement
    kf.H[3:, 6:] = np.eye(4)  # orientation measurement
    
    kf.R = np.eye(7) * 0.01    # measurement noise
    kf.Q = np.eye(10) * 0.001  # process noise
    kf.Q[3:6, 3:6] *= 0.1     # velocity noise
    kf.Q[6:, 6:] *= 0.01      # quaternion noise
    
    return kf

def low_pass_filter(new_value, prev_value, alpha=0.85):
    if prev_value is None:
        return new_value
    return alpha * prev_value + (1 - alpha) * new_value

def check_pose_validity(pose, prev_pose, dt=1/30.0):
    if prev_pose is None:
        return True
        
    pos_change = np.linalg.norm(np.array(pose['position']) - np.array(prev_pose['position'])) / dt
    if pos_change > MAX_VELOCITY:
        return False
        
    q1 = np.array(pose['orientation'])
    q2 = np.array(prev_pose['orientation'])
    angle_change = 2 * np.arccos(np.abs(np.dot(q1, q2))) / dt
    if angle_change > MAX_ANGULAR_VELOCITY:
        return False
        
    return True

def detect_markers_and_estimate_pose(image, K, D, prev_orientation=None):
    global T_cam_to_cube_initial

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    parameters = aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMin = 5
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 4
    parameters.adaptiveThreshConstant = 7
    parameters.minMarkerPerimeterRate = 0.02
    parameters.maxMarkerPerimeterRate = 0.4
    parameters.polygonalApproxAccuracyRate = 0.005
    parameters.minCornerDistanceRate = 0.05
    parameters.minDistanceToBorder = 5
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    parameters.cornerRefinementWinSize = 2
    parameters.cornerRefinementMaxIterations = 10
    parameters.cornerRefinementMinAccuracy = 0.001

    corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=parameters)
    if ids is None:
        return image, None

    # Process cube markers only (skip gripper markers 6 and 7)
    cube_centers = []
    marker_size = 0.048
    
    for i in range(len(corners)):
        marker_id = ids[i][0]
        if marker_id in [6, 7]:
            continue
            
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners[i:i+1], marker_size, K, D)
        Rcm, _ = cv2.Rodrigues(rvecs[0][0])
        mrk = {
            'position': tvecs[0][0],
            'orientation': R.from_matrix(Rcm).as_quat()
        }
        cube = transform_to_cube_center(marker_id, mrk)
        if cube: 
            cube_centers.append(cube)

    if not cube_centers:
        return image, None

    # Fuse estimates from multiple visible cube faces
    ref_q = cube_centers[0]['orientation']
    for c in cube_centers:
        c['orientation'] = normalize_quaternion_sign(c['orientation'], ref_q)
    avg_p = np.mean([c['position'] for c in cube_centers], axis=0)
    avg_q = R.from_quat([c['orientation'] for c in cube_centers]).mean().as_quat()

    pos_local = avg_p
    quat_local = avg_q
    
    vis = image.copy()
    vis = aruco.drawDetectedMarkers(vis, corners, ids)
    rvec, _ = cv2.Rodrigues(R.from_quat(quat_local).as_matrix())
    tvec = pos_local
    cv2.drawFrameAxes(vis, K, D, rvec, tvec, 0.06)

    return vis, {'position': pos_local.tolist(),
                 'orientation': quat_local.tolist()}

def detect_gripper_markers(image, K, D, marker_size=0.021):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    parameters = aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMin = 5
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 4
    parameters.adaptiveThreshConstant = 7
    parameters.minMarkerPerimeterRate = 0.02
    parameters.maxMarkerPerimeterRate = 0.4
    parameters.polygonalApproxAccuracyRate = 0.005
    parameters.minCornerDistanceRate = 0.05
    parameters.minDistanceToBorder = 5
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    parameters.cornerRefinementWinSize = 2
    parameters.cornerRefinementMaxIterations = 10
    parameters.cornerRefinementMinAccuracy = 0.001
    corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=parameters)
    marker_infos = []
    if ids is not None and len(corners) > 0:
        for i in range(len(corners)):
            marker_id = ids[i][0]
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners[i:i+1], marker_size, K, D)
            rvec = rvecs[0][0]
            tvec = tvecs[0][0]
            marker_infos.append({'id': marker_id, 'tvec': tvec, 'rvec': rvec, 'corners': corners[i][0]})
    return marker_infos

def get_handeye_camera_intrinsics():
    fx, fy = 640.0, 640.0
    cx, cy = 640.0, 400.0
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    D = np.zeros(5)
    return K, D

def detect_handeye_markers_and_estimate_pose(image, K, D, marker_size=0.021):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_with_axes = image.copy()
    parameters = aruco.DetectorParameters()
    parameters.adaptiveThreshWinSizeMin = 5
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 4
    parameters.adaptiveThreshConstant = 7
    parameters.minMarkerPerimeterRate = 0.02
    parameters.maxMarkerPerimeterRate = 0.4
    parameters.polygonalApproxAccuracyRate = 0.005
    parameters.minCornerDistanceRate = 0.05
    parameters.minDistanceToBorder = 5
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
    parameters.cornerRefinementWinSize = 2
    parameters.cornerRefinementMaxIterations = 10
    parameters.cornerRefinementMinAccuracy = 0.001
    corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=parameters)
    marker_infos = []
    if ids is not None and len(corners) > 0:
        image_with_axes = aruco.drawDetectedMarkers(image_with_axes, corners, ids)
        for i in range(len(corners)):
            marker_id = ids[i][0]
            if marker_id not in [6, 7]:
                continue
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners[i:i+1], marker_size, K, D)
            rvec = rvecs[0][0]
            tvec = tvecs[0][0]
            
            corner_points = corners[i][0]
            marker_center = np.mean(corner_points, axis=0)
            
            marker_infos.append({
                'id': marker_id, 
                'tvec': tvec, 
                'rvec': rvec, 
                'corners': corner_points,
                'center_x': marker_center[0],
                'center_y': marker_center[1]
            })
            cv2.drawFrameAxes(image_with_axes, K, D, rvec, tvec, 0.060)
            cv2.putText(image_with_axes, 
                       f"ID {marker_id}: X={tvec[0]:.3f}, Y={tvec[1]:.3f}, Z={tvec[2]:.3f}",
                       (10, 30 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(image_with_axes, 
                   f"Detected Markers: {len(marker_infos)}",
                   (10, image_with_axes.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return marker_infos, image_with_axes

def update_gripper_state_from_pixel_velocity(marker_infos, dt=1/30.0, velocity_threshold=3.0):
    """
    Detect gripper state (open/close) from pixel-space velocity of gripper markers.
    
    Args:
        marker_infos: List of detected marker info dicts.
        dt: Frame interval in seconds.
        velocity_threshold: Pixel velocity to trigger a state change (px/s).
    
    Returns:
        state: 'open', 'close', or None (no change detected).
        debug_info: Dict with diagnostics.
    """
    global marker_pixel_history, pixel_velocity_history, initial_state_determined
    
    velocities = []
    debug_info = {'velocities': [], 'avg_velocity': 0, 'smoothed_velocity': 0, 'markers_detected': 0}
    
    gripper_markers = [m for m in marker_infos if m['id'] in [6, 7]]
    debug_info['markers_detected'] = len(gripper_markers)
    
    # On first call, determine initial state from marker distance
    if not initial_state_determined:
        initial_state_determined = True
        
        if len(gripper_markers) == 2:
            marker_6 = next((m for m in marker_infos if m['id'] == 6), None)
            marker_7 = next((m for m in marker_infos if m['id'] == 7), None)
            
            if marker_6 and marker_7:
                distance = abs(marker_7['center_x'] - marker_6['center_x'])
                if distance >= 500:
                    return 'open', debug_info
                else:
                    return 'close', debug_info
        
        return 'open', debug_info
    
    for m in marker_infos:
        if m['id'] not in [6, 7]:
            continue
            
        marker_id = m['id']
        x_now = m['center_x']
        
        if marker_id not in marker_pixel_history:
            marker_pixel_history[marker_id] = deque(maxlen=2)
        marker_pixel_history[marker_id].append(x_now)
        
        if len(marker_pixel_history[marker_id]) == 2:
            x_prev, x_curr = marker_pixel_history[marker_id][0], marker_pixel_history[marker_id][1]
            v = (x_curr - x_prev) / dt
            
            # Marker 6 is on the left: opening moves it left (x decreases), so flip sign.
            # Marker 7 is on the right: opening moves it right (x increases), keep sign.
            if marker_id == 6:
                adjusted_v = -v
            elif marker_id == 7:
                adjusted_v = +v
            else:
                adjusted_v = 0
                
            velocities.append(adjusted_v)
            debug_info['velocities'].append({
                'marker_id': marker_id,
                'raw_velocity': v,
                'adjusted_velocity': adjusted_v,
                'x_prev': x_prev,
                'x_curr': x_curr
            })
    
    if velocities:
        avg_v = np.mean(velocities)
        pixel_velocity_history.append(avg_v)
        avg_v_smoothed = np.mean(pixel_velocity_history)
        
        debug_info['avg_velocity'] = avg_v
        debug_info['smoothed_velocity'] = avg_v_smoothed
        
        if avg_v_smoothed > velocity_threshold:
            return 'open', debug_info
        elif avg_v_smoothed < -velocity_threshold:
            return 'close', debug_info
    
    return None, debug_info

def reset_global_variables():
    """Reset all per-episode global tracking variables."""
    global marker_pixel_history, pixel_velocity_history, initial_state_determined
    global state_stability_buffer, pose_buffer, velocity_buffer
    
    marker_pixel_history = {}
    pixel_velocity_history = deque(maxlen=5)
    initial_state_determined = False
    state_stability_buffer = deque(maxlen=5)
    pose_buffer = deque(maxlen=5)
    velocity_buffer = deque(maxlen=3)

def process_images_from_directory(image_dir, output_json_path, visualize=False):
    global initial_tcp_pos, initial_tcp_quat, last_valid_pose, T_cam_to_cube_initial
    global initial_state_determined
    
    reset_global_variables()
    T_cam_to_cube_initial = None
    
    # Load DJI camera intrinsics from checkerboard calibration
    K_azure, D_azure = load_dji_intrinsics("calibration/calibration_dji_intrinsics.json")
    
    # Handeye (RealSense) camera intrinsics
    K_handeye, D_handeye = get_handeye_camera_intrinsics()
    
    results = []
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))])
    if not image_files:
        print(f"No PNG or JPG files found in {image_dir}")
        return
    
    detected_count = 0
    failed_count = 0
    
    state = 'open'  # Default initial gripper state
    velocity_threshold = 10.0  # px/s
    state_stability_buffer = deque(maxlen=3)
    
    handeye_dir = os.path.join(os.path.dirname(image_dir), 'handeye')
    
    for idx, image_file in tqdm(enumerate(image_files), total=len(image_files), desc="Processing images"):
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            failed_count += 1
            continue

        # --- Gripper state detection from handeye camera ---
        handeye_path = os.path.join(handeye_dir, image_file)
        gripper_state = state
        gripper_vis = None
        if os.path.exists(handeye_path):
            handeye_img = cv2.imread(handeye_path)
            if handeye_img is not None:
                if visualize:
                    marker_infos, gripper_vis = detect_handeye_markers_and_estimate_pose(handeye_img, K_handeye, D_handeye)
                else:
                    # Lightweight detection without visualization
                    gray = cv2.cvtColor(handeye_img, cv2.COLOR_BGR2GRAY)
                    parameters = aruco.DetectorParameters()
                    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_APRILTAG
                    parameters.cornerRefinementWinSize = 2
                    corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=parameters)
                    marker_infos = []
                    if ids is not None:
                        for i, corner in enumerate(corners):
                            marker_id = ids[i][0]
                            if marker_id in [6, 7]:
                                corner_points = corner[0]
                                marker_center = np.mean(corner_points, axis=0)
                                marker_infos.append({
                                    'id': marker_id,
                                    'center_x': marker_center[0],
                                    'center_y': marker_center[1]
                                })
                
                detected_state, debug_info = update_gripper_state_from_pixel_velocity(
                    marker_infos, dt=1/30.0, velocity_threshold=velocity_threshold)
                
                # Majority-vote state transition
                if detected_state is not None:
                    state_stability_buffer.append(detected_state)
                    
                    if len(state_stability_buffer) >= 2:
                        votes = list(state_stability_buffer)
                        open_votes = votes.count('open')
                        close_votes = votes.count('close')
                        
                        if open_votes > close_votes:
                            state = 'open'
                        elif close_votes > open_votes:
                            state = 'close'
                
                gripper_state = state
                
                if visualize and gripper_vis is not None:
                    state_text = 'close' if gripper_state == 'close' else 'open'
                    color = (0,0,255) if state_text=='close' else (0,255,0)
                    cv2.putText(gripper_vis, f'State: {state_text}', (20, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
                    
                    if debug_info['velocities']:
                        smoothed_v = debug_info['smoothed_velocity']
                        cv2.putText(gripper_vis, f'Velocity: {smoothed_v:.1f} px/s', (20, 120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                        cv2.putText(gripper_vis, f'Threshold: +/-{velocity_threshold:.1f}', (20, 160), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                        
                        for i, vel_info in enumerate(debug_info['velocities']):
                            marker_id = vel_info['marker_id']
                            adj_vel = vel_info['adjusted_velocity']
                            cv2.putText(gripper_vis, f'M{marker_id}: {adj_vel:.1f}', (20, 200 + i*30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                
        # --- Cube pose estimation from DJI camera ---
        image_with_axes, aruco_pose = detect_markers_and_estimate_pose(image, K_azure, D_azure)
        if aruco_pose is not None:
            tcp_pose = transform_aruco_to_tcp(
                aruco_pose,
                tcp_offset=np.array([0.0, -0.13423, 0.135])
            )
            tcp_pose['gripper_state'] = 1 if gripper_state == 'open' else 0
            
            result = {
                "image_file": image_file,
                "pose": tcp_pose,
                "timestamp": time.time()
            }
            results.append(result)
            detected_count += 1
            
            if visualize:
                if gripper_vis is not None:
                    vis_pose_small = cv2.resize(image_with_axes, (640, 480))
                    vis_gripper_small = cv2.resize(gripper_vis, (640, 480))
                    vis_concat = np.concatenate([vis_pose_small, vis_gripper_small], axis=1)
                    cv2.imshow('PoseTracking (Cube) | Handeye (Gripper)', vis_concat)
                else:
                    vis_pose_small = cv2.resize(image_with_axes, (640, 480))
                    cv2.imshow('PoseTracking (Cube)', vis_pose_small)
                cv2.waitKey(1)
        else:
            failed_count += 1

    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Processed {len(image_files)} images: {detected_count} success, {failed_count} failed ({detected_count/len(image_files)*100:.1f}% success rate)")
    
    if visualize:
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Process images and detect ArUco markers')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the data directory containing episode folders')
    parser.add_argument('--visualize', action='store_true',
                        help='Enable real-time visualization (slower but shows progress)')
    args = parser.parse_args()
    
    base_path = args.data_path
    episode_dirs = sorted([d for d in os.listdir(base_path) if d.startswith('episode_')])
    
    print(f"🚀 Pose detection starting...")
    print(f"Visualization: {'Enabled' if args.visualize else 'Disabled (faster)'}")
    print(f"Episodes to process: {len(episode_dirs)}")
    
    start_time = time.time()
    
    skipped_count = 0
    for episode_dir in episode_dirs:
        episode_path = os.path.join(base_path, episode_dir)
        image_directory = os.path.join(episode_path, 'images', 'pose_tracking')
        output_json = os.path.join(episode_path, 'raw_pose.json')
        if not os.path.exists(image_directory):
            print(f"Skipping {episode_dir}: Image directory not found")
            continue
        if os.path.exists(output_json):
            skipped_count += 1
            continue

        print(f"\nProcessing {episode_dir}...")
        process_images_from_directory(image_directory, output_json, visualize=args.visualize)
    
    elapsed_time = time.time() - start_time
    if skipped_count > 0:
        print(f"\nSkipped {skipped_count} episodes (raw_pose.json already exists)")
    print(f"\n🎯 Processing completed!")
    print(f"Total time: {elapsed_time:.1f} seconds")

if __name__ == "__main__":
    main()