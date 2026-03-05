
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import cv2
import cv2.aruco as aruco
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
import time

from utils.rs_capture import DJICapture

# ArUco marker settings
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)
T_cam_to_cube_initial = None

def load_dji_intrinsics(json_path: str = "calibration/calibration_dji_intrinsics.json"):
    """
    Load the intrinsic matrix K and distortion coefficients dist from a
    DJI action-camera chessboard calibration result (JSON).
    Assumes the same output format as dji_calibrate_intrinsics.py.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    K = np.array(data["K"], dtype=np.float32)
    dist = np.array(data["dist"], dtype=np.float32)
    return K, dist


def reduce_overexposure(image_bgr: np.ndarray) -> np.ndarray:
    """Dynamically reduce the V channel to mitigate overexposure when hardware exposure control is unavailable."""
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32)
    mean_v = float(v.mean())
    target_mean = 120.0  # target a darker image
    if mean_v > target_mean:
        factor = max(0.3, target_mean / (mean_v + 1e-6))  # aggressive reduction in range 0.3–1.0
        v = np.clip(v * factor, 0, 255)
        hsv[..., 2] = v.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image_bgr

def transform_to_cube_center(marker_id, marker_pose):
    """
    Transform a marker's coordinates into the cube-centre coordinate frame.
    All markers are assumed to be attached with their x-axis pointing to the right.
    """
    cube_size = 0.058  # 48mm
    half_size = cube_size / 2
    
    # Transformation matrices from each marker to the cube centre
    transforms = {
        1: {  # Front face
            'translation': np.array([0, 0, -half_size]),
            'rotation': np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
        },
        3: {  # Left face
            'translation': np.array([0, 0, -half_size]),
            'rotation': np.array([
                [0, 0, 1],    # 90-degree rotation about the y-axis
                [0, 1, 0],
                [-1, 0, 0]
            ])
        },
        0: {  # Right face
            'translation': np.array([0, 0, -half_size]),
            'rotation': np.array([
                [0, 0, -1],   # -90-degree rotation about the y-axis (x -> -z)
                [0, 1, 0],    # y-axis unchanged
                [1, 0, 0]     # z-axis -> x direction
            ])
        },
        2: {  # Back face
            'translation': np.array([0, 0, -half_size]),
            'rotation': np.array([
                [-1, 0, 0],   # 180-degree rotation about the y-axis
                [0, 1, 0],    # y-axis unchanged
                [0, 0, -1]    # z-axis reversed
            ])
        },
        4: {  # Top face
            'translation': np.array([0, 0, -half_size]),
            'rotation': np.array([
                [1, 0, 0],    # x-axis unchanged
                [0, 0, -1],   # y-axis -> -z direction
                [0, 1, 0]     # z-axis -> y direction
            ])
        }
    }

    
    if marker_id not in transforms:
            print(f"Marker ID {marker_id} is not supported.")
            return None
        
    transform = transforms[marker_id]

    # Current position and orientation of the marker
    marker_pos = np.array(marker_pose['position'])
    marker_rot = R.from_quat(marker_pose['orientation']).as_matrix()

    # Transform into the cube-centre coordinate frame
    cube_rot = marker_rot @ transform['rotation']
    cube_pos = marker_pos + (marker_rot @ transform['translation'])
    
    return {
        'position': cube_pos,
        'orientation': R.from_matrix(cube_rot).as_quat()
    }
def normalize_quaternion_sign(curr_quat, ref_quat):
    """Flip the sign of curr_quat to match that of ref_quat (ensures continuity)."""
    if np.dot(curr_quat, ref_quat) < 0:
        return -np.array(curr_quat)
    return np.array(curr_quat)

def low_pass_filter(new_value, prev_value, alpha=0.8):
    if prev_value is None:
        return new_value
    return alpha * np.array(prev_value) + (1 - alpha) * np.array(new_value)

def detect_markers_and_estimate_pose(image, K, D):
    # Aligned with the single-stage detection logic in get_wrist_pose_adv.py for improved stability and reduced jitter
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

    # Marker physical size
    marker_size = 0.0483

    cube_centers = []
    for i in range(len(corners)):
        marker_id = ids[i][0]
        # Skip gripper markers (IDs 6 and 7) as they are not cube markers
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

    # Fuse cube centres from multiple markers (average)
    ref_q = cube_centers[0]['orientation']
    for c in cube_centers:
        c['orientation'] = normalize_quaternion_sign(c['orientation'], ref_q)
    avg_p = np.mean([c['position'] for c in cube_centers], axis=0)
    avg_q = R.from_quat([c['orientation'] for c in cube_centers]).mean().as_quat()

    pos_local = avg_p
    quat_local = avg_q

    # Visualise
    vis = image.copy()
    vis = aruco.drawDetectedMarkers(vis, corners, ids)
    rvec, _ = cv2.Rodrigues(R.from_quat(quat_local).as_matrix())
    tvec = pos_local
    cv2.drawFrameAxes(vis, K, D, rvec, tvec, 0.06)

    return vis, {'position': pos_local.tolist(),
                 'orientation': quat_local.tolist()}

def transform_aruco_to_tcp(aruco_pose, tcp_offset=None, R_aruco_to_tcp=None):
    """
    Transform a pose expressed in the ArUco cube frame to the TCP frame.

    Parameters:
    aruco_pose (dict): Pose in the ArUco frame {'position': [...], 'orientation': [...]}
    tcp_offset (ndarray, optional): TCP offset relative to the ArUco cube [x, y, z]. Default: [0, -0.136, 0.11]
    R_aruco_to_tcp (ndarray, optional): Rotation matrix from ArUco to TCP. Default: identity (no rotation)

    Returns:
    dict: TCP pose {'position': [...], 'orientation': [...]}
    """
    # Set defaults

    if tcp_offset is None:
        tcp_offset = np.array([0.0, -0.136, 0.11])  # hand-eye calibration offset
    if R_aruco_to_tcp is None:
        R_aruco_to_tcp = np.eye(3)

    # 1. Extract position and orientation
    aruco_position = np.array(aruco_pose['position'])  # x, y, z
    aruco_orientation = np.array(aruco_pose['orientation'])  # quaternion or Euler angles

    # 2. Convert orientation to rotation matrix
    if len(aruco_orientation) == 4:  # quaternion [x, y, z, w]
        # scipy uses [x, y, z, w] convention
        R_aruco = R.from_quat(aruco_orientation).as_matrix()
    elif len(aruco_orientation) == 3:  # Euler angles [rx, ry, rz]
        R_aruco = R.from_euler('xyz', aruco_orientation).as_matrix()
    else:
        raise ValueError(f"Unsupported orientation format: {aruco_orientation}")

    # 3. Compute TCP position: ArUco position + rotated offset
    tcp_position = aruco_position + R_aruco @ tcp_offset

    # 4. Compute TCP orientation: apply additional rotation to ArUco orientation
    R_tcp = R_aruco @ R_aruco_to_tcp
    tcp_orientation = R.from_matrix(R_tcp).as_quat()

    # 5. Build and return the final TCP pose dictionary
    tcp_pose = {
        'position': tcp_position.tolist(),
        'orientation': tcp_orientation.tolist()
    }

    return tcp_pose


def main():
    # Load DJI action-camera intrinsics
    K_cam, D_cam = load_dji_intrinsics("calibration/calibration_dji_intrinsics.json")
    print("[INFO] DJI camera intrinsics loaded.")
    print("K:\n", K_cam)
    print("dist:", D_cam)

    # Open the camera via DJICapture (auto-detects DJI device, MJPG codec, 720p)
    dji_cam = DJICapture(
        name='realtime_tracking',
        dim=(1280, 720),
        fps=30,
        zero_config=False,
        threaded=True
    )
    # Initialise GUI (prevents thread conflicts)
    cv2.namedWindow('dummy')
    cv2.waitKey(1)
    cv2.destroyWindow('dummy')

    print(f"[INFO] DJI camera opened: {dji_cam.device}")

    prev_pose = None
    alpha = 0.3  # lower value (0.3) maximises responsiveness (0.0 = no filter, 1.0 = heavy filter)
    fps_start_time = time.time()
    fps_counter = 0

    while True:
        ok, result = dji_cam.read()

        if not ok or result is None or result[0] is None:
            continue

        frame = result[0].copy()  # thread-safe copy of the original frame

        # Software overexposure mitigation when hardware control is unavailable
        frame = reduce_overexposure(frame)

        # Marker detection and pose estimation
        image_with_axes, aruco_pose = detect_markers_and_estimate_pose(frame, K_cam, D_cam)

        # Transform to TCP pose and display both coordinate frames
        if aruco_pose is not None:
            # Apply low-pass filter (LPF) to reduce jitter
            if prev_pose is not None:
                # smaller alpha = more responsive
                aruco_pose['position'] = low_pass_filter(aruco_pose['position'], prev_pose['position'], alpha).tolist()
                aruco_pose['orientation'] = normalize_quaternion_sign(aruco_pose['orientation'], prev_pose['orientation'])
                aruco_pose['orientation'] = low_pass_filter(aruco_pose['orientation'], prev_pose['orientation'], alpha).tolist()
                aruco_pose['orientation'] = (np.array(aruco_pose['orientation']) / np.linalg.norm(aruco_pose['orientation'])).tolist()

            prev_pose = aruco_pose

            tcp_pose = transform_aruco_to_tcp(
                aruco_pose,
                tcp_offset=np.array([0.0, -0.13423, 0.135])
            )

            # Draw TCP coordinate frame on the image
            tcp_rvec = cv2.Rodrigues(R.from_quat(tcp_pose['orientation']).as_matrix())[0]
            tcp_tvec = np.array(tcp_pose['position'])
            cv2.drawFrameAxes(image_with_axes, K_cam, D_cam, tcp_rvec, tcp_tvec, 0.08)

        cv2.imshow('DJI Realtime ArUco Tracking', image_with_axes)
        
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            print(f"[Loop] Processing FPS: {fps_counter}")
            fps_counter = 0
            fps_start_time = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    dji_cam.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()