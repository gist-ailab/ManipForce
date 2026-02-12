
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
# import pyk4a  # 제거
# from pyk4a import Config, PyK4A  # 제거
from filterpy.kalman import KalmanFilter
from collections import deque
from tqdm import tqdm
from utils.rs_capture import AzureImageCapture

# ArUco 마커 설정
# dictionary = aruco.Dictionary_get(aruco.DICT_6X6_100)
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)

# 파일 최상단에 선언
T_cam_to_cube_initial = None
initial_tcp_pos   = None
initial_tcp_quat  = None

# 전역 변수 추가
last_valid_pose = None
pose_buffer = deque(maxlen=5)  # 최근 5개 포즈 저장
velocity_buffer = deque(maxlen=3)  # 속도 추적용
MAX_VELOCITY = 0.5  # 최대 허용 속도 (m/s)
MAX_ANGULAR_VELOCITY = 1.0  # 최대 허용 각속도 (rad/s)

dist_buffer = deque(maxlen=7)
vote_buffer = deque(maxlen=7)
ref_dist = None
state = 'open'
open_thres = 0.008  # open으로 돌아갈 때는 좀 더 작은 변화만 있어도 됨
close_thres = 0.015 # close로 바뀔 때는 더 큰 변화 필요

state_buffer = deque(maxlen=5)  # 최근 5번의 open/close 판정

# 전역 변수로 마커 픽셀 히스토리 저장
marker_pixel_history = {}  # marker_id: deque of past x-coords
pixel_velocity_history = deque(maxlen=5)

# 초기 상태 판별을 위한 변수들 추가
initial_state_determined = False  # 초기 상태가 결정되었는지 여부

def build_T(position, quaternion):
    """병진 + 쿼터니언 -> 4x4 homogeneous transform"""
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quaternion).as_matrix()
    T[:3,  3] = position
    return T

def transform_aruco_to_tcp(aruco_pose, tcp_offset=None, R_aruco_to_tcp=None):
    """
    ArUco 큐브 좌표계 기준 포즈를 TCP 좌표계 기준 포즈로 변환
    
    매개변수:
    aruco_pose (dict): ArUco 좌표계 기준 포즈 딕셔너리 {'position': [...], 'orientation': [...]}
    tcp_offset (ndarray, optional): TCP의 ArUco 큐브 기준 오프셋 [x, y, z]. 기본값: [0, -0.136, 0.13]
    R_aruco_to_tcp (ndarray, optional): ArUco에서 TCP로의 회전 행렬. 기본값: 단위 행렬(회전 없음)
    
    반환값:
    dict: TCP 좌표계 기준 포즈 {'position': [...], 'orientation': [...]}
    """
    # 기본값 설정

    if tcp_offset is None:
        tcp_offset = np.array([0.0, -0.136, 0.135])  # hand–eye 캘리브레이션 오프셋 #긴게 0.13, 짧은게 0.11
    if R_aruco_to_tcp is None:
        R_aruco_to_tcp = np.eye(3)
    
    # 1. 위치와 방향 추출
    aruco_position = np.array(aruco_pose['position'])  # x, y, z
    aruco_orientation = np.array(aruco_pose['orientation'])  # 쿼터니언 또는 오일러 각도
    
    # 2. 방향 회전 행렬로 변환
    if len(aruco_orientation) == 4:  # 쿼터니언 [x, y, z, w] 또는 [w, x, y, z]
        # 쿼터니언 형식에 따른 처리 (scipy는 [x, y, z, w] 형식 사용)
        R_aruco = R.from_quat(aruco_orientation).as_matrix()
    elif len(aruco_orientation) == 3:  # 오일러 각도 [rx, ry, rz]
        R_aruco = R.from_euler('xyz', aruco_orientation).as_matrix()
    else:
        raise ValueError(f"지원되지 않는 방향 형식: {aruco_orientation}")
    
    # 3. TCP 위치 계산: ArUco 위치 + (ArUco 방향에 따라 회전된 오프셋)
    tcp_position = aruco_position + R_aruco @ tcp_offset
    
    # 4. TCP 방향 계산: ArUco 방향에 추가 회전 적용
    R_tcp = R_aruco @ R_aruco_to_tcp
    tcp_orientation = R.from_matrix(R_tcp).as_quat()
    
    # 5. 최종 TCP 포즈 생성 (딕셔너리 형태로 반환)
    tcp_pose = {
        'position': tcp_position.tolist(),
        'orientation': tcp_orientation.tolist()
    }
    
    return tcp_pose

# 쿼터니언 부호 정규화 함수
def normalize_quaternion_sign(curr_quat, ref_quat):
    """현재 쿼터니언을 기준 쿼터니언과 부호를 일치시킴"""
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

def load_camera_parameters():
    try:
        with open('camera_intrinsics.json', 'r') as f:
            intrinsic_data = json.load(f)
            K1 = np.array(intrinsic_data['camera1']['camera_matrix'])
            D1 = np.array(intrinsic_data['camera1']['dist_coeffs'])
            K2 = np.array(intrinsic_data['camera2']['camera_matrix'])
            D2 = np.array(intrinsic_data['camera2']['dist_coeffs'])
    except FileNotFoundError:
        K1, D1, K2, D2 = None, None, None, None
    
        with open('calibration/camera_calibration_brio_1920.json', 'r') as f:
            brio_intrinsics = json.load(f)          
            K3 = np.array(brio_intrinsics["camera_matrix"])
            D3 = np.array(brio_intrinsics["distortion_coeffs"])

        return K1, D1, K2, D2, K3, D3


def load_dji_intrinsics(json_path: str = "calibration/calibration_dji_intrinsics.json"):
    """
    DJI 액션캠 체스보드 캘리브레이션 결과(JSON)에서 내적 행렬 K와 왜곡 계수 dist를 로드.
    dji_calibrate_intrinsics.py 출력 포맷과 동일하다고 가정.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    K = np.array(data["K"], dtype=np.float32)
    dist = np.array(data["dist"], dtype=np.float32)
    return K, dist



def transform_to_cube_center(marker_id, marker_pose):
    """
    마커의 좌표를 큐브 중심 좌표계로 변환합니다.
    모든 마커의 x축이 원래 오른쪽을 향하도록 부착된 상태입니다.
    """
    cube_size = 0.058  # 48mm
    half_size = cube_size / 2
    
    # 각 마커에서 큐브 중심까지의 변환 행렬
    transforms = {
        1: {  # Front face (정면)
            'translation': np.array([0, 0, -half_size]),
            'rotation': np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
        },
        3: {  # Left face (왼쪽)
            'translation': np.array([0, 0, -half_size]),
            'rotation': np.array([
                [0, 0, 1],    # y축으로 90도 회전
                [0, 1, 0],
                [-1, 0, 0]
            ])
        },
        0: {  # Right face (오른쪽)
            'translation': np.array([0, 0, -half_size]),
            'rotation': np.array([
                [0, 0, -1],   # y축으로 -90도 회전 (x축이 -z방향으로)
                [0, 1, 0],    # y축은 그대로
                [1, 0, 0]     # z축이 x방향으로
            ])
        },
        2: {  # Back face (뒷면)
            'translation': np.array([0, 0, -half_size]),  # half_size로 수정
            'rotation': np.array([
                [-1, 0, 0],   # y축으로 180도 회전
                [0, 1, 0],    # y축은 그대로
                [0, 0, -1]    # z축이 반대로
            ])
        },
        4: {  # Top face (윗면)
            'translation': np.array([0, 0, -half_size]),  # 다른 마커들과 동일하게
            'rotation': np.array([
                [1, 0, 0],    # x축은 그대로
                [0, 0, -1],   # y축이 -z방향으로
                [0, 1, 0]     # z축이 y방향으로
            ])
        }
    }
    
    # 그리퍼 마커 (6, 7)는 큐브 마커가 아니므로 None 반환
    if marker_id in [6, 7]:
        return None
    
    # 정의되지 않은 마커 ID 처리
    if marker_id not in transforms:
        print(f"Warning: Unknown marker ID {marker_id}, skipping...")
        return None
    
    transform = transforms[marker_id]

    # 마커의 현재 위치와 방향
    marker_pos = np.array(marker_pose['position'])
    marker_rot = R.from_quat(marker_pose['orientation']).as_matrix()
    
    # 큐브 중심 기준 좌표계로 변환
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
    # 상태: [x,y,z, vx,vy,vz, qw,qx,qy,qz], 측정: [x,y,z, qw,qx,qy,qz]
    kf = KalmanFilter(dim_x=10, dim_z=7)  # 각속도 제거하고 위치, 속도, 쿼터니언만 추적
    dt = 1.0/30.0  # 30fps 가정
    
    # 상태 전이 행렬
    kf.F = np.eye(10)
    kf.F[:3, 3:6] = np.eye(3) * dt  # 위치 업데이트
    
    # 측정 행렬
    kf.H = np.zeros((7, 10))
    kf.H[:3, :3] = np.eye(3)  # 위치 측정
    kf.H[3:, 6:] = np.eye(4)  # 방향 측정
    
    # 노이즈 매트릭스
    kf.R = np.eye(7) * 0.01  # 측정 노이즈
    kf.Q = np.eye(10) * 0.001  # 프로세스 노이즈
    kf.Q[3:6, 3:6] *= 0.1  # 속도 노이즈
    kf.Q[6:, 6:] *= 0.01  # 쿼터니언 노이즈
    
    return kf

def low_pass_filter(new_value, prev_value, alpha=0.85):
    if prev_value is None:
        return new_value
    return alpha * prev_value + (1 - alpha) * new_value

def check_pose_validity(pose, prev_pose, dt=1/30.0):
    if prev_pose is None:
        return True
        
    # 위치 변화 검사
    pos_change = np.linalg.norm(np.array(pose['position']) - np.array(prev_pose['position'])) / dt
    if pos_change > MAX_VELOCITY:
        return False
        
    # 방향 변화 검사
    q1 = np.array(pose['orientation'])
    q2 = np.array(prev_pose['orientation'])
    angle_change = 2 * np.arccos(np.abs(np.dot(q1, q2))) / dt
    if angle_change > MAX_ANGULAR_VELOCITY:
        return False
        
    return True

def detect_markers_and_estimate_pose(image, K, D, prev_orientation=None):
    global T_cam_to_cube_initial

    # 단순화된 1단계 검출 (get_gripper_state.py 방식)
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

    # 큐브 마커만 처리 (그리퍼 마커 제외)
    cube_centers = []
    marker_size = 0.0483
    
    for i in range(len(corners)):
        marker_id = ids[i][0]
        # 그리퍼 마커(6, 7)는 큐브 마커가 아니므로 건너뛰기
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

    # 큐브 중심 융합
    ref_q = cube_centers[0]['orientation']
    for c in cube_centers:
        c['orientation'] = normalize_quaternion_sign(c['orientation'], ref_q)
    avg_p = np.mean([c['position'] for c in cube_centers], axis=0)
    avg_q = R.from_quat([c['orientation'] for c in cube_centers]).mean().as_quat()

    pos_local = avg_p
    quat_local = avg_q
    
    # 단순화된 시각화
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
            
            # Calculate marker center in pixel coordinates
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
    픽셀 좌표 속도 변화를 이용한 gripper state detection
    
    Args:
        marker_infos: 검출된 마커 정보 리스트
        dt: 프레임 간격 (초)
        velocity_threshold: 상태 변경을 위한 속도 임계값 (픽셀/초)
    
    Returns:
        state: 'open', 'close', 또는 None (변화 없음)
        debug_info: 디버그 정보 딕셔너리
    """
    global marker_pixel_history, pixel_velocity_history, initial_state_determined
    
    velocities = []
    debug_info = {'velocities': [], 'avg_velocity': 0, 'smoothed_velocity': 0, 'markers_detected': 0}
    
    # 그리퍼 마커(6, 7) 검출 개수 확인
    gripper_markers = [m for m in marker_infos if m['id'] in [6, 7]]
    debug_info['markers_detected'] = len(gripper_markers)
    
    # 초기 상태가 결정되지 않았으면 거리 기준으로 간단 판단
    if not initial_state_determined:
        initial_state_determined = True
        
        # 두 마커가 모두 있을 때만 거리 계산
        if len(gripper_markers) == 2:
            marker_6 = next((m for m in marker_infos if m['id'] == 6), None)
            marker_7 = next((m for m in marker_infos if m['id'] == 7), None)
            
            if marker_6 and marker_7:
                distance = abs(marker_7['center_x'] - marker_6['center_x'])
                print(f"🎯 초기 마커 거리: {distance:.1f} 픽셀")
                
                # 500픽셀 이상이면 open, 미만이면 close
                if distance >= 500:
                    print("   → Open (거리 >= 500픽셀)")
                    return 'open', debug_info
                else:
                    print("   → Close (거리 < 500픽셀)")
                    return 'close', debug_info
        
        # 마커가 2개가 아니면 기본값 open
        return 'open', debug_info
    
    for m in marker_infos:
        if m['id'] not in [6, 7]:  # 그리퍼 마커만 사용
            continue
            
        marker_id = m['id']
        x_now = m['center_x']
        
        # 이 마커의 과거 x 기록 가져오기
        if marker_id not in marker_pixel_history:
            marker_pixel_history[marker_id] = deque(maxlen=2)
        marker_pixel_history[marker_id].append(x_now)
        
        # 속도 계산 (최소 2개 포인트 필요)
        if len(marker_pixel_history[marker_id]) == 2:
            x_prev, x_curr = marker_pixel_history[marker_id][0], marker_pixel_history[marker_id][1]
            v = (x_curr - x_prev) / dt
            
            # 방향 조정: 
            # - 마커 6 (왼쪽): 그리퍼가 열릴 때 x가 감소하므로 속도에 -1 곱함
            # - 마커 7 (오른쪽): 그리퍼가 열릴 때 x가 증가하므로 속도 그대로 사용
            if marker_id == 6:  # 왼쪽 마커
                adjusted_v = -v  # 음의 속도가 open 방향
            elif marker_id == 7:  # 오른쪽 마커  
                adjusted_v = +v  # 양의 속도가 open 방향
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
    
    # 평균 속도 계산 (마커가 하나만 있어도 계산)
    if velocities:
        avg_v = np.mean(velocities)
        pixel_velocity_history.append(avg_v)
        
        # 스무딩된 평균 속도
        avg_v_smoothed = np.mean(pixel_velocity_history)
        
        debug_info['avg_velocity'] = avg_v
        debug_info['smoothed_velocity'] = avg_v_smoothed
        
        # 상태 판별
        if avg_v_smoothed > velocity_threshold:  # 픽셀/초
            return 'open', debug_info
        elif avg_v_smoothed < -velocity_threshold:
            return 'close', debug_info
    
    
    # 마커가 없거나 속도가 임계값을 넘지 않으면 None 반환 (이전 상태 유지)
    return None, debug_info

def reset_global_variables():
    """전역 변수 초기화"""
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
    
    # 전역 변수 초기화
    reset_global_variables()
    T_cam_to_cube_initial = None
    
    # DJI 액션캠 내적 파라미터 사용 (체스보드 캘리브레이션 결과)
    K_azure, D_azure = load_dji_intrinsics("calibration/calibration_dji_intrinsics.json")
    
    # For handeye (Realsense) camera
    K_handeye, D_handeye = get_handeye_camera_intrinsics()
    
    results = []
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))])
    if not image_files:
        print(f"No PNG or JPG files found in {image_dir}")
        return
    
    detected_count = 0
    failed_count = 0
    
    # --- 단순화된 그리퍼 상태 감지 변수 ---
    state = 'open'  # Always start as 'open' (그리퍼의 자연스러운 초기 상태)
    velocity_threshold = 10.0  # 픽셀/초 임계값 (get_gripper_state.py와 동일)
    state_stability_buffer = deque(maxlen=3)  # 버퍼 크기 감소로 반응성 향상
    
    # handeye image dir
    handeye_dir = os.path.join(os.path.dirname(image_dir), 'handeye')
    
    for idx, image_file in tqdm(enumerate(image_files), total=len(image_files), desc="Processing images"):
        image_path = os.path.join(image_dir, image_file)
        image = cv2.imread(image_path)
        if image is None:
            failed_count += 1
            continue
        # --- Pixel velocity-based gripper state detection from handeye image ---
        handeye_path = os.path.join(handeye_dir, image_file)
        gripper_state = state
        gripper_vis = None
        if os.path.exists(handeye_path):
            handeye_img = cv2.imread(handeye_path)
            if handeye_img is not None:
                if visualize:
                    marker_infos, gripper_vis = detect_handeye_markers_and_estimate_pose(handeye_img, K_handeye, D_handeye)
                else:
                    # 시각화 없이 마커만 검출
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
                
                # Pixel velocity-based state detection
                detected_state, debug_info = update_gripper_state_from_pixel_velocity(
                    marker_infos, dt=1/30.0, velocity_threshold=velocity_threshold)
                
                # 단순화된 상태 전환 (get_gripper_state.py 방식)
                if detected_state is not None:
                    state_stability_buffer.append(detected_state)
                    
                    # 단순한 다수결 투표 (버퍼 크기 3으로 감소)
                    if len(state_stability_buffer) >= 2:
                        votes = list(state_stability_buffer)
                        open_votes = votes.count('open')
                        close_votes = votes.count('close')
                        
                        if open_votes > close_votes:
                            state = 'open'
                        elif close_votes > open_votes:
                            state = 'close'
                        # 동점이면 현재 상태 유지
                
                gripper_state = state
                
                # 시각화가 활성화된 경우에만 화면에 정보 표시
                if visualize and gripper_vis is not None:
                    # Enhanced visualization with velocity info
                    state_text = 'close' if gripper_state == 'close' else 'open'
                    color = (0,0,255) if state_text=='close' else (0,255,0)
                    cv2.putText(gripper_vis, f'State: {state_text}', (20, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
                    
                    # Show velocity debug info
                    if debug_info['velocities']:
                        smoothed_v = debug_info['smoothed_velocity']
                        cv2.putText(gripper_vis, f'Velocity: {smoothed_v:.1f} px/s', (20, 120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                        cv2.putText(gripper_vis, f'Threshold: +/-{velocity_threshold:.1f}', (20, 160), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                        
                        # Show individual marker velocities
                        for i, vel_info in enumerate(debug_info['velocities']):
                            marker_id = vel_info['marker_id']
                            adj_vel = vel_info['adjusted_velocity']
                            cv2.putText(gripper_vis, f'M{marker_id}: {adj_vel:.1f}', (20, 200 + i*30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
                
                
        # --- 기존 pose 추정 ---
        image_with_axes, aruco_pose = detect_markers_and_estimate_pose(image, K_azure, D_azure)
        if aruco_pose is not None:
            tcp_pose = transform_aruco_to_tcp(
                aruco_pose,
                tcp_offset=np.array([0.0, -0.13423, 0.135])  # 괄호 닫기 수정
            )
            # gripper_state를 pose 딕셔너리 안에 추가 (open=1, close=0)
            tcp_pose['gripper_state'] = 1 if gripper_state == 'open' else 0
            
            result = {
                "image_file": image_file,
                "pose": tcp_pose,
                "timestamp": time.time()
            }
            results.append(result)
            detected_count += 1
            
            # 시각화가 활성화된 경우에만 화면 표시 (최적화)
            if visualize:
                # 단순화된 시각화 - 리사이징 최소화
                if gripper_vis is not None:
                    # 작은 크기로 리사이징하여 성능 향상
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
    # 결과 저장
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Processed {len(image_files)} images: {detected_count} success, {failed_count} failed ({detected_count/len(image_files)*100:.1f}% success rate)")
    
    # 시각화가 활성화된 경우에만 창 닫기
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
    
    for episode_dir in episode_dirs:
        print(f"\nProcessing {episode_dir}...")
        episode_path = os.path.join(base_path, episode_dir)
        image_directory = os.path.join(episode_path, 'images', 'pose_tracking')
        output_json = os.path.join(episode_path, 'raw_pose.json')
        if not os.path.exists(image_directory):
            print(f"Skipping {episode_dir}: Image directory not found")
            continue
            
        process_images_from_directory(image_directory, output_json, visualize=args.visualize)
    
    elapsed_time = time.time() - start_time
    print(f"\n🎯 Processing completed!")
    print(f"Total time: {elapsed_time:.1f} seconds")

if __name__ == "__main__":
    main()