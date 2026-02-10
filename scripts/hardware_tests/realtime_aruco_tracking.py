
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import cv2
import cv2.aruco as aruco
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
import time
import glob

# ArUco 마커 설정
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_100)

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


def find_first_video() -> str:
    """가장 번호가 작은 /dev/video* 디바이스를 찾습니다."""
    vids = sorted(glob.glob("/dev/video*"))
    if not vids:
        raise SystemExit("비디오 장치를 찾지 못했습니다.")
    return vids[0]


def reduce_overexposure(image_bgr: np.ndarray) -> np.ndarray:
    """하드웨어 노출 제어가 불가할 때, 과노출 완화를 위해 V 채널을 동적으로 낮춤."""
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32)
    mean_v = float(v.mean())
    target_mean = 120.0  # 더 어둡게 목표 설정
    if mean_v > target_mean:
        factor = max(0.3, target_mean / (mean_v + 1e-6))  # 0.3 ~ 1.0 사이로 더 공격적 축소
        v = np.clip(v * factor, 0, 255)
        hsv[..., 2] = v.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return image_bgr

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

    
    if marker_id not in transforms:
            print(f"마커 ID {marker_id}는 지원되지 않습니다.")
            return None
        
    transform = transforms[marker_id]
    
    # 마커의 현재 위치와 방향
    marker_pos = np.array(marker_pose['position'])
    marker_rot = R.from_quat(marker_pose['orientation']).as_matrix()
    
    # 디버깅 출력: 입력 데이터
    print(f"\n=== 마커 ID {marker_id} ===")
    print(f"마커 위치 (입력): {marker_pos}")
    print(f"마커 회전 행렬 (입력):\n{marker_rot}")
    print(f"적용할 변환 Translation: {transform['translation']}")
    print(f"적용할 변환 Rotation:\n{transform['rotation']}")
    
    # 큐브 중심 기준 좌표계로 변환
    cube_rot = marker_rot @ transform['rotation']
    cube_pos = marker_pos + (marker_rot @ transform['translation'])
    
    
    # 디버깅 출력: 출력 데이터
    print(f"큐브 중심 위치 (출력): {cube_pos}")
    print(f"큐브 중심 회전 행렬 (출력):\n{cube_rot}")
    
    return {
        'position': cube_pos,
        'orientation': R.from_matrix(cube_rot).as_quat()
    }
def normalize_quaternion_sign(curr_quat, ref_quat):
    """현재 쿼터니언을 기준 쿼터니언과 부호를 일치시킴"""
    if np.dot(curr_quat, ref_quat) < 0:
        return -np.array(curr_quat)
    return np.array(curr_quat)

def detect_markers_and_estimate_pose(image, K, D, prev_orientation=None):
    global T_cam_to_cube_initial

    # 1) Fast coarse detection on a downscaled gray image (no refinement)
    gray_full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = 0.4  # 더 작게 스케일링하여 속도 향상
    gray = cv2.resize(gray_full, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    params = aruco.DetectorParameters()
    # coarse settings - 더 빠른 검출을 위해 조정
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 15
    params.adaptiveThreshWinSizeStep = 2
    params.adaptiveThreshConstant = 7
    params.minMarkerPerimeterRate = 0.03  # 조금 더 큰 마커만 검출
    params.maxMarkerPerimeterRate = 0.5
    # NO corner refinement here
    params.cornerRefinementMethod = aruco.CORNER_REFINE_NONE

    corners_coarse, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)
    if ids is None:
        return image, None

    # 2) Upscale corner coordinates, crop tight ROIs and run APRILTAG refinement there
    refined_corners = []
    refined_ids     = []
    for corner_small, mid in zip(corners_coarse, ids.flatten()):
        # scale up to full res
        corner = (corner_small.reshape(-1,2) / scale).astype(int)

        # compute a tight bounding box + padding
        x, y, w, h = cv2.boundingRect(corner)
        pad = int(max(w,h)*0.3)
        x0, y0 = max(x-pad,0), max(y-pad,0)
        x1 = min(x+w+pad, image.shape[1])
        y1 = min(y+h+pad, image.shape[0])
        roi = gray_full[y0:y1, x0:x1]

        # detect & refine corners inside ROI
        params2 = aruco.DetectorParameters()
        params2.cornerRefinementMethod      = aruco.CORNER_REFINE_APRILTAG
        params2.cornerRefinementWinSize     = 3  # 더 작은 윈도우
        params2.cornerRefinementMaxIterations = 15  # 더 적은 반복
        params2.cornerRefinementMinAccuracy   = 0.01  # 더 낮은 정확도 허용

        # shift ROI back to full-image coords
        corners_refined, ids_r, _ = aruco.detectMarkers(roi, dictionary, parameters=params2)
        if ids_r is None or mid not in ids_r:
            continue
        # pull out the one you want
        idx = list(ids_r.flatten()).index(mid)
        cr = corners_refined[idx].reshape(-1,2) + np.array([x0,y0])
        refined_corners.append(cr)
        refined_ids.append(mid)

    if not refined_ids:
        return image, None

    # 3) Estimate pose from refined corners
    cube_centers = []
    marker_size = 0.0483
    
    # refined_corners를 올바른 형식으로 변환
    corners_for_drawing = []
    for corner in refined_corners:
        # reshape to correct format (1,4,2) and convert to float32
        corner_reshaped = corner.reshape(1,4,2).astype(np.float32)
        corners_for_drawing.append(corner_reshaped)
    
    for corner, mid in zip(refined_corners, refined_ids):
        # 그리퍼 마커(6, 7)는 큐브 마커가 아니므로 건너뛰기
        if mid in [6, 7]:
            continue
            
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            np.array([corner], dtype=np.float32), marker_size, K, D)
        Rcm, _ = cv2.Rodrigues(rvecs[0][0])
        mrk = {
            'position': tvecs[0][0],
            'orientation': R.from_matrix(Rcm).as_quat()
        }
        cube = transform_to_cube_center(mid, mrk)
        if cube: cube_centers.append(cube)

    if not cube_centers:
        return image, None

    # 4) Fuse and build your local-frame transform exactly as before
    ref_q = cube_centers[0]['orientation']
    for c in cube_centers:
        c['orientation'] = normalize_quaternion_sign(c['orientation'], ref_q)
    avg_p = np.mean([c['position'] for c in cube_centers], axis=0)
    avg_q = R.from_quat([c['orientation'] for c in cube_centers]).mean().as_quat()

    pos_local = avg_p
    quat_local = avg_q
    # 시각화 부분 수정
    vis = image.copy()
    vis = aruco.drawDetectedMarkers(vis, corners_for_drawing, np.array(refined_ids))
    rvec, _ = cv2.Rodrigues(R.from_quat(quat_local).as_matrix())
    tvec = pos_local
    cv2.drawFrameAxes(vis, K, D, rvec, tvec, 0.06)

    return vis, {'position': pos_local.tolist(),
                 'orientation': quat_local.tolist()}

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
        tcp_offset = np.array([0.0, -0.136, 0.11])  # hand–eye 캘리브레이션 오프셋 #긴게 0.13, 짧은게 0.11
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


def main():
    # DJI 액션캠 intrinsic 로드
    K_cam, D_cam = load_dji_intrinsics("calibration/calibration_dji_intrinsics.json")
    print("[INFO] DJI camera intrinsics loaded.")
    print("K:\n", K_cam)
    print("dist:", D_cam)

    # 비디오 장치 열기 (/dev/video* 중 첫 번째)
    device = find_first_video()
    print(f"[INFO] Using video device: {device}")
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[ERROR] 카메라를 열 수 없습니다: {device}")
        return

    # 내부 버퍼 줄이기 (지연 감소)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        ret2, frame2 = cap.read()

        # 프레임이 유효하지 않으면 다음 루프로 진행
        if not ret2 or frame2 is None or frame2.size == 0:
            continue

        # 하드웨어 제어 불가 시 소프트웨어로 과노출 완화
        frame2 = reduce_overexposure(frame2)

        # 마커 감지 및 자세 추정
        image_with_axes, aruco_pose = detect_markers_and_estimate_pose(frame2, K_cam, D_cam)
 
        # TCP 포즈 변환 및 두 좌표계 모두 표시
        if aruco_pose is not None:
            tcp_pose = transform_aruco_to_tcp(
                aruco_pose,
                tcp_offset=np.array([0.0, -0.13423, 0.135])  # 괄호 닫기 수정
            )
            
            # TCP 좌표계도 화면에 그리기
            tcp_rvec = cv2.Rodrigues(R.from_quat(tcp_pose['orientation']).as_matrix())[0]
            tcp_tvec = np.array(tcp_pose['position'])
            cv2.drawFrameAxes(image_with_axes, K_cam, D_cam, tcp_rvec, tcp_tvec, 0.08)  # TCP는 좀 더 크게

        # 두 영상을 가로로 합치기
        cv2.imshow('DJI Realtime ArUco Tracking', image_with_axes)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
     
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()