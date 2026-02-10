
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import json
import numpy as np
import os
import argparse
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

def load_pose_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def detect_outliers(poses, window_size=9, pos_threshold=3.5, ori_threshold=0.8, velocity_threshold=0.25):
    """
    이상치 감지 - ArUco 마커 튐 현상을 잡기 위해 속도와 윈도우 기반 감지를 강화
    """
    outliers = []
    n_poses = len(poses)
    
    def quaternion_distance(q1, q2):
        dot_product = abs(np.dot(q1, q2))
        return np.arccos(min(dot_product, 1.0)) * 2.0
    
    # 속도 기반 이상치 감지 (ArUco 튐 현상 잡기)
    for i in range(1, n_poses):
        if poses[i]['state'] is None or poses[i-1]['state'] is None:
            outliers.append(i)
            continue
        
        curr_pos = np.array(poses[i]['state']['position'])
        prev_pos = np.array(poses[i-1]['state']['position'])
        velocity = np.linalg.norm(curr_pos - prev_pos)
        
        if velocity > velocity_threshold:
            outliers.append(i)
            continue
    
    # 윈도우 기반 이상치 감지
    for i in range(n_poses):
        if i in outliers or poses[i]['state'] is None:
            continue
            
        start_idx = max(0, i - window_size // 2)
        end_idx = min(n_poses, i + window_size // 2 + 1)
        window_poses = [p['state'] for j, p in enumerate(poses[start_idx:end_idx]) if (start_idx + j) != i and p['state'] is not None]
        
        if len(window_poses) < 2:
            continue
            
        current_pose = poses[i]['state']
        window_positions = np.array([p['position'] for p in window_poses])
        window_orientations = np.array([p['orientation'] for p in window_poses])
        
        # 위치 이상치 체크
        pos_mean = np.mean(window_positions, axis=0)
        pos_std = np.std(window_positions, axis=0) + 1e-6
        current_pos = np.array(current_pose['position'])
        pos_zscore = np.abs(current_pos - pos_mean) / pos_std
        if np.any(pos_zscore > pos_threshold):
            outliers.append(i)
            continue
        
        # 방향 이상치 체크
        current_ori = np.array(current_pose['orientation']) / np.linalg.norm(current_pose['orientation'])
        ori_distances = [quaternion_distance(current_ori, ori / np.linalg.norm(ori)) for ori in window_orientations]
        if np.mean(ori_distances) > ori_threshold:
            outliers.append(i)
    
    return sorted(list(set(outliers)))

def interpolate_missing_poses(poses):
    """
    누락된 pose 보간 - 시간 오프셋 방지를 위해 원래 타임스탬프 사용
    """
    timestamps = np.array([p['timestamp'] for p in poses])
    valid_indices = [i for i, p in enumerate(poses) if p['state'] is not None]
    
    if len(valid_indices) < 2:
        return
    
    valid_times = timestamps[valid_indices]
    valid_positions = np.array([poses[i]['state']['position'] for i in valid_indices])
    valid_orientations = np.array([poses[i]['state']['orientation'] for i in valid_indices])
    
    # 위치 보간
    position_interp = interp1d(valid_times, valid_positions, axis=0, kind='linear', bounds_error=False, fill_value="extrapolate")
    
    # 방향 보간 (SLERP)
    for i in range(1, len(valid_orientations)):
        if np.dot(valid_orientations[i], valid_orientations[i-1]) < 0:
            valid_orientations[i] = -valid_orientations[i]
    
    # --- 타임스탬프 정렬 및 중복 제거 ---
    valid_times, unique_idx = np.unique(valid_times, return_index=True)
    valid_orientations = valid_orientations[unique_idx]
    
    rotations = Rotation.from_quat(valid_orientations)
    slerp = Slerp(valid_times, rotations)
    
    # 그리퍼 상태 보간 (가장 가까운 유효한 값 사용)
    valid_gripper_states = [poses[i]['gripper_state'] for i in valid_indices]
    valid_gripper_states = np.array(valid_gripper_states)[unique_idx]  # 같은 인덱스로 정렬
    gripper_interp = interp1d(valid_times, valid_gripper_states, kind='nearest', bounds_error=False, fill_value="extrapolate")
    
    # 보간 적용
    for i in range(len(poses)):
        if poses[i]['state'] is None:
            t = timestamps[i]
            if valid_times[0] <= t <= valid_times[-1]:
                poses[i]['state'] = {
                    'position': position_interp(t).tolist(), 
                    'orientation': slerp(t).as_quat().tolist()
                }
                # 그리퍼 상태도 보간 (가장 가까운 값으로)
                poses[i]['gripper_state'] = float(gripper_interp(t))
            else:
                # 양 끝단 처리: 가장 가까운 유효한 값 사용
                if t < valid_times[0]:
                    # 시작 부분: 첫 번째 유효한 값 사용
                    closest_idx = valid_indices[0]
                else:
                    # 끝 부분: 마지막 유효한 값 사용
                    closest_idx = valid_indices[-1]
                
                poses[i]['state'] = {
                    'position': poses[closest_idx]['state']['position'].copy(),
                    'orientation': poses[closest_idx]['state']['orientation'].copy()
                }
                poses[i]['gripper_state'] = poses[closest_idx]['gripper_state']

def smooth_trajectories(poses, window=11, poly_order=2, use_ema=False, alpha=0.7):
    """
    궤적 스무딩 - Savitzky-Golay와 EMA 선택 가능
    """
    valid_poses = [p for p in poses if p['state'] is not None]
    if not valid_poses:
        return
    
    timestamps = np.array([p['timestamp'] for p in valid_poses])
    positions = np.array([p['state']['position'] for p in valid_poses])
    orientations = np.array([p['state']['orientation'] for p in valid_poses])
    
    if use_ema:
        # EMA 적용
        smoothed_positions = positions.copy()
        smoothed_orientations = orientations.copy()
        for i in range(1, len(positions)):
            smoothed_positions[i] = alpha * positions[i] + (1 - alpha) * smoothed_positions[i-1]
            ori = alpha * orientations[i] + (1 - alpha) * smoothed_orientations[i-1]
            smoothed_orientations[i] = ori / np.linalg.norm(ori)
    else:
        # Savitzky-Golay 적용
        print(positions.shape)
        smoothed_positions = savgol_filter(positions, window, poly_order, axis=0)
        for i in range(1, len(orientations)):
            if np.dot(orientations[i], orientations[i-1]) < 0:
                orientations[i] = -orientations[i]
        smoothed_orientations = savgol_filter(orientations, window, poly_order, axis=0)
        norms = np.linalg.norm(smoothed_orientations, axis=1)
        smoothed_orientations /= norms[:, np.newaxis]
    
    # 결과 반영
    idx = 0
    for i in range(len(poses)):
        if poses[i]['state'] is not None:
            poses[i]['state']['position'] = smoothed_positions[idx].tolist()
            poses[i]['state']['orientation'] = smoothed_orientations[idx].tolist()
            idx += 1

def calculate_relative_pose(poses):
    formatted_poses = []
    if not poses:
        return formatted_poses

    # 첫 번째 유효한 포즈 찾기
    start_pose_idx = None
    for i, pose in enumerate(poses):
        if pose['state'] is not None:
            start_pose_idx = i
            break
    
    if start_pose_idx is None:
        print("Warning: No valid poses found!")
        return formatted_poses

    # 시작점 저장
    start_pos = np.array(poses[start_pose_idx]['state']['position'])
    start_ori = np.array(poses[start_pose_idx]['state']['orientation'])
    start_rot = Rotation.from_quat(start_ori)

    for i in range(len(poses)):  # 모든 프레임 처리하도록 수정
        if poses[i]['state'] is None:
            continue
            
        curr_pos = np.array(poses[i]['state']['position'])
        curr_ori = np.array(poses[i]['state']['orientation'])
        
        # 시작점-현재 프레임 간의 상대 자세 (상태 표현용) - 항상 계산
        pos_wrt_start = curr_pos - start_pos  # 단순히 위치 차이
        rot_wrt_start = start_rot.inv() * Rotation.from_quat(curr_ori)
        
        # 현재-다음 프레임 간의 상대 변화량 (액션용)
        if i < len(poses) - 1 and poses[i+1]['state'] is not None:
            next_pos = np.array(poses[i+1]['state']['position'])
            next_ori = np.array(poses[i+1]['state']['orientation'])
            
            curr_rot = Rotation.from_quat(curr_ori)
            relative_position = curr_rot.inv().apply(next_pos - curr_pos)
            # relative_position[1] = -relative_position[1]
            relative_rotation = (curr_rot.inv() * Rotation.from_quat(next_ori))
            
            action_data = {
                'relative_position': relative_position.tolist(),
                'relative_orientation': relative_rotation.as_quat().tolist()
            }
        else:
            # 마지막 프레임이거나 다음 프레임이 없는 경우 영벡터 사용
            action_data = {
                'relative_position': [0.0, 0.0, 0.0],
                'relative_orientation': [0.0, 0.0, 0.0, 1.0]
            }

        
        formatted_pose = {
            'timestamp': poses[i]['timestamp'],
            'image_file': poses[i]['image_file'],
            'pose': {
                'position': poses[i]['state']['position'],
                'orientation': poses[i]['state']['orientation'],
                'gripper_state': poses[i]['gripper_state']
            },
            'state': {
                'position': pos_wrt_start.tolist(),
                'orientation': rot_wrt_start.as_quat().tolist()
            },
            'action': action_data,
            'pose_wrt_start': {
                'position': pos_wrt_start.tolist(),
                'orientation': rot_wrt_start.as_quat().tolist()
            }
        }
        formatted_poses.append(formatted_pose)
    
    return formatted_poses

def state_interpolation_wrt_img(poses, episode_path=None):
    """
    매 에피소드마다 이미지 갯수와 state 갯수를 비교해서 state가 이미지갯수보다 작으면, 
    해당 이미지에 해당하는 state를 앞뒤 값을 사용해서 보간.
    최종적으로 len(image) = len(state)가 돼야함.
    """
    if not poses or not episode_path:
        return poses
    
    # pose_tracking 디렉토리에서 실제 이미지 파일들 가져오기
    pose_tracking_dir = os.path.join(episode_path, 'images', 'pose_tracking')
    if not os.path.exists(pose_tracking_dir):
        print(f"  - Warning: pose_tracking directory not found")
        return poses
    
    actual_images = sorted([f for f in os.listdir(pose_tracking_dir) if f.endswith(('.jpg', '.png'))])
    actual_image_count = len(actual_images)
    print(f"  - Actual images in directory: {actual_image_count}")
    
    # 현재 pose 데이터의 이미지 파일 갯수
    current_pose_count = len(poses)
    print(f"  - Current poses in data: {current_pose_count}")
    
    # 기존 pose 데이터를 파일명으로 인덱싱
    pose_dict = {}
    for pose in poses:
        if 'image_file' in pose:
            pose_dict[pose['image_file']] = pose
    
    # 새로운 pose 리스트 생성 (모든 이미지에 대해)
    new_poses = []
    
    for image_file in actual_images:
        if image_file in pose_dict:
            # 기존 pose 데이터가 있는 경우
            new_poses.append(pose_dict[image_file])
        else:
            # pose 데이터가 없는 경우, 보간 필요
            # 타임스탬프 추출 (파일명에서)
            import re
            match = re.search(r'(\d+)', image_file)
            if match:
                timestamp = int(match.group(1))
            else:
                timestamp = len(new_poses)  # 인덱스 기반 대체
            
            # 빈 pose 생성 (나중에 보간)
            new_pose = {
                'timestamp': timestamp,
                'image_file': image_file,
                'state': None,
                'gripper_state': 1.0  # 기본값
            }
            new_poses.append(new_pose)
    
    print(f"  - Created {len(new_poses)} poses for all images")
    
    # 이제 누락된 pose들을 보간
    valid_poses = [p for p in new_poses if p['state'] is not None]
    invalid_poses = [p for p in new_poses if p['state'] is None]
    
    print(f"  - Valid poses: {len(valid_poses)}")
    print(f"  - Invalid poses (need interpolation): {len(invalid_poses)}")
    
    if len(valid_poses) < 2:
        print("  - Warning: Not enough valid poses for interpolation")
        return new_poses
    
    # 각 이미지에 대해 state 보간
    interpolated_count = 0
    copied_prev_count = 0
    copied_next_count = 0
    
    for i in range(len(new_poses)):
        if new_poses[i]['state'] is None:
            # 앞뒤 유효한 state 찾기
            prev_valid_idx = None
            next_valid_idx = None
            
            # 이전 유효한 state 찾기
            for j in range(i-1, -1, -1):
                if new_poses[j]['state'] is not None:
                    prev_valid_idx = j
                    break
            
            # 다음 유효한 state 찾기
            for j in range(i+1, len(new_poses)):
                if new_poses[j]['state'] is not None:
                    next_valid_idx = j
                    break
            
            # 보간 수행
            if prev_valid_idx is not None and next_valid_idx is not None:
                # 선형 보간
                alpha = (i - prev_valid_idx) / (next_valid_idx - prev_valid_idx)
                
                prev_pos = np.array(new_poses[prev_valid_idx]['state']['position'])
                next_pos = np.array(new_poses[next_valid_idx]['state']['position'])
                interpolated_pos = prev_pos + alpha * (next_pos - prev_pos)
                
                prev_ori = np.array(new_poses[prev_valid_idx]['state']['orientation'])
                next_ori = np.array(new_poses[next_valid_idx]['state']['orientation'])
                
                # 쿼터니언 SLERP (SciPy 방식)
                # 쿼터니언의 부호 정규화
                if np.dot(prev_ori, next_ori) < 0:
                    next_ori = -next_ori
                
                # Slerp 객체 생성 및 보간
                key_times = [0, 1]
                key_rots = Rotation.from_quat([prev_ori, next_ori])
                slerp = Slerp(key_times, key_rots)
                interpolated_rot = slerp([alpha])[0]
                interpolated_ori = interpolated_rot.as_quat()
                
                # 그리퍼 상태 보간
                prev_gripper = new_poses[prev_valid_idx]['gripper_state']
                next_gripper = new_poses[next_valid_idx]['gripper_state']
                interpolated_gripper = prev_gripper + alpha * (next_gripper - prev_gripper)
                
                new_poses[i]['state'] = {
                    'position': interpolated_pos.tolist(),
                    'orientation': interpolated_ori.tolist()
                }
                new_poses[i]['gripper_state'] = interpolated_gripper
                interpolated_count += 1
                
            elif prev_valid_idx is not None:
                # 이전 값만 있는 경우 복사
                new_poses[i]['state'] = {
                    'position': new_poses[prev_valid_idx]['state']['position'].copy(),
                    'orientation': new_poses[prev_valid_idx]['state']['orientation'].copy()
                }
                new_poses[i]['gripper_state'] = new_poses[prev_valid_idx]['gripper_state']
                copied_prev_count += 1
                
            elif next_valid_idx is not None:
                # 다음 값만 있는 경우 복사
                new_poses[i]['state'] = {
                    'position': new_poses[next_valid_idx]['state']['position'].copy(),
                    'orientation': new_poses[next_valid_idx]['state']['orientation'].copy()
                }
                new_poses[i]['gripper_state'] = new_poses[next_valid_idx]['gripper_state']
                copied_next_count += 1
    
    # 보간 결과 출력
    print(f"  - Interpolation results:")
    print(f"    * Linear interpolated: {interpolated_count}")
    print(f"    * Copied from previous: {copied_prev_count}")
    print(f"    * Copied from next: {copied_next_count}")
    print(f"    * Total filled: {interpolated_count + copied_prev_count + copied_next_count}")
    
    return new_poses

def save_pose_data(poses, output_path):
    print(f"Saving {len(poses)} poses to {output_path}")  # 디버깅 출력
    if len(poses) == 0:
        print("Warning: No poses to save!")
        return
    with open(output_path, 'w') as f:
        json.dump(poses, f, indent=4)
    print(f"Successfully saved to {output_path}")  # 디버깅 출력

def main():
    parser = argparse.ArgumentParser(description='Refine pose data from raw measurements')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory')
    args = parser.parse_args()
    
    base_dir = args.data_path
    episode_dirs = [d for d in os.listdir(base_dir) if d.startswith('episode_')]
    
    for episode_dir in episode_dirs:
        episode_path = os.path.join(base_dir, episode_dir)
        input_json = os.path.join(episode_path, 'raw_pose.json')
        output_json = os.path.join(episode_path, 'pose_data.json')
        
        print(f"\nProcessing {episode_dir}...")
        
        if not os.path.exists(input_json):
            print(f"Input file does not exist: {input_json}")
            continue
            
        # 데이터 로드 및 구조 조정
        raw_poses = load_pose_data(input_json)
        print(f"Loaded {len(raw_poses)} raw poses from {input_json}")
        poses = []
        for p in raw_poses:
            pose_data = {
                'timestamp': p['timestamp'], 
                'image_file': p['image_file'], 
                'state': {
                    'position': p['pose']['position'],
                    'orientation': p['pose']['orientation']
                }
            }
            # gripper_state 보존
            if 'gripper_state' in p['pose']:
                pose_data['gripper_state'] = p['pose']['gripper_state']
            else:
                pose_data['gripper_state'] = 1.0  # 기본값 (open)
            poses.append(pose_data)
        
        # 1. 이상치 감지 및 제거
        outliers = detect_outliers(poses)
        for idx in outliers:
            poses[idx]['state'] = None
        
        # 2. 이미지에 맞춘 state 보간 (먼저 행 개수 맞추기)
        poses = state_interpolation_wrt_img(poses, episode_path)
        
        # 3. 누락된 pose 보간 (SLERP로 일관성 확보)
        interpolate_missing_poses(poses)
        
        # 4. 궤적 스무딩
        smooth_trajectories(poses, window=20, poly_order=2, use_ema=False, alpha=0.6)
        
        # 5. 상대 pose 계산 (액션 생성)
        relative_poses = calculate_relative_pose(poses)
        
        # 최종 이미지와 state 갯수 로깅
        total_images = len(poses)
        valid_states = sum(1 for p in poses if p['state'] is not None)
        final_poses = len(relative_poses)
        
        print(f"Final counts:")
        print(f"  - Total images: {total_images}")
        print(f"  - Valid states: {valid_states}")
        print(f"  - Final poses with actions: {final_poses}")
        print(f"  - Image-state alignment: {'✓' if total_images == valid_states else '✗'}")
        
        # 6. pose_data.json에 저장
        save_pose_data(relative_poses, output_json)
        print(f"Saved refined relative poses to {output_json}")

if __name__ == "__main__":
    main()