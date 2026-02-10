
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
from scipy.spatial.transform import Rotation  # 추가
import socket
import click
from typing import Optional, Dict, Any
from collections import deque
import threading

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from umi.real_world.real_inference_util import (get_real_gumi_obs_dict,
                                              get_real_gumi_action)
from franka_api import FrankaAPI

# 전역 변수 선언
prev_frames = []
prev_timestamps = []
current_img_idx = 0
BASE_DIR = None  # ← 설정에서 주입받을 데이터셋 경로


def transform_pose(pos, quat, current_ee_pose):
    """
    Position과 Quaternion을 새로운 좌표계로 변환
    pose: [px, py, pz, qw, qx, qy, qz]
    return: transformed [px, py, pz, qw, qx, qy, qz]
    """

    # # Don't change this (pushing, trajectory following)
    # R_mat = np.array([
    #     [ 0,  0, -1],  # x_robot = -z_original
    #     [ 1,  0,  0],  # y_robot =  x_original
    #     [ 0,  1,  0]   # z_robot =  y_original
    # ])
    

    # R_mat_trans = np.array([
    #         [0, 0, -1],   # new_x = 0*ox + 1*oy + 0*oz
    #         [1, 0, 0],   # new_y = 1*ox + 0*oy + 0*oz
    #         [0, -1, 0]   # new_z = 0*ox + 0*oy + -1*oz
    #     ])

    # 위치 매핑: (사용자 변경값 유지)
    R_mat_pos = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1]
        ])

    # 회전 매핑: 오른손 좌표계를 유지하는 정규 회전(det=+1)
    # z축 반전 없이 xy 교환(+90deg z-회전)
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

    # 현재 EE의 orientation을 반영
    current_quat = current_ee_pose[3:]  # [w, x, y, z]
    current_quat = np.roll(current_quat, -1)  # [x, y, z, w]로 변환
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
    quat = R.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w] 순서
    # 7D 벡터로 조합
    action_8d = np.concatenate([
        position,    # 위치 (3)
        quat,       # 쿼터니언 (4)
        gripper     # 그리퍼 (1)
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
    
    # GT 데이터 로드
    if not hasattr(get_obs, 'action_gt_data'):
        with open(os.path.join(base_dir, 'pose_data.json'), 'r') as f:
            get_obs.action_gt_data = json.load(f)
            print(f"전체 GT 데이터 개수: {len(get_obs.action_gt_data)}")
    
    # 이미지 파일 목록 가져오기
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
    
    # 버퍼가 비어있으면 초기화
    if len(prev_frames) == 0:
        for _ in range(n_obs_steps):
            if current_img_idx >= len(image_files):
                return None, None, None
            
            img_path = os.path.join(image_dir, image_files[current_img_idx])
            frame = cv2.imread(img_path)
            if frame is None:
                raise RuntimeError(f"이미지를 읽을 수 없습니다: {img_path}")
            
            prev_frames.append(frame)
            prev_timestamps.append(time.time())
            current_img_idx += 1
    
    # 현재 이미지 로드
    img_path = os.path.join(image_dir, image_files[current_img_idx])
    frame = cv2.imread(img_path)
    
    # pose tracking 이미지 로드
    pose_tracking_path = os.path.join(pose_tracking_dir, pose_tracking_files[current_img_idx])
    pose_tracking_frame = cv2.imread(pose_tracking_path)
    if pose_tracking_frame is None:
        raise RuntimeError(f"Pose tracking 이미지를 읽을 수 없습니다: {pose_tracking_path}")
    
    # GT action 찾기
    current_img_file = image_files[current_img_idx]
    current_gt = None
    for data in get_obs.action_gt_data:
        if data['image_file'] == current_img_file:
            current_gt = data['action']
            break
    
    # 버퍼 업데이트
    prev_frames.append(frame)
    prev_timestamps.append(time.time())
    prev_frames.pop(0)
    prev_timestamps.pop(0)
    
    # 다음 이미지 인덱스 업데이트
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
        print(f"서버 연결 성공: {ip}:{port}")
        return client_socket
    except Exception as e:
        print(f"서버 연결 실패: {e}")
        return None

def measure_zero_force(api, num_samples=100, delay=0.01):
    """FT 센서에서 num_samples만큼 읽어서 평균값을 zero_force로 반환"""
    print(f"[ZeroForce] 측정 시작... ({num_samples} samples)")
    ft_samples = []
    for _ in range(num_samples):
        f = api.get_force_sync()
        t = api.get_torque_sync()
        if f is not None and t is not None:
            ft_samples.append(np.concatenate([f, t]))
        time.sleep(delay)
    zero_force = np.mean(ft_samples, axis=0)
    print(f"[ZeroForce] 완료! zero_force = {zero_force}")
    return zero_force

# FTCollector 클래스 추가
class FTCollector:
    """FrankaAPI 로부터 힘 벡터를 원하는 Hz 로 계속 가져와 저장"""
    def __init__(self, api, rate_hz: float = 100., zero_force=None):
        self.api = api
        self.rate = rate_hz
        self.full_ts = []  # 모든 타임스탬프 저장
        self.full_ft = []  # 모든 FT 데이터 저장
        self.zero_force = zero_force  # ← 외부에서 전달받음
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
                # zero_force 보정
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
    """FTCollector.buf 안의 (timestamp, [fx,fy,fz,tx,ty,tz]) 전체를 CSV로 저장"""
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
    
    print(f"저장 완료 ▶ {filename} ({csv_data.shape[0]} samples)")

@click.command()
@click.option('--server_ip', '-si', default='172.27.190.125', help="서버 IP 주소")
@click.option('--server_port', '-sp', default=5001, type=int, help="서버 포트 번호")
@click.option('--operator', '-op', is_flag=True, help="수동 모드 (Enter로 다음 자세로 이동)")
@click.option('--base_dir', '-bd', default='/home/ailab-2204/Workspace/gail-umi/data/Test_Pose/episode_1', help='데이터셋 에피소드 경로 (images/* 포함)')
def main(server_ip, server_port, operator, base_dir):
    # 설정 경로 주입
    global BASE_DIR
    BASE_DIR = base_dir

    # 서버 연결
    client_socket = connect_to_server(server_ip, server_port)
    if client_socket is None:
        print("초기 연결 실패. 프로그램 종료.")
        return
    franka_api = FrankaAPI(server_ip)
    
    # === 1) zero_force 측정 ===
    zero_force = measure_zero_force(franka_api, num_samples=100, delay=0.01)

    # === 2) FTCollector에 zero_force 전달 ===
    ft_collector = FTCollector(franka_api, rate_hz=200, zero_force=zero_force)
    ft_collector.start()
    
    pose = franka_api.get_pose_sync()
    print(f"Current pose: {pose}")
    
    # 초기 포즈 설정
    initial_pose = np.array([0.56, -0.0,  0.17,  0.99959016, -0.01183819,  0.02564274, -0.00467247])

    # 먼저 initial pose로 이동
    print("초기 포즈로 이동합니다. Enter를 누르면 시작합니다.")
    input()
    
    try:
        # 초기 포즈 명령 전송
        data = {
            'target_pose': initial_pose.tolist(),
            'timestamp': time.time(),
            'gripper_command': False,
            'reset': False
        }
        message = json.dumps(data, separators=(',', ':'))
        client_socket.sendall(message.encode('utf-8') + b'\n')
        print(f"Initial pose로 이동 중: {initial_pose[:3].round(3)}")
        time.sleep(2.0)
        
        print("초기 포즈 도달 완료. 이미지 창에 포커스를 두고 Enter를 누르면 다음으로 진행됩니다.")
        cv2.namedWindow('Pose Tracking')
        input()  # 여기서 Enter를 누르면 실제 재생 시작
        
        prev_target_pose = initial_pose.copy()
        current_idx = 0
        
        while True:
            obs, gt_action, pose_tracking_frame = get_obs()
            
            if obs is None:
                print("\n데이터셋 처리 완료")
                # FT 데이터 저장
                save_ft_data(ft_collector, filename="replay_ft_data.csv")
                break
            
            if gt_action is not None:
                # 현재 end-effector pose 가져오기
                current_ee_pose = franka_api.get_pose_sync()
                
                # 현재 프레임과 다음 프레임의 GT 액션
                current_action = gt_action
                next_action = get_obs.action_gt_data[current_idx + 1]['action'] if current_idx + 1 < len(get_obs.action_gt_data) else None
                
                if next_action is not None:
                    # 현재 프레임의 상대 포즈
                    current_rel_pos = np.array(current_action['relative_position'])
                    current_rel_quat = np.array(current_action['relative_orientation'])
                    current_transformed = transform_pose(current_rel_pos, current_rel_quat, current_ee_pose)
                    
                    # 다음 프레임의 상대 포즈
                    next_rel_pos = np.array(next_action['relative_position'])
                    next_rel_quat = np.array(next_action['relative_orientation'])
                    next_transformed = transform_pose(next_rel_pos, next_rel_quat, current_ee_pose)
                    
                    # 현재 프레임 실행
                    this_target_pose = prev_target_pose.copy()
                    this_target_pose[:3] += current_transformed[:3]
                    transformed_rot = Rotation.from_quat(current_transformed[3:])
                    current_rot = Rotation.from_quat(prev_target_pose[3:])
                    new_rot = current_rot * transformed_rot
                    this_target_pose[3:] = new_rot.as_quat()
                    
                    # 서버로 명령 전송
                    data = {
                        'target_pose': this_target_pose.tolist(),
                        'timestamp': time.time(),
                        'gripper_command': False,
                        'reset': False
                    }
                    message = json.dumps(data, separators=(',', ':'))
                    client_socket.sendall(message.encode('utf-8') + b'\n')
                    
                    # 상태 출력 및 이미지 표시
                    print(f"Replay Step {current_idx} | Pos: {this_target_pose[:3].round(3)} | Quat: {this_target_pose[3:].round(3)}")
                    cv2.imshow('Pose Tracking', cv2.resize(pose_tracking_frame, (0,0), fx=0.5, fy=0.5))
                    
                    should_quit = False
                    
                    if operator:
                        # 수동 모드: Enter 키 대기 (한 스텝씩)
                        print("엔터를 누르면 다음 스텝으로 이동합니다. 'q'를 누르면 종료합니다.")
                        user_input = input().strip().lower()
                        if user_input == 'q':
                            print("\n사용자가 종료를 요청했습니다.")
                            should_quit = True
                    else:
                        # 자동 모드: GT action을 그대로 재생 (서버로 전송)
                        # pose_tracking 이미지 업데이트
                        cv2.imshow('Pose Tracking', cv2.resize(pose_tracking_frame, (0,0), fx=0.5, fy=0.5))
                        # 속도 조절: 10Hz로 재생 (0.1초 간격)
                        time.sleep(0.1)
                        key = cv2.waitKey(1) & 0xFF  # 1ms 대기
                        if key == ord('q'):
                            print("\n사용자가 종료를 요청했습니다.")
                            should_quit = True
                    
                    if should_quit:
                        break
                    
                    prev_target_pose = this_target_pose.copy()
                    current_idx += 1

    finally:
        # FT 데이터 수집 중지 및 저장
        save_ft_data(ft_collector, filename="replay_ft_data.csv")
        cv2.destroyAllWindows()
        if client_socket:
            client_socket.close()

if __name__ == "__main__":
    main()