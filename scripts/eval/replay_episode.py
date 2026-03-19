#!/usr/bin/env python3
"""
학습 데이터 에피소드를 로봇에 리플레이하는 스크립트.

Mode A (--mode direct):  pose_wrt_start를 직접 적용 (Ground Truth)
Mode B (--mode action):  action delta를 transform_pose 경유해서 적용 (인퍼런스와 동일)

두 모드의 결과가 같으면 transform_pose가 올바른 것.

Usage:
  # Ground Truth 리플레이
  python scripts/eval/replay_episode.py \
    --zarr_path /home/ailab-2204/Workspace/ManipForce_v2/data/battery_assem.zarr \
    --episode 0 --mode direct

  # 인퍼런스 파이프라인 리플레이 (transform_pose 경유)
  python scripts/eval/replay_episode.py \
    --zarr_path /home/ailab-2204/Workspace/ManipForce_v2/data/battery_assem.zarr \
    --episode 0 --mode action \
    --config_path eval_config/battery_assemb.yaml
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import json
import socket
import time
import numpy as np
import yaml
import zarr
from scipy.spatial.transform import Rotation

from utils.franka_api import FrankaAPI


# ── transform_pose: eval_robot.py와 동일 ──
def transform_pose(pose, current_ee_pose, config):
    R_mat_pos = np.array(config['coordinate_transform']['R_mat_pos'])
    R_mat_rot = np.array(config['coordinate_transform']['R_mat_rot'])

    # Position
    rel_pos = R_mat_pos @ pose[:3]

    if 'position_x_scale' in config['action'] and 'position_y_scale' in config['action']:
        px = config['action']['position_x_scale']
        py = config['action']['position_y_scale']
    else:
        pxy = config['action']['position_xy_scale']
        px = py = pxy
    pz = config['action']['position_z_scale']

    rel_pos = np.array([rel_pos[0] * px, rel_pos[1] * py, rel_pos[2] * pz])

    # Rotation
    if 'rotation_x_scale' in config['action'] and 'rotation_y_scale' in config['action']:
        rx = config['action']['rotation_x_scale']
        ry = config['action']['rotation_y_scale']
    else:
        rxy = config['action']['rotation_xy_scale']
        rx = ry = rxy
    rz = config['action']['rotation_z_scale']

    delta_quat = np.array(pose[3:7])
    rotvec = Rotation.from_quat(delta_quat).as_rotvec()
    rotvec[0] *= rx
    rotvec[1] *= ry
    rotvec[2] *= rz

    R_scaled = Rotation.from_rotvec(rotvec).as_matrix()
    R_new = R_mat_rot @ R_scaled @ R_mat_rot.T
    rel_quat = Rotation.from_matrix(R_new).as_quat()

    return np.concatenate([rel_pos, rel_quat])


def connect_to_server(host, port=4999, timeout=5):
    while True:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        client_socket.settimeout(timeout)
        try:
            client_socket.connect((host, port))
            print(f"[연결] 로봇 서버 연결 성공: {host}:{port}")
            return client_socket
        except (ConnectionRefusedError, socket.timeout) as e:
            print(f"연결 실패: {host}:{port} ({e}). 1초 후 재시도...")
            time.sleep(1)


def load_episode(zarr_path, episode_idx):
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store)

    episode_ends = root['meta']['episode_ends'][:]
    n_episodes = len(episode_ends)

    if episode_idx < 0 or episode_idx >= n_episodes:
        raise ValueError(f"에피소드 {episode_idx}는 범위 밖 (0~{n_episodes-1})")

    start = 0 if episode_idx == 0 else int(episode_ends[episode_idx - 1])
    end = int(episode_ends[episode_idx])

    pose_wrt_start = root['data']['pose_wrt_start'][start:end]
    state = root['data']['state'][start:end]
    action = root['data']['action'][start:end]
    timestamps = root['data']['img_timestamps'][start:end]

    print(f"\n=== 에피소드 {episode_idx} ===")
    print(f"  길이: {end - start} steps")
    print(f"  pose_wrt_start[0]: {pose_wrt_start[0]}")
    print(f"  pose_wrt_start[-1]: {pose_wrt_start[-1]}")
    print(f"  action[0]: {action[0]}")
    print(f"  action shape: {action.shape}")

    if len(timestamps) > 1:
        diffs = np.diff(timestamps)
        median_dt = np.median(diffs)
        print(f"  타임스탬프 간격: {median_dt:.1f}ms ({1000.0/max(median_dt, 1):.1f} Hz)")

    return pose_wrt_start, state, action, timestamps


def send_pose(client_socket, target_pose, server_ip, server_port):
    try:
        data = {
            'target_pose': target_pose.tolist(),
            'gripper_command': 'keep',
            'reset': False,
            'timestamp': time.time(),
        }
        message = json.dumps(data, separators=(',', ':'))
        client_socket.sendall(message.encode('utf-8') + b'\n')
        return client_socket
    except (socket.error, BrokenPipeError) as e:
        print(f"\n[오류] 서버 연결 오류: {e}")
        return connect_to_server(server_ip, server_port)


def main():
    parser = argparse.ArgumentParser(description="학습 데이터 에피소드 리플레이")
    parser.add_argument('--zarr_path', type=str, required=True)
    parser.add_argument('--episode', type=int, default=0)
    parser.add_argument('--mode', type=str, default='direct',
                        choices=['direct', 'action'],
                        help='direct=pose_wrt_start 직접, action=action delta+transform_pose')
    parser.add_argument('--config_path', type=str, default=None,
                        help='action 모드에서 사용할 yaml config (transform_pose용)')
    parser.add_argument('--server_ip', type=str, default='172.27.190.125')
    parser.add_argument('--server_port', type=int, default=4999)
    parser.add_argument('--http_port', type=int, default=5000)
    parser.add_argument('--speed', type=float, default=1.0)
    parser.add_argument('--dry_run', action='store_true',
                        help='로봇에 안 보내고 trajectory만 출력')
    args = parser.parse_args()

    # Config 로드 (action 모드)
    config = None
    if args.mode == 'action':
        if args.config_path is None:
            print("[오류] --mode action 사용 시 --config_path 필요")
            sys.exit(1)
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"[Config] {args.config_path}")
        print(f"  R_mat_pos: {config['coordinate_transform']['R_mat_pos']}")
        print(f"  R_mat_rot: {config['coordinate_transform']['R_mat_rot']}")
        print(f"  action scales: {dict((k,v) for k,v in config['action'].items() if 'scale' in k)}")

    # 데이터 로드
    pose_wrt_start, state, action_data, timestamps = load_episode(args.zarr_path, args.episode)
    n_steps = len(pose_wrt_start)

    # 로봇 연결
    client_socket = None
    if not args.dry_run:
        franka_api = FrankaAPI(args.server_ip, rest_port=args.http_port)
        client_socket = connect_to_server(args.server_ip, args.server_port)
        init_pose = np.array(franka_api.get_pose_sync(), dtype=np.float64)
    else:
        init_pose = np.array([0.54, 0.02, -0.1, 0.0, 0.0, 0.17365, 0.98481], dtype=np.float64)
        print("\n[DRY RUN] 로봇에 보내지 않음")

    init_rot = Rotation.from_quat(init_pose[3:])
    print(f"\n[초기 위치] pos={init_pose[:3]} | quat={init_pose[3:]}")
    print(f"[모드] {'DIRECT (Ground Truth)' if args.mode == 'direct' else 'ACTION (transform_pose 경유)'}")
    print(f"\n{n_steps}개 스텝을 {args.speed}x 속도로 리플레이합니다.")
    input("Enter를 누르면 시작합니다...")

    # ── 리플레이 루프 ──
    dt = 1.0 / 30.0 / args.speed
    prev_target = init_pose.copy()

    # trajectory 기록 (비교용)
    trajectory = []

    for step in range(n_steps):
        t_start = time.time()

        if args.mode == 'direct':
            # ── Mode A: pose_wrt_start 직접 적용 ──
            rel_pos = pose_wrt_start[step, :3]
            rel_quat = pose_wrt_start[step, 3:7]
            rel_rot = Rotation.from_quat(rel_quat)

            abs_pos = init_pose[:3] + rel_pos
            abs_rot = init_rot * rel_rot
            target_pose = np.concatenate([abs_pos, abs_rot.as_quat()])

        else:
            # ── Mode B: action delta → transform_pose → 누적 ──
            # action: [dx, dy, dz, qx, qy, qz, qw, gripper]
            act = action_data[step]
            action_7d = act[:7]  # [dx, dy, dz, qx, qy, qz, qw]

            transformed = transform_pose(action_7d, prev_target, config)

            target_pose = prev_target.copy()
            target_pose[:3] += transformed[:3]

            cur_rot = Rotation.from_quat(prev_target[3:])
            delta_rot = Rotation.from_quat(transformed[3:])
            new_rot = cur_rot * delta_rot
            target_pose[3:] = new_rot.as_quat()

            prev_target = target_pose.copy()

        trajectory.append(target_pose.copy())

        if step % 10 == 0 or step == n_steps - 1:
            pos = target_pose[:3]
            print(f"  [{step:4d}/{n_steps}] pos=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

        if not args.dry_run:
            client_socket = send_pose(client_socket, target_pose,
                                      args.server_ip, args.server_port)

        elapsed = time.time() - t_start
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    print(f"\n[리플레이 완료] {n_steps} steps, 총 {n_steps * dt:.1f}초")

    # trajectory 저장
    traj = np.array(trajectory)
    out_file = f"replay_{args.mode}_ep{args.episode}.npy"
    np.save(out_file, traj)
    print(f"[저장] {out_file} ({traj.shape})")
    print(f"  최종 위치: {traj[-1, :3]}")
    print(f"  이동 범위 X: [{traj[:, 0].min():.4f}, {traj[:, 0].max():.4f}]")
    print(f"  이동 범위 Y: [{traj[:, 1].min():.4f}, {traj[:, 1].max():.4f}]")
    print(f"  이동 범위 Z: [{traj[:, 2].min():.4f}, {traj[:, 2].max():.4f}]")

    if not args.dry_run:
        client_socket.close()
        franka_api.stop()


if __name__ == '__main__':
    main()
