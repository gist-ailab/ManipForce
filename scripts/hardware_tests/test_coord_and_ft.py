#!/usr/bin/env python3
"""
좌표계 & FT 정합성 진단 스크립트
================================
로봇을 각 축 방향(+x,-x,+y,-y,+z,-z)으로 순차 이동시키고,
이동 전/후의 FT 응답을 기록하여 좌표계 정합성을 검증한다.

검증 항목:
  1. 쿼터니온 컨벤션 (wxyz) 일관성 — get_pose_sync 반환값 검증
  2. 위치 이동 → FT 응답 축 대응 (로봇이 +x로 누르면 FT fx가 변하는지 등)
  3. 회전 이동 → FT 토크 축 대응
  4. 중력보상 후 잔차 확인
  5. 좌표 변환 행렬(R_mat_pos, R_mat_rot) 적용 전후 비교

사용법:
  python scripts/hardware_tests/test_coord_and_ft.py -c eval_config/gear_insertion_2.yaml
  python scripts/hardware_tests/test_coord_and_ft.py -c eval_config/gear_insertion_2.yaml --delta 0.01
  python scripts/hardware_tests/test_coord_and_ft.py -c eval_config/gear_insertion_2.yaml --skip-move  (FT만 확인)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import time
import json
import csv
import argparse
from datetime import datetime

import numpy as np
from scipy.spatial.transform import Rotation
import yaml
import pyrealsense2 as rs
import requests

from utils.franka_api import FrankaAPI
from utils.ft_capture import AidinFTSensorUDP
from utils.gravity_compensation_utils import GravityCompensator


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def pose_to_str(pose):
    """[x,y,z, qw,qx,qy,qz] → readable string."""
    return (f"pos=[{pose[0]:+.5f}, {pose[1]:+.5f}, {pose[2]:+.5f}]  "
            f"quat(wxyz)=[{pose[3]:+.5f}, {pose[4]:+.5f}, {pose[5]:+.5f}, {pose[6]:+.5f}]")


def quat_wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]])


def quat_xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])


def send_pose(server_ip, pose, port=5000, timeout=2.0):
    """Fire-and-forget pose command via REST."""
    url = f"http://{server_ip}:{port}/pose"
    requests.post(url, json={"arr": pose.tolist()}, timeout=timeout)


def collect_ft_samples(ft_collector, n_samples=50, rate_hz=200):
    """Collect n_samples of gravity-compensated FT at ~rate_hz. Returns (N,6) array."""
    samples = []
    dt = 1.0 / rate_hz
    for _ in range(n_samples):
        ft_buf, _ = ft_collector.window(1)
        if len(ft_buf) > 0:
            samples.append(ft_buf[-1].copy())
        time.sleep(dt)
    return np.array(samples, dtype=np.float32) if samples else np.zeros((1, 6), dtype=np.float32)


class SimpleFTCollector:
    """eval_robot.py의 FTCollector 간소화 버전 (그래프 없이 수집만)."""
    import threading

    def __init__(self, ft_reader, imu_pipe, gravity_compensator, rate_hz=200, buf_len=500):
        self.ft_reader = ft_reader
        self.imu_pipe = imu_pipe
        self.gc = gravity_compensator
        self.rate_hz = rate_hz
        self._th = None
        self._stop = SimpleFTCollector.threading.Event()
        self._lock = SimpleFTCollector.threading.Lock()
        self.buf = []
        self.ts_buf = []
        self.f_bias = None
        self.t_bias = None
        self.bias_ok = False
        self.buf_len = buf_len
        self._bias_samples_f = []
        self._bias_samples_t = []
        self._bias_n = 100

    def _loop(self):
        period = 1.0 / self.rate_hz
        nxt = time.perf_counter()
        while not self._stop.is_set():
            try:
                self.gc.update_imu(self.imu_pipe)
                try:
                    _, f_raw, t_raw = self.ft_reader.get_frame(timeout=0.001)
                except Exception:
                    f_raw = t_raw = None
                if f_raw is not None and t_raw is not None:
                    f_filt, t_filt = self.gc.process_ft_data(f_raw, t_raw)
                    comp_f, comp_t = self.gc.compensate_gravity(f_filt, t_filt, gravity_compensation_on=True)
                    if not self.bias_ok:
                        self._bias_samples_f.append(comp_f.copy())
                        self._bias_samples_t.append(comp_t.copy())
                        if len(self._bias_samples_f) >= self._bias_n:
                            self.f_bias = np.mean(self._bias_samples_f, axis=0)
                            self.t_bias = np.mean(self._bias_samples_t, axis=0)
                            self.bias_ok = True
                            print(f"FT bias set ({self._bias_n} samples): F={self.f_bias}")
                        continue
                    f_final = comp_f - self.f_bias
                    t_final = comp_t - self.t_bias
                    ft_vec = np.concatenate([f_final, t_final]).astype(np.float32)
                    ts = time.time()
                    with self._lock:
                        self.buf.append(ft_vec)
                        self.ts_buf.append(ts)
                        if len(self.buf) > self.buf_len:
                            self.buf = self.buf[-self.buf_len:]
                            self.ts_buf = self.ts_buf[-self.buf_len:]
            except Exception:
                pass
            nxt += period
            time.sleep(max(0, nxt - time.perf_counter()))

    def start(self):
        if self._th is None or not self._th.is_alive():
            self._stop.clear()
            import threading
            self._th = threading.Thread(target=self._loop, daemon=True)
            self._th.start()

    def stop(self):
        self._stop.set()
        if self._th:
            self._th.join(timeout=2)

    def window(self, length):
        with self._lock:
            ft = list(self.buf)[-length:]
            ts = list(self.ts_buf)[-length:]
        if len(ft) < length:
            pad = length - len(ft)
            ft = [np.zeros(6, dtype=np.float32)] * pad + ft
            ts = [0.0] * pad + ts
        return np.array(ft, dtype=np.float32), np.array(ts, dtype=np.float64)

    def reset_bias(self):
        self.bias_ok = False
        self._bias_samples_f = []
        self._bias_samples_t = []


# ─────────────────────────────────────────────────────────────
# Test routines
# ─────────────────────────────────────────────────────────────
def test_quaternion_convention(franka_api):
    """로봇 pose의 쿼터니온 컨벤션이 wxyz인지 검증."""
    print("\n" + "=" * 70)
    print("[TEST 1] 쿼터니온 컨벤션 검증")
    print("=" * 70)

    pose = franka_api.get_pose_sync()
    if pose is None:
        print("  [FAIL] 로봇 pose 읽기 실패")
        return False

    print(f"  Raw pose (7D): {pose}")
    print(f"  {pose_to_str(pose)}")

    q = pose[3:]  # [qw, qx, qy, qz] expected
    q_norm = np.linalg.norm(q)
    print(f"  |q| = {q_norm:.6f} (should be ~1.0)")

    # wxyz 가정: q[0]이 w이고, identity에 가까우면 q[0]≈1
    # xyzw 가정: q[3]이 w이고, identity에 가까우면 q[3]≈1
    # 실제 로봇은 identity가 아닐 수 있으므로, scipy로 변환해서 euler을 보자

    # 방법 1: wxyz로 해석
    q_xyzw = quat_wxyz_to_xyzw(q)  # scipy는 xyzw
    try:
        rot_wxyz = Rotation.from_quat(q_xyzw)
        euler_wxyz = rot_wxyz.as_euler('xyz', degrees=True)
    except Exception:
        euler_wxyz = np.array([999, 999, 999])

    # 방법 2: xyzw로 해석 (이미 xyzw)
    try:
        rot_xyzw = Rotation.from_quat(q)
        euler_xyzw = rot_xyzw.as_euler('xyz', degrees=True)
    except Exception:
        euler_xyzw = np.array([999, 999, 999])

    print(f"\n  wxyz로 해석시 euler(xyz): [{euler_wxyz[0]:+.1f}, {euler_wxyz[1]:+.1f}, {euler_wxyz[2]:+.1f}] deg")
    print(f"  xyzw로 해석시 euler(xyz): [{euler_xyzw[0]:+.1f}, {euler_xyzw[1]:+.1f}, {euler_xyzw[2]:+.1f}] deg")

    # 일반적으로 Franka는 EE가 아래를 향하므로 pitch가 ~180 또는 ~-180에 가깝다
    # wxyz가 맞으면 euler 값이 합리적일 것
    # 단, 어떤 값이 "합리적"인지는 config의 initial_pose와 비교
    print(f"\n  [INFO] Franka EE는 보통 거의 수직 (pitch~180 또는 roll~180)")
    print(f"  [INFO] config initial_pose quat: 결과와 비교하여 wxyz/xyzw 판별")

    # 최종 판정: q[0] (첫 원소)의 크기로 유추
    if abs(q[0]) > abs(q[3]):
        print(f"  [결과] q[0]={q[0]:.4f} > q[3]={q[3]:.4f} → wxyz 컨벤션으로 추정 (w가 첫 번째)")
    else:
        print(f"  [결과] q[3]={q[3]:.4f} > q[0]={q[0]:.4f} → xyzw 컨벤션으로 추정 (w가 마지막)")

    return True


def test_quaternion_rotation(franka_api, server_ip, delta_deg=3.0, settle_sec=1.5):
    """작은 회전을 적용하고 실제 로봇 pose 변화를 확인."""
    print("\n" + "=" * 70)
    print("[TEST 2] 쿼터니온 회전 적용 테스트")
    print("=" * 70)

    pose0 = np.array(franka_api.get_pose_sync(), dtype=np.float64)
    print(f"  시작 pose: {pose_to_str(pose0)}")

    # 현재 orientation (wxyz → scipy xyzw)
    q0_xyzw = quat_wxyz_to_xyzw(pose0[3:])
    rot0 = Rotation.from_quat(q0_xyzw)
    euler0 = rot0.as_euler('xyz', degrees=True)
    print(f"  시작 euler(xyz): [{euler0[0]:+.1f}, {euler0[1]:+.1f}, {euler0[2]:+.1f}] deg")

    results = []
    for axis_name, rotvec in [("roll(+x)", [1, 0, 0]),
                               ("pitch(+y)", [0, 1, 0]),
                               ("yaw(+z)", [0, 0, 1])]:
        rv = np.array(rotvec, dtype=np.float64) * np.deg2rad(delta_deg)
        d_rot = Rotation.from_rotvec(rv)
        new_rot = rot0 * d_rot
        new_q_xyzw = new_rot.as_quat()
        new_q_wxyz = quat_xyzw_to_wxyz(new_q_xyzw)
        expected_euler = new_rot.as_euler('xyz', degrees=True)

        target_pose = pose0.copy()
        target_pose[3:] = new_q_wxyz

        print(f"\n  [{axis_name}] +{delta_deg}° 회전 명령...")
        print(f"    target quat(wxyz): [{new_q_wxyz[0]:+.5f}, {new_q_wxyz[1]:+.5f}, {new_q_wxyz[2]:+.5f}, {new_q_wxyz[3]:+.5f}]")
        print(f"    expected euler: [{expected_euler[0]:+.1f}, {expected_euler[1]:+.1f}, {expected_euler[2]:+.1f}]")

        send_pose(server_ip, target_pose)
        time.sleep(settle_sec)

        actual_pose = np.array(franka_api.get_pose_sync(), dtype=np.float64)
        actual_q_xyzw = quat_wxyz_to_xyzw(actual_pose[3:])
        actual_rot = Rotation.from_quat(actual_q_xyzw)
        actual_euler = actual_rot.as_euler('xyz', degrees=True)
        euler_err = actual_euler - expected_euler

        print(f"    actual euler:   [{actual_euler[0]:+.1f}, {actual_euler[1]:+.1f}, {actual_euler[2]:+.1f}]")
        print(f"    error:          [{euler_err[0]:+.2f}, {euler_err[1]:+.2f}, {euler_err[2]:+.2f}] deg")

        results.append({
            'axis': axis_name, 'delta_deg': delta_deg,
            'expected_euler': expected_euler.tolist(),
            'actual_euler': actual_euler.tolist(),
            'error_deg': euler_err.tolist(),
        })

        # 원위치 복귀
        send_pose(server_ip, pose0)
        time.sleep(settle_sec)

    return results


def test_translation_ft(franka_api, server_ip, ft_collector, delta_m=0.005,
                         settle_sec=1.5, sample_n=100):
    """각 축 방향으로 이동하고 FT 응답을 기록. 어떤 FT 축이 반응하는지 확인."""
    print("\n" + "=" * 70)
    print("[TEST 3] 위치 이동 → FT 응답 테스트")
    print(f"  이동량: {delta_m*1000:.1f}mm, 샘플 수: {sample_n}")
    print("=" * 70)

    pose0 = np.array(franka_api.get_pose_sync(), dtype=np.float64)
    print(f"  기준 pose: {pose_to_str(pose0)}")

    directions = [
        ("+x", np.array([delta_m, 0, 0])),
        ("-x", np.array([-delta_m, 0, 0])),
        ("+y", np.array([0, delta_m, 0])),
        ("-y", np.array([0, -delta_m, 0])),
        ("+z", np.array([0, 0, delta_m])),
        ("-z", np.array([0, 0, -delta_m])),
    ]

    results = []
    for dir_name, d_pos in directions:
        # 원위치에서 매번 fresh baseline 측정 (드리프트 방지)
        print(f"\n  [{dir_name}] baseline 수집 중...")
        ft_baseline = collect_ft_samples(ft_collector, n_samples=sample_n)
        ft_base_mean = ft_baseline.mean(axis=0)
        print(f"    baseline: {np.array2string(ft_base_mean, precision=3)}")

        target_pose = pose0.copy()
        target_pose[:3] += d_pos

        print(f"    이동 → {pose_to_str(target_pose)}")
        send_pose(server_ip, target_pose)
        time.sleep(settle_sec)

        # 이동 후 FT 수집
        ft_samples = collect_ft_samples(ft_collector, n_samples=sample_n)
        ft_mean = ft_samples.mean(axis=0)
        ft_delta = ft_mean - ft_base_mean

        # 어떤 축이 가장 큰 변화를 보이는지
        axis_labels = ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']
        max_idx = np.argmax(np.abs(ft_delta))
        max_axis = axis_labels[max_idx]
        max_val = ft_delta[max_idx]

        print(f"    FT mean:   {np.array2string(ft_mean, precision=3)}")
        print(f"    FT delta:  {np.array2string(ft_delta, precision=3)}")
        print(f"    최대 변화: {max_axis} = {max_val:+.3f}")

        # 기대: +x 이동 → fx에 변화, +z 이동 → fz에 변화, etc.
        # 하지만 좌표계가 다를 수 있으므로 매핑을 기록
        robot_axis = dir_name[1]  # 'x', 'y', 'z'
        ft_response_axis = max_axis

        if robot_axis in ft_response_axis:
            verdict = "OK (같은 축)"
        else:
            verdict = f"MISMATCH (robot {robot_axis} → FT {ft_response_axis})"

        print(f"    판정: {verdict}")

        results.append({
            'direction': dir_name,
            'delta_pos': d_pos.tolist(),
            'ft_baseline': ft_base_mean.tolist(),
            'ft_after': ft_mean.tolist(),
            'ft_delta': ft_delta.tolist(),
            'max_response_axis': max_axis,
            'max_response_val': float(max_val),
            'verdict': verdict,
        })

        # 원위치 복귀
        send_pose(server_ip, pose0)
        time.sleep(settle_sec)

    return results


def test_coord_transform(config):
    """R_mat_pos / R_mat_rot 변환이 의미하는 바를 출력."""
    print("\n" + "=" * 70)
    print("[TEST 4] 좌표 변환 행렬 분석")
    print("=" * 70)

    R_pos = np.array(config['coordinate_transform']['R_mat_pos'])
    R_rot = np.array(config['coordinate_transform']['R_mat_rot'])

    print(f"  R_mat_pos:\n{R_pos}")
    print(f"  R_mat_rot:\n{R_rot}")
    print(f"  det(R_pos) = {np.linalg.det(R_pos):+.1f}  (reflection이면 -1)")
    print(f"  det(R_rot) = {np.linalg.det(R_rot):+.1f}  (정규 회전이면 +1)")

    # 각 모델 축이 로봇 어디에 매핑되는지
    model_axes = {'model_x': [1, 0, 0], 'model_y': [0, 1, 0], 'model_z': [0, 0, 1]}
    print("\n  [위치 변환] model → robot:")
    for name, v in model_axes.items():
        mapped = R_pos @ np.array(v)
        robot_label = ""
        for i, ax in enumerate(['x', 'y', 'z']):
            if abs(mapped[i]) > 0.5:
                sign = '+' if mapped[i] > 0 else '-'
                robot_label += f"{sign}robot_{ax} "
        print(f"    {name} → [{mapped[0]:+.0f}, {mapped[1]:+.0f}, {mapped[2]:+.0f}] = {robot_label.strip()}")

    print("\n  [회전 변환] model → robot:")
    for name, v in model_axes.items():
        mapped = R_rot @ np.array(v)
        robot_label = ""
        for i, ax in enumerate(['x', 'y', 'z']):
            if abs(mapped[i]) > 0.5:
                sign = '+' if mapped[i] > 0 else '-'
                robot_label += f"{sign}robot_{ax} "
        print(f"    {name} → [{mapped[0]:+.0f}, {mapped[1]:+.0f}, {mapped[2]:+.0f}] = {robot_label.strip()}")

    # R_mat_rot @ R_mat_rot.T should show the "sandwich" effect
    print(f"\n  R_rot @ I @ R_rot.T (회전 변환 합성):")
    combined = R_rot @ R_rot.T
    print(f"    {combined}")

    return {'R_mat_pos': R_pos.tolist(), 'R_mat_rot': R_rot.tolist(),
            'det_pos': float(np.linalg.det(R_pos)),
            'det_rot': float(np.linalg.det(R_rot))}


def test_gravity_residual(ft_collector, duration_sec=5.0):
    """중력보상 후 정지 상태의 FT 잔차를 확인."""
    print("\n" + "=" * 70)
    print("[TEST 5] 중력보상 잔차 확인")
    print(f"  {duration_sec}초 동안 정지 상태 FT 수집")
    print("=" * 70)

    n_samples = int(duration_sec * 200)
    ft_samples = collect_ft_samples(ft_collector, n_samples=n_samples)
    ft_mean = ft_samples.mean(axis=0)
    ft_std = ft_samples.std(axis=0)
    ft_max = np.abs(ft_samples).max(axis=0)

    print(f"  mean: {np.array2string(ft_mean, precision=4)}")
    print(f"  std:  {np.array2string(ft_std, precision=4)}")
    print(f"  max:  {np.array2string(ft_max, precision=4)}")

    force_residual = np.linalg.norm(ft_mean[:3])
    torque_residual = np.linalg.norm(ft_mean[3:])
    print(f"  |F| residual: {force_residual:.4f} N")
    print(f"  |T| residual: {torque_residual:.4f} Nm")

    if force_residual < 1.0:
        print(f"  [OK] Force residual < 1N")
    else:
        print(f"  [WARNING] Force residual >= 1N — 중력보상 또는 bias 확인 필요")

    return {
        'mean': ft_mean.tolist(), 'std': ft_std.tolist(),
        'max_abs': ft_max.tolist(),
        'force_residual_N': float(force_residual),
        'torque_residual_Nm': float(torque_residual),
    }


def print_axis_mapping_summary(translation_results):
    """FT 축 매핑 요약 테이블 출력."""
    print("\n" + "=" * 70)
    print("[요약] 로봇 이동 → FT 응답 축 매핑")
    print("=" * 70)
    print(f"  {'Robot Move':<10} {'FT Response':<15} {'Value':>10} {'Verdict'}")
    print(f"  {'-'*10:<10} {'-'*15:<15} {'-'*10:>10} {'-'*20}")
    for r in translation_results:
        print(f"  {r['direction']:<10} {r['max_response_axis']:<15} "
              f"{r['max_response_val']:>+10.3f} {r['verdict']}")

    # 매핑 테이블 구성
    print(f"\n  축 매핑 추론:")
    pos_map = {}
    neg_map = {}
    for r in translation_results:
        d = r['direction']
        axis = d[1]  # x, y, z
        sign = d[0]  # +, -
        ft_axis = r['max_response_axis']
        ft_val = r['max_response_val']
        if sign == '+':
            pos_map[axis] = (ft_axis, ft_val)
        else:
            neg_map[axis] = (ft_axis, ft_val)

    for axis in ['x', 'y', 'z']:
        if axis in pos_map and axis in neg_map:
            p_ax, p_val = pos_map[axis]
            n_ax, n_val = neg_map[axis]
            if p_ax == n_ax and np.sign(p_val) != np.sign(n_val):
                print(f"    robot_{axis} ↔ FT_{p_ax} (반대 부호: +{p_val:+.3f} / -{n_val:+.3f}) ✓ 일관됨")
            elif p_ax == n_ax:
                print(f"    robot_{axis} ↔ FT_{p_ax} (같은 부호: {p_val:+.3f} / {n_val:+.3f}) ⚠ 부호 확인")
            else:
                print(f"    robot_{axis}: +방향→FT_{p_ax}, -방향→FT_{n_ax} ⚠ 축 불일치")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="좌표계 & FT 정합성 진단")
    parser.add_argument('-c', '--config', default='eval_config/gear_insertion_2.yaml',
                        help="설정 파일 경로")
    parser.add_argument('--delta', type=float, default=0.05,
                        help="이동 테스트 거리 (m, default=5mm)")
    parser.add_argument('--rot-delta', type=float, default=30.0,
                        help="회전 테스트 각도 (deg, default=3)")
    parser.add_argument('--settle', type=float, default=1.5,
                        help="이동 후 안정 대기 시간 (s)")
    parser.add_argument('--samples', type=int, default=100,
                        help="FT 수집 샘플 수")
    parser.add_argument('--skip-move', action='store_true',
                        help="로봇 이동 없이 FT/좌표계 분석만")
    parser.add_argument('--output', default=None,
                        help="결과 JSON 저장 경로 (default: auto)")
    args = parser.parse_args()

    config = load_config(args.config)
    server_ip = config['robot']['server_ip']

    print("=" * 70)
    print("  좌표계 & FT 정합성 진단 스크립트")
    print(f"  Config: {args.config}")
    print(f"  Server: {server_ip}")
    print("=" * 70)

    # ── Hardware init ──
    franka_api = FrankaAPI(server_ip)

    ft_ip = config.get('ft_sensor', {}).get('ip', '172.27.190.4')
    ft_port = int(config.get('ft_sensor', {}).get('port', 8890))
    ft_reader = AidinFTSensorUDP(ft_ip, ft_port)
    ft_reader.start()

    gc = GravityCompensator(
        mass_for_x=config['gravity_compensator']['mass_for_x'],
        mass_for_y=config['gravity_compensator']['mass_for_y'],
        mass_for_z=config['gravity_compensator']['mass_for_z'],
        com_ft=np.array(config['gravity_compensator']['com_ft']),
        g_const=config['gravity_compensator']['g_const'],
    )

    print("\nIMU 초기화 중...")
    imu_pipe = rs.pipeline()
    imu_cfg = rs.config()
    imu_cfg.enable_stream(rs.stream.accel)
    imu_cfg.enable_stream(rs.stream.gyro)
    try:
        imu_pipe.start(imu_cfg)
    except Exception as e:
        print(f"IMU 초기화 실패: {e}")
        imu_pipe = None

    print("IMU warmup & 중력보상 캘리브레이션 중...")
    warmup_sec = float(config['gravity_compensator'].get('warmup_sec', 5.0))
    gc.calibrate_baseline(imu_pipe, ft_reader, warmup_sec=warmup_sec)
    print("캘리브레이션 완료")

    ft_collector = SimpleFTCollector(ft_reader, imu_pipe, gc, rate_hz=200)
    ft_collector.start()
    print("FT 수집 시작. 3초 대기 (bias 안정화)...")
    time.sleep(3.0)
    print("준비 완료\n")

    # ── Run tests ──
    all_results = {'timestamp': datetime.now().isoformat(), 'config': args.config}

    # Test 1: 쿼터니온 컨벤션
    test_quaternion_convention(franka_api)

    # Test 4: 좌표 변환 분석 (이동 불필요)
    all_results['coord_transform'] = test_coord_transform(config)

    # Test 5: 중력보상 잔차
    all_results['gravity_residual'] = test_gravity_residual(ft_collector)

    if not args.skip_move:
        # 안전 확인
        print("\n" + "!" * 70)
        print(f"  로봇 이동 테스트를 시작합니다.")
        print(f"  이동량: ±{args.delta*1000:.1f}mm (위치), ±{args.rot_delta:.1f}° (회전)")
        print(f"  현재 위치에서 각 축으로 소량 이동 후 복귀합니다.")
        input("  계속하려면 Enter를 누르세요 (Ctrl+C로 중단)...")
        print("!" * 70)

        # Test 2: 쿼터니온 회전
        all_results['rotation_test'] = test_quaternion_rotation(
            franka_api, server_ip, delta_deg=args.rot_delta, settle_sec=args.settle)

        # Test 3: 위치 이동 → FT 응답
        all_results['translation_ft'] = test_translation_ft(
            franka_api, server_ip, ft_collector,
            delta_m=args.delta, settle_sec=args.settle, sample_n=args.samples)

        # 요약
        print_axis_mapping_summary(all_results['translation_ft'])
    else:
        print("\n[SKIP] --skip-move 모드: 로봇 이동 테스트 건너뜀")

    # ── Save results ──
    if args.output is None:
        ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = 'eval_results/coord_ft_test'
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'coord_ft_test_{ts_str}.json')
    else:
        out_path = args.output
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[결과 저장] {out_path}")

    # ── Cleanup ──
    ft_collector.stop()
    ft_reader.stop()
    try:
        imu_pipe.stop()
    except Exception:
        pass
    print("\n완료.")


if __name__ == '__main__':
    main()
