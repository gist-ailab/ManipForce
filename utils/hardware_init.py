"""Hardware initialization and reset utilities for eval_robot."""

import os
import time
import numpy as np
import scipy.spatial.transform as st
import pyrealsense2 as rs

from utils.rs_capture import RSCapture
from utils.ft_capture import AidinFTSensorUDP
from utils.gravity_compensation_utils import GravityCompensator
from utils.ft_collector import FTCollector
from utils.flask_pose_client import send_pose_command
from utils.franka_api import FrankaAPI


def init_hardware(config):
    """Initialize cameras, IMU, FT sensor, gravity compensator.

    Returns:
        (franka_api, ft_reader, ft_collector, imu_source, gravity_compensator, camera, additional_cam)
    """
    server_ip = config['robot']['server_ip']
    franka_api = FrankaAPI(server_ip)

    ft_ip = config.get('ft_sensor', {}).get('ip', '172.27.190.4')
    ft_port = int(config.get('ft_sensor', {}).get('port', 8890))
    ft_reader = AidinFTSensorUDP(ft_ip, ft_port)
    ft_reader.start()

    gravity_compensator = GravityCompensator(
        mass_for_x=config['gravity_compensator']['mass_for_x'],
        mass_for_y=config['gravity_compensator']['mass_for_y'],
        mass_for_z=config['gravity_compensator']['mass_for_z'],
        com_ft=np.array(config['gravity_compensator']['com_ft']),
        g_const=config['gravity_compensator']['g_const']
    )

    print(f"카메라 초기화 중 (Serial: {config['camera']['main_camera']['serial_number']})...")
    camera = RSCapture(
        name=config['camera']['main_camera']['name'],
        serial_number=config['camera']['main_camera']['serial_number'],
        dim=tuple(config['camera']['main_camera']['resolution']),
        fps=config['camera']['main_camera']['fps'],
        depth=False,
        enable_imu=True
    )
    print(f"추가 카메라 초기화 중 (Serial: {config['camera']['additional_camera']['serial_number']})...")
    additional_cam = RSCapture(
        name=config['camera']['additional_camera']['name'],
        serial_number=config['camera']['additional_camera']['serial_number'],
        dim=tuple(config['camera']['additional_camera']['resolution']),
        fps=config['camera']['additional_camera']['fps'],
        depth=False
    )

    # 카메라 안정화 대기 후 IMU 시작 (motion sensor callback, 파이프라인과 분리)
    import time as _time
    _time.sleep(1.0)  # 양쪽 카메라 USB 안정화
    camera.start_imu()
    imu_source = camera
    print(f"IMU: motion sensor callback (serial: {camera.serial_number})")

    ft_collector = FTCollector(
        ft_reader=ft_reader,
        imu_source=imu_source,
        gravity_compensator=gravity_compensator,
        rate_hz=config['ft_sensor']['rate_hz'],
        buf_len=config['ft_sensor']['buffer_length']
    )

    return franka_api, ft_reader, ft_collector, imu_source, gravity_compensator, camera, additional_cam


def compute_initial_pose(config_path, config):
    """Compute randomized initial pose from config."""
    initial_pose = np.array(config['robot']['initial_pose'], dtype=np.float64)
    random_range = config['robot']['random_pos_range']
    noise = np.random.uniform(-random_range, random_range, size=3)

    cfg_name = os.path.basename(config_path).lower() if isinstance(config_path, str) else ''
    if 'battery_assemb' in cfg_name:
        initial_pose[0] += noise[0]
        initial_pose[2] += noise[2]
    else:
        initial_pose[:3] += noise

    initial_tilt_deg = float(config['robot'].get('initial_tilt_deg', 0.0))
    base_rot = st.Rotation.from_quat([1, 0, 0, 0])
    half_angle = np.deg2rad(initial_tilt_deg) / 2
    y_tilt = st.Rotation.from_quat([0, np.sin(half_angle), 0, np.cos(half_angle)])
    initial_pose[3:] = (base_rot * y_tilt).as_quat()
    return initial_pose


def reset_to_home(client_socket, franka_api, initial_pose, config, state,
                  hw_refs, ft_collector=None, gripper='keep'):
    """현재 위치에서 z방향으로 lift → gravity_comp_pose에서 캘리브레이션 → initial_pose 이동.

    Args:
        hw_refs: dict with keys 'gravity_comp', 'imu_source', 'ft_reader'.
    """
    lift_z = config.get('robot', {}).get('reset_lift_z', 0.0)
    if lift_z > 0:
        cur_pose = np.array(franka_api.get_pose_sync(), dtype=np.float64)
        lift_pose = cur_pose.copy()
        lift_pose[2] += lift_z
        print(f"[RESET] z +{lift_z*1000:.0f}mm 리프트 중...")
        send_pose_command(client_socket, lift_pose, gripper=gripper, reset=True)
        time.sleep(1.0)

    gc_pose_cfg = config['robot'].get('gravity_comp_pose')
    if gc_pose_cfg is not None:
        gc_pose = np.array(gc_pose_cfg, dtype=np.float64)
        print("[RESET] 중력 보상 pose로 이동 중...")
        send_pose_command(client_socket, gc_pose, gripper=gripper, reset=True)
        time.sleep(1.5)
        # 중력 보상 재캘리브레이션
        gravity_comp = hw_refs.get('gravity_comp')
        imu_source = hw_refs.get('imu_source')
        ft_reader = hw_refs.get('ft_reader')
        if gravity_comp is not None and imu_source is not None and ft_reader is not None:
            print("[RESET] 중력 보상 재캘리브레이션...")
            gravity_comp.calibrate_baseline(
                imu_source, ft_reader,
                warmup_sec=float(config['gravity_compensator']['warmup_sec']))
            print("[RESET] 중력 보상 재캘리브레이션 완료")
        # 실제 initial_pose로 이동
        print("[RESET] 초기 위치로 이동 중...")
        send_pose_command(client_socket, initial_pose, gripper=gripper, reset=True)
        time.sleep(1.5)
    else:
        print("[RESET] Home pose 이동 중...")
        send_pose_command(client_socket, initial_pose, gripper=gripper, reset=True)
        time.sleep(1.5)
        if ft_collector is not None:
            state.ft_latest = np.zeros(6, dtype=np.float32)

    # FT baseline 재측정 (contact 감지 + CA 공용)
    if ft_collector is not None:
        print("[RESET] FT baseline 재측정 중 (2s 안정화)...")
        time.sleep(2.0)
        ft_buf, _ = ft_collector.window(30)
        state.gripping_ft_bias = ft_buf.mean(axis=0) if len(ft_buf) > 0 else np.zeros(6, dtype=np.float32)
        print(f"[RESET] FT baseline: F=[{state.gripping_ft_bias[0]:+.2f},{state.gripping_ft_bias[1]:+.2f},{state.gripping_ft_bias[2]:+.2f}]N")

    print("[RESET] 완료")


def calibrate_startup(client_socket, initial_pose, config, state, hw_refs, ft_collector):
    """초기 캘리브레이션 시퀀스: 이동 → gravity calibration → FT bias → CA baseline."""
    gravity_compensator = hw_refs['gravity_comp']
    imu_source = hw_refs['imu_source']
    ft_reader = hw_refs['ft_reader']

    gravity_comp_pose = config['robot'].get('gravity_comp_pose')
    if gravity_comp_pose is not None:
        gc_pose = np.array(gravity_comp_pose, dtype=np.float64)
        print("중력 보상 pose로 이동 중...")
        try:
            send_pose_command(client_socket, gc_pose, reset=True)
            time.sleep(3.0)
            print("중력 보상 pose 이동 완료, 안정화 대기 5초...")
            time.sleep(5.0)
        except (OSError, BrokenPipeError) as e:
            print(f"중력 보상 pose 이동 실패: {e}")
    else:
        print("초기 위치로 이동 중...")
        try:
            send_pose_command(client_socket, initial_pose, reset=True)
            time.sleep(3.0)
            print("초기 위치 이동 완료, 안정화 대기 5초...")
            time.sleep(5.0)
        except (OSError, BrokenPipeError) as e:
            print(f"초기 위치 이동 실패: {e}")

    print("IMU warmup 및 중력 보상 캘리브레이션 시작...")
    gravity_compensator.calibrate_baseline(
        imu_source, ft_reader,
        warmup_sec=float(config['gravity_compensator']['warmup_sec']))
    print("IMU warmup 및 중력 보상 캘리브레이션 완료")

    print("FT 보정 시작, 안정화 대기 5초...")
    ft_collector.start()
    time.sleep(5.0)
    print("FT 보정 완료")

    # 중력 보상 + FT bias 완료 후 실제 initial_pose로 이동
    if gravity_comp_pose is not None:
        print("실제 초기 위치로 이동 중...")
        try:
            send_pose_command(client_socket, initial_pose, reset=True)
            time.sleep(2.0)
            print("초기 위치 이동 완료")
        except (OSError, BrokenPipeError) as e:
            print(f"초기 위치 이동 실패: {e}")

    # FT baseline 측정
    print("[FT] initial_pose에서 FT baseline 측정 중 (2s 안정화)...")
    time.sleep(2.0)
    ft_buf, _ = ft_collector.window(30)
    state.gripping_ft_bias = ft_buf.mean(axis=0) if len(ft_buf) > 0 else np.zeros(6, dtype=np.float32)
    print(f"[FT] baseline: F=[{state.gripping_ft_bias[0]:+.2f},{state.gripping_ft_bias[1]:+.2f},{state.gripping_ft_bias[2]:+.2f}]N  "
          f"T=[{state.gripping_ft_bias[3]:+.3f},{state.gripping_ft_bias[4]:+.3f},{state.gripping_ft_bias[5]:+.3f}]")
