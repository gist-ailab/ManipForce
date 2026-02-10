#!/usr/bin/env python3
from typing import Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, lfilter
from collections import deque
from ahrs.filters import Madgwick
import queue
import time
import pyrealsense2 as rs
from tqdm import trange

def butter_lowpass(cutoff, fs, order=2):
    """Butterworth 저역 통과 필터 계수 계산"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='low', analog=False)

def apply_filter(data_buf, b, a):
    """필터 적용하여 최신 결과만 반환"""
    return lfilter(b, a, data_buf)[-1]

def read_latest_imu(pipe):
    """
    RealSense pipeline에서 남아 있는 프레임을 다 버리고
    accel + gyro Motion Frame 한 쌍을 numpy 벡터로 돌려준다.
    새 프레임이 없으면 None 반환
    """
    frames = pipe.poll_for_frames()
    if not frames:
        return None          # → 다음 루프에서 다시 시도

    # 버퍼에 더 남아 있으면 끝까지 빼낸다
    while True:
        more = pipe.poll_for_frames()
        if not more:
            break
        frames = more        # 가장 마지막 세트만 남김

    accel = frames.first_or_default(rs.stream.accel)
    gyro  = frames.first_or_default(rs.stream.gyro)
    if accel is None or gyro is None:
        return None          # 드물게 한쪽이 없으면 skip

    a = accel.as_motion_frame().get_motion_data()
    g = gyro .as_motion_frame().get_motion_data()
    acc_vec  = np.array([a.x, a.y, a.z], dtype=float)
    gyro_vec = np.array([g.x, g.y, g.z], dtype=float)
    return acc_vec, gyro_vec

def convert_ft_imu_to_rs_units(acc_ft, gyro_ft, gyro_bias=None,
                               g_const: float = 9.81,
                               assume_deg_per_sec: bool = True):
    """
    Aidin FT IMU에서 읽은 acc, gyro를 RealSense IMU와 비슷한 단위로 변환.
    
    - acc_ft: FT IMU 가속도 (보통 1g ≈ 1.0 형태) -> m/s^2 로 스케일링
    - gyro_ft: FT IMU 자이로(raw, 큰 오프셋 포함)를 bias 제거 후 rad/s로 변환 (dps 가정)
    - gyro_bias: 정지 상태에서 측정한 gyro 평균값(오프셋). None이면 0으로 가정.
    """
    acc_ft = np.array(acc_ft, dtype=float)
    gyro_ft = np.array(gyro_ft, dtype=float)

    # 가속도: 1g ≈ 1.0 이라고 가정하고 m/s^2 로 스케일
    acc_rs = acc_ft * g_const

    # 자이로: 정지 상태 bias 제거
    if gyro_bias is None:
        gyro_bias = np.zeros(3, dtype=float)
    gyro_d = gyro_ft - gyro_bias  # bias 제거된 값 (센서 고유 단위, dps에 비례한다고 가정)

    # rad/s 로 변환 (Madgwick, RS와 동일 단위)
    if assume_deg_per_sec:
        gyro_rs = np.deg2rad(gyro_d)
    else:
        gyro_rs = gyro_d

    return acc_rs, gyro_rs

def convert_ft_imu_to_rs_units_with_scale(acc_ft, gyro_ft, gyro_bias=None,
                                          gyro_scale: np.ndarray = np.array([10, 10, 10]),
                                          g_const: float = 9.81) -> Tuple[np.ndarray, np.ndarray]:
    """
    FT IMU를 RealSense IMU와 유사한 단위/반응성으로 맞추기 위한 helper.

    - acc_ft: FT IMU 가속도 (1g ≈ 1.0 가정)  -> m/s^2
    - gyro_ft: FT IMU 자이로(raw)           -> (raw-bias) 를 dps로 가정해 rad/s 변환 후, 추가로 gyro_scale 배 스케일
    """
    acc_rs, gyro_rs = convert_ft_imu_to_rs_units(
        acc_ft, gyro_ft, gyro_bias=gyro_bias, g_const=g_const, assume_deg_per_sec=True
    )
    # gyro_rs는 현재 (raw-bias)를 1 dps로 가정해 rad/s로 변환한 값이므로,
    # 추가적인 scale을 곱해 RealSense IMU와의 동작 감도를 맞춘다.
    # x,y,z 축마다 다른 scale을 적용할 수 있도록 수정
    gyro_rs_scaled = gyro_rs * gyro_scale
    
    # gyro에 lowpass filter 적용
    return acc_rs, gyro_rs_scaled

def normalize(v):
    return v / np.linalg.norm(v)

def rotation_from_vectors(a, b):
    """
    a, b: 두 단위벡터 (3,)
    return: 3×3 회전 행렬 R so that R @ a ≈ b
    """
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[    0, -v[2],  v[1]],
                   [ v[2],     0, -v[0]],
                   [-v[1],  v[0],     0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))

class GravityCompensator:
    """중력 보상 클래스 - 1-7-1.calib_gravity_compensator_aidin.py 기반"""
    
    def __init__(self, mass_for_x=0.58, mass_for_y=0.53, mass_for_z=0.7, 
                 com_ft=np.array([0.01, 0.02, 0.02]), g_const=9.81,
                 madgwick_frequency=100.0, filter_fs=100.0):
        """
        중력 보상기 초기화
        
        Args:
            mass_for_x: X축 질량 (kg)
            mass_for_y: Y축 질량 (kg) 
            mass_for_z: Z축 질량 (kg)
            com_ft: FT 센서 기준 중심 of mass (m)
            g_const: 중력 가속도 (m/s²)
            madgwick_frequency: Madgwick 필터 주파수 (Hz)
            filter_fs: Butterworth 필터 샘플링 주파수 (Hz)
        """
        self.mass_for_x = mass_for_x
        self.mass_for_y = mass_for_y
        self.mass_for_z = mass_for_z
        self.com_ft = com_ft
        self.g_const = g_const
        
        # Madgwick 필터 초기화
        self.madgwick = Madgwick(gain=1, frequency=madgwick_frequency)
        self.q_prev = np.array([1.0, 0.0, 0.0, 0.0])  # 초기 쿼터니언 [w, x, y, z]
        
        # 필터 설정 (노이즈 감소를 위해 강한 필터링)
        cutoff = 10.0   # Hz (낮을수록 강한 필터링, 노이즈 제거 효과 큼)
        fs = filter_fs   # 샘플링 주파수
        order = 4  # 필터 차수 증가 (2차 -> 4차로 더 강한 필터링)
        self.b, self.a = butter_lowpass(cutoff, fs, order)
        
        # 버퍼 초기화 (버퍼 크기 증가로 더 부드러운 필터링)
        buffer_size = 40  # 20 -> 40으로 증가 (더 많은 샘플 평균)
        self.force_bufs = [deque(maxlen=buffer_size) for _ in range(3)]
        self.torque_bufs = [deque(maxlen=buffer_size) for _ in range(3)]
        
        # bias 초기화
        self.f_bias = np.zeros(3)
        self.t_bias = np.zeros(3)
        
        # 초기 IMU 상태
        self.init_rot_imu = None
        self.g_world = np.array([0, 0, -1])  # 이상적인 중력 방향
        
        # 중력 보상 토글 (True: ON, False: OFF)
        self.gravity_compensation_on = True
        
        # 주파수 모니터링 변수
        self.freq_check_interval = 5.0  # 주파수 체크 간격(초)
        self.last_freq_check_time = time.time()
        self.imu_count = 0
        self.ft_count = 0
        self.loop_count = 0
        
        # FT IMU 관련 파라미터 (선택적 사용)
        self.ft_gyro_bias = np.zeros(3)
        self.ft_gyro_scale = np.array([10, 10, 10])

        # IMU 소스 타입: 'rs' (기본, RealSense) 또는 'ft' (FT 내장 IMU)
        # RealSense 경로는 기존 동작을 그대로 유지하고,
        # FT IMU 경로에서만 이 플래그를 True로 설정해 좌표계/중력 보상 로직을 분기한다.
        self.use_ft_imu = False
    
    def calibrate_baseline(self, imu_pipe, ft_reader, warmup_sec=5.0):
        """
        베이스라인 데이터 수집 및 bias 계산
        
        Args:
            imu_pipe: RealSense IMU 파이프라인
            ft_reader: FT 센서 리더
            warmup_sec: 워밍업 시간 (초)
        """
        print("베이스라인 데이터 수집 시작...")
        f_list = []
        t_list = []
        q_list = []
        euler_list = []
        gravity_f_base_list = []
        gravity_t_base_list = []
        
        # 워밍업: 고정된 상태로 여러 번 업데이트
        iters = int(warmup_sec * 50)
        for _ in trange(iters, desc="IMU Warmup"):
            frames = imu_pipe.wait_for_frames()
            acc0 = frames.first_or_default(rs.stream.accel).as_motion_frame().get_motion_data()
            gyr0 = frames.first_or_default(rs.stream.gyro).as_motion_frame().get_motion_data()
            acc0 = np.array([acc0.x, acc0.y, acc0.z], float)
            gyr0 = np.array([gyr0.x, gyr0.y, gyr0.z], float)

            self.q_prev = self.madgwick.updateIMU(self.q_prev, gyr=gyr0, acc=acc0)
            q_list.append(np.copy(self.q_prev))  # 쿼터니언 저장

            # change to euler angle
            r = R.from_quat([self.q_prev[1], self.q_prev[2], self.q_prev[3], self.q_prev[0]])
            euler_angles = r.as_euler('xyz', degrees=True)
            
            # IMU 업데이트 → q0_prev
            rot_imu0 = R.from_quat([self.q_prev[1], self.q_prev[2], self.q_prev[3], self.q_prev[0]])
            g_sensor_base = rot_imu0.inv().apply([0,0,-1])  # 단위벡터
            
            # IMU를 FT 센서 좌표계로 변환
            angle_x_rad = np.radians(-20)  # x축으로 -20도 회전
            rot_imu2ft = R.from_euler('x', angle_x_rad)
            g_ft_base = rot_imu2ft.apply(g_sensor_base)
            g_ft_base = np.array([-g_ft_base[0], -g_ft_base[1], g_ft_base[2]])
            
            # X, Y축에 대해 다른 질량 적용
            gravity_f_base = np.array([
                self.mass_for_x * self.g_const * g_ft_base[0],  # X축
                self.mass_for_y * self.g_const * g_ft_base[1],  # Y축
                self.mass_for_x * self.g_const * g_ft_base[2]   # Z축 (X축과 동일하게)
            ])
            gravity_t_base = np.cross(self.com_ft, gravity_f_base)  # 툴 중력(torque)
            gravity_f_base_list.append(gravity_f_base)
            gravity_t_base_list.append(gravity_t_base)
            
            ts, f_raw, t_raw = ft_reader.get_frame(timeout=1.0)
            
            # FT 축 변환 적용
            ft_data = np.concatenate([f_raw, t_raw])
            f_raw_converted = ft_data[:3]
            t_raw_converted = ft_data[3:]
            
            for i in range(3):
                self.force_bufs[i].append(f_raw_converted[i])
                self.torque_bufs[i].append(t_raw_converted[i])
            forces_filt = [apply_filter(self.force_bufs[i], self.b, self.a) for i in range(3)]
            torques_filt = [apply_filter(self.torque_bufs[i], self.b, self.a) for i in range(3)]

            f_list.append(forces_filt)
            t_list.append(torques_filt)
            # euler_list.append(euler_angles)

        # bias 계산
        f_mean = np.mean(f_list, axis=0)
        t_mean = np.mean(t_list, axis=0)
        gf_mean = np.mean(gravity_f_base_list, axis=0)
        gt_mean = np.mean(gravity_t_base_list, axis=0)
        self.f_bias = f_mean - gf_mean
        self.t_bias = t_mean - gt_mean
        
        # 초기 IMU 상태 저장
        rot_imu_final = R.from_quat([self.q_prev[1], self.q_prev[2], self.q_prev[3], self.q_prev[0]])
        self.init_rot_imu = rot_imu_final
        
        # 초기 imu가 중력 기준으로 얼마나 틀어져있는지 파악
        # g_imu_measured = rot_imu_final.inv().apply(self.g_world)
        # tilt_angle = np.arccos(np.clip(np.dot(g_imu_measured, self.g_world), -1.0, 1.0)) * 180 / np.pi
        
        # # 틀어진 축 계산 (회전축)
        # if tilt_angle > 1e-6:  # 각도가 0이 아닌 경우에만
        #     tilt_axis = np.cross(self.g_world, g_imu_measured)
        #     tilt_axis = tilt_axis / np.linalg.norm(tilt_axis)
        # else:
        #     tilt_axis = np.array([0, 0, 0])
            
        # 오일러 각으로 표현 (롤, 피치, 요)
        # euler_angles = rot_imu_final.as_euler('xyz', degrees=True)
        
        # 이 정보를 실시간 루프에서 사용    
        # self.init_imu_tilt_angle = tilt_angle
        
    
    def update_imu(self, imu_pipe):
        """IMU 데이터 업데이트"""
        if imu_pipe is None:
            return None, None
            
        imu_data = None
        while imu_data is None:
            imu_data = read_latest_imu(imu_pipe)
        acc_vec, gyro_vec = imu_data
        self.q_prev = self.madgwick.updateIMU(self.q_prev, gyr=gyro_vec, acc=acc_vec)
        self.imu_count += 1
        return acc_vec, gyro_vec
    
    def calibrate_ft_imu_bias(self, ft_reader, warmup_sec: float = 3.0, gyro_scale: np.ndarray = np.array([10, 10, 10])):
        """
        베이스라인 데이터 수집 및 bias 계산
        
        Args:
            ft_reader: FT 센서 리더
            warmup_sec: 워밍업 시간 (초)
        """
        print("베이스라인 데이터 수집 시작...(FT IMU 기반)")
        # FT IMU 전용 설정
        self.use_ft_imu = True
        self.ft_gyro_scale = gyro_scale
        f_list = []
        t_list = []
        q_list = []
        gravity_f_base_list = []
        gravity_t_base_list = []
        
        # 워밍업: 고정된 상태로 여러 번 업데이트
        iters = int(warmup_sec * 50)

        for _ in trange(iters, desc="FT IMU Warmup"):
            # FT IMU에서 raw 데이터 읽기
            ts_ft, acc_ft, gyro_ft = ft_reader.get_imu_data(timeout=1.0)

            # 현재 추정된 bias(초기에는 0)를 사용해 Madgwick 업데이트
            acc_vec, gyro_vec = convert_ft_imu_to_rs_units_with_scale(
                acc_ft, gyro_ft, gyro_bias=self.ft_gyro_bias,
                gyro_scale=self.ft_gyro_scale, g_const=self.g_const
            )

            acc_vec = np.array([-acc_vec[0], -acc_vec[1], -acc_vec[2]])
            gyro_vec = np.array([-gyro_vec[0], -gyro_vec[1], -gyro_vec[2]])
            
            self.q_prev = self.madgwick.updateIMU(self.q_prev, gyr=gyro_vec, acc=acc_vec)
            q_list.append(np.copy(self.q_prev))  # 쿼터니언 저장

            # IMU 업데이트 → q_prev
            rot_imu0 = R.from_quat([self.q_prev[1], self.q_prev[2], self.q_prev[3], self.q_prev[0]])
            g_ft_base = rot_imu0.inv().apply([0, 0, -1])  # 단위벡터
            g_ft_base = np.array([-g_ft_base[0], -g_ft_base[1], g_ft_base[2]])

            # X, Y축에 대해 다른 질량 적용
            gravity_f_base = np.array([
                self.mass_for_x * self.g_const * g_ft_base[0],  # X축
                self.mass_for_y * self.g_const * g_ft_base[1],  # Y축
                self.mass_for_x * self.g_const * g_ft_base[2]   # Z축 (X축과 동일하게)
            ])
            gravity_t_base = np.cross(self.com_ft, gravity_f_base)  # 툴 중력(torque)
            gravity_f_base_list.append(gravity_f_base)
            gravity_t_base_list.append(gravity_t_base)

            ts, f_raw, t_raw = ft_reader.get_frame(timeout=1.0)

            # FT 축 변환 적용 (force/torque는 이미 FT 기준이라고 가정)
            ft_data = np.concatenate([f_raw, t_raw])
            f_raw_converted = ft_data[:3]
            t_raw_converted = ft_data[3:]

            for i in range(3):
                self.force_bufs[i].append(f_raw_converted[i])
                self.torque_bufs[i].append(t_raw_converted[i])
            forces_filt = [apply_filter(self.force_bufs[i], self.b, self.a) for i in range(3)]
            torques_filt = [apply_filter(self.torque_bufs[i], self.b, self.a) for i in range(3)]

            f_list.append(forces_filt)
            t_list.append(torques_filt)

        # bias 계산 (FT force/torque vs 중력 예측)
        f_mean = np.mean(f_list, axis=0)
        t_mean = np.mean(t_list, axis=0)
        gf_mean = np.mean(gravity_f_base_list, axis=0)
        gt_mean = np.mean(gravity_t_base_list, axis=0)
        self.f_bias = f_mean - gf_mean
        self.t_bias = t_mean - gt_mean

        # FT IMU 기준 초기 자세 저장 (RealSense 경로와 유사하게)
        rot_imu_final = R.from_quat([self.q_prev[1], self.q_prev[2], self.q_prev[3], self.q_prev[0]])
        self.init_rot_imu = rot_imu_final

    def update_imu_from_ft(self, ft_reader):
        """
        FT IMU에서 acc, gyro를 읽어 Madgwick 필터를 업데이트한다.
        (convert_ft_imu_to_rs_units_with_scale 사용)
        update_imu와 동일하게 데이터를 받을 때까지 대기한다.
        """
        imu_data = None
        while imu_data is None:
            try:
                ts_ft, acc_ft, gyro_ft = ft_reader.get_imu_data(timeout=0.1)
                imu_data = (acc_ft, gyro_ft)
            except queue.Empty:
                continue  # 데이터가 없으면 다시 시도

        acc_ft, gyro_ft = imu_data

        # FT IMU를 사용하는 경우이므로 플래그 설정
        self.use_ft_imu = True

        acc_vec, gyro_vec = convert_ft_imu_to_rs_units_with_scale(
            acc_ft, gyro_ft, gyro_bias=self.ft_gyro_bias,
            gyro_scale=self.ft_gyro_scale, g_const=self.g_const
        )
        
        # calibrate_ft_imu_bias와 동일하게 부호 반전 적용
        acc_vec = np.array([-acc_vec[0], -acc_vec[1], -acc_vec[2]])
        gyro_vec = np.array([-gyro_vec[0], -gyro_vec[1], -gyro_vec[2]])
        
        self.q_prev = self.madgwick.updateIMU(self.q_prev, gyr=gyro_vec, acc=acc_vec)
        return acc_vec, gyro_vec
    
    def process_ft_data(self, f_raw, t_raw):
        """FT 데이터 처리 및 필터링"""
        # FT 축 변환 적용
        ft_data = np.concatenate([f_raw, t_raw])
        f_raw_converted = ft_data[:3]
        t_raw_converted = ft_data[3:]
        
        for i in range(3):
            self.force_bufs[i].append(f_raw_converted[i])
            self.torque_bufs[i].append(t_raw_converted[i])
        forces_filt = [apply_filter(self.force_bufs[i], self.b, self.a) for i in range(3)]
        torques_filt = [apply_filter(self.torque_bufs[i], self.b, self.a) for i in range(3)]
        
        return forces_filt, torques_filt
    
    def calculate_gravity_compensation(self):
        """중력 보상 계산"""
        # 현재 IMU 회전
        rot_imu_now = R.from_quat([self.q_prev[1], self.q_prev[2], self.q_prev[3], self.q_prev[0]])
        g_imu_now = rot_imu_now.inv().apply([0,0,-1])

        # IMU 타입에 따라 FT 좌표계로의 변환 방법이 다름
        if not self.use_ft_imu:
            # RealSense IMU인 경우: FT 센서 기준으로 x축 -20도 회전 + x,y 부호 반전
            angle_x_rad = np.radians(-20)  # x축으로 -20도 회전
            rot_imu2ft = R.from_euler('x', angle_x_rad)  # x축(roll) 회전만 적용
            g_ft_now = rot_imu2ft.apply(g_imu_now)
            g_ft_now = np.array([-g_ft_now[0], -g_ft_now[1], g_ft_now[2]])
        else:
            # FT IMU인 경우: IMU 좌표계가 곧 FT 좌표계라고 가정
            g_ft_now = np.array([-g_imu_now[0], -g_imu_now[1], g_imu_now[2]])
        
        # 중력보상 - X, Y축에 대해 다른 질량 적용
        gravity_force = np.array([
            self.mass_for_x * self.g_const * g_ft_now[0],  # X축
            self.mass_for_y * self.g_const * g_ft_now[1],  # Y축
            self.mass_for_x * self.g_const * g_ft_now[2]   # Z축 (X축과 동일하게)
        ])
        gravity_torque = np.cross(self.com_ft, gravity_force)
        
        return gravity_force, gravity_torque
    
    def compensate_gravity(self, forces_filt, torques_filt, gravity_compensation_on=None):
        """
        중력 보상 적용
        
        Args:
            forces_filt: 필터링된 힘 데이터
            torques_filt: 필터링된 토크 데이터
            gravity_compensation_on: 중력 보상 활성화 여부 (None이면 클래스 설정 사용)
            
        Returns:
            compensated_force: 중력 보상된 힘
            compensated_torque: 중력 보상된 토크
        """
        if gravity_compensation_on is None:
            gravity_compensation_on = self.gravity_compensation_on
            
        if gravity_compensation_on:
            gravity_force, gravity_torque = self.calculate_gravity_compensation()
            compensated_force = forces_filt - self.f_bias - gravity_force
            compensated_torque = torques_filt - self.t_bias - gravity_torque
        else:
            compensated_force = forces_filt - self.f_bias  # bias만 제거
            compensated_torque = torques_filt - self.t_bias  # bias만 제거
        
        return compensated_force, compensated_torque
    
    
    def process_single_frame(self, imu_pipe=None, ft_reader=None, use_ft_imu=None):
        """
        단일 프레임 처리 (IMU + FT + 중력 보상)
        
        Args:
            imu_pipe: RealSense IMU 파이프라인
            ft_reader: FT 센서 리더
            
        Returns:
            compensated_force: 중력 보상된 힘
            compensated_torque: 중력 보상된 토크
            debug_info: 디버그 정보 딕셔너리
        """
        if use_ft_imu is None:
            flag = self.use_ft_imu
        else:
            flag = use_ft_imu

        if not flag:
            self.update_imu(imu_pipe)
        else:
            self.update_imu_from_ft(ft_reader)
        
        # FT 데이터 읽기
        try:
            ts, f_raw, t_raw = ft_reader.read_latest(timeout=0.001)
            self.ft_count += 1
        except queue.Empty:
            return None, None, None
        
        # FT 데이터 처리 및 중력 보상
        forces_filt, torques_filt = self.process_ft_data(f_raw, t_raw)
        compensated_force, compensated_torque = self.compensate_gravity(forces_filt, torques_filt)
        
        # 디버그 정보 수집
        rot_imu_now = R.from_quat([self.q_prev[1], self.q_prev[2], self.q_prev[3], self.q_prev[0]])
        g_imu_now = rot_imu_now.inv().apply([0,0,-1])
        
        g_world_estimated = rot_imu_now.apply(g_imu_now)
        direction_error = np.arccos(np.clip(np.dot(g_world_estimated, self.g_world), -1.0, 1.0)) * 180 / np.pi
        
        relative_rot = rot_imu_now * self.init_rot_imu.inv()        
        relative_angles = relative_rot.as_euler('xyz', degrees=True)
        
        current_tilt_angle = np.arccos(np.clip(np.dot(g_imu_now, self.g_world), -1.0, 1.0)) * 180 / np.pi
        
        debug_info = {
            'direction_error': direction_error,
            'relative_angles': relative_angles,
            'current_tilt_angle': current_tilt_angle,
            'gravity_compensation_on': self.gravity_compensation_on
        }
        
        return compensated_force, compensated_torque, debug_info
    
    
    def set_gravity_compensation(self, on_off):
        """중력 보상 토글 설정"""
        self.gravity_compensation_on = on_off
    
    def get_frequency_stats(self):
        """주파수 통계 반환"""
        current_time = time.time()
        elapsed = current_time - self.last_freq_check_time
        
        if elapsed >= self.freq_check_interval:
            imu_freq = self.imu_count / elapsed
            ft_freq = self.ft_count / elapsed
            loop_freq = self.loop_count / elapsed
            
            # 카운터 재설정
            self.last_freq_check_time = current_time
            self.imu_count = 0
            self.ft_count = 0
            self.loop_count = 0
            
            return {
                'imu_freq': imu_freq,
                'ft_freq': ft_freq,
                'loop_freq': loop_freq,
                'elapsed': elapsed
            }
        return None
    
    def increment_loop_count(self):
        """루프 카운터 증가"""
        self.loop_count += 1 