#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import queue
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, lfilter
from tqdm import trange
import os
import cv2
from collections import deque
from utils.ft_capture import AidinFTSensorUDP
import argparse

import rospy
from geometry_msgs.msg import WrenchStamped
import pyrealsense2 as rs
from ahrs.filters import Madgwick

import torch

# 중력 보상 유틸리티 및 RealSense IMU 유틸 함수 import
from utils.gravity_compensation_utils import GravityCompensator

if __name__ == '__main__':
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='중력 보상 캘리브레이션')
    parser.add_argument('--gravity_compensate', action='store_true', 
                       help='중력 보상 활성화 (기본값: False, raw 값만 표시)')
    parser.add_argument('--robot', action='store_true', 
                       help='Robot 모드 (기본값: False, Gumi 모드)')
    parser.add_argument('--use_ft_imu', action='store_true',
                       help='RealSense 대신 FT 내장 IMU를 사용해 중력 보상 (실험용)')
    parser.add_argument('--global_top_setup', action='store_true',
                       help='GlobalTop 세팅 파라미터 사용 (Gumi 기본과 동일 질량/COM)')
    args = parser.parse_args()
    
    print("프로그램 시작")
    print(f"중력 보상: {'ON' if args.gravity_compensate else 'OFF (Raw 값만 표시)'}")
    
    rospy.init_node('gravity_compensation_calibration', anonymous=True)
    pub = rospy.Publisher('/ft300/wrench', WrenchStamped, queue_size=10)
    rate = rospy.Rate(200)  # 100Hz 목표
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ====== FT 센서 시작 ======
    print("FT 센서 초기화 중...")
    ft_reader = AidinFTSensorUDP('172.27.190.4', 8890)
    ft_reader.start()
    print("FT 센서 초기화 완료")

    # ====== IMU 설정 (RealSense, FT IMU 비교용/기존 경로) ======
    print("IMU 초기화 중 (RealSense)...")
    imu_pipe = rs.pipeline()
    imu_cfg = rs.config()
    imu_cfg.enable_stream(rs.stream.accel)
    imu_cfg.enable_stream(rs.stream.gyro)
    imu_pipe.start(imu_cfg)
    print("IMU 초기화 완료")
    
    # ====== 중력 보상기 초기화 ======
    gravity_compensator = None
    global_top_setup = args.global_top_setup

    if args.gravity_compensate:
        # 중력 보상기 초기화
        # Gumi Setup
        if args.robot:
            # Robot Setup
            gravity_compensator = GravityCompensator(
                mass_for_x=0.73,
                mass_for_y=0.73,
                mass_for_z=0.73,
                com_ft=np.array([0.0, 0.01, 0.03]),
                g_const=9.81,
                madgwick_frequency=200.0,
                filter_fs=200.0
            )
        elif global_top_setup:
            gravity_compensator = GravityCompensator(
                mass_for_x=0.0,
                mass_for_y=0.0,
                mass_for_z=0.0,
                com_ft=np.array([0.0, 0.0, 0.0]),
                g_const=9.81
            )
        else:
            # Gumi Setup
            gravity_compensator = GravityCompensator(
                mass_for_x=0.58,
                mass_for_y=0.53, 
                mass_for_z=0.7,
                com_ft=np.array([0.01, 0.01, 0.03]),
                g_const=9.81
            )
        
        if args.use_ft_imu:
            # FT IMU 기반 중력 보상용 초기화 (RealSense 없이 FT IMU만 사용)
            print("[FT-IMU] FT IMU 기반 중력 보상용 초기화 (gyro bias, 자세 warmup)...")
            gravity_compensator.calibrate_ft_imu_bias(
                ft_reader,
                warmup_sec=5.0,
                gyro_scale=np.array([0.1, 0.1, 0.1])
            )
            # 내부 플래그도 명시적으로 세팅 (안전장치)
            gravity_compensator.use_ft_imu = True
            print("[FT-IMU] 초기화 완료")
        else:
            # 기존 RealSense IMU 기반 베이스라인 수집
            print("베이스라인 데이터 수집 중 (RealSense IMU)...")
            gravity_compensator.calibrate_baseline(imu_pipe, ft_reader, warmup_sec=5.0)
            gravity_compensator.use_ft_imu = False
            print("베이스라인 데이터 수집 완료")
    
    # gravity_compensator.calibrate_ft_imu_bias(
    #     ft_reader,
    #     warmup_sec=5.0,
    #     gyro_scale=np.array([0.01, 0.01, 0.01])
    # )

    # gravity_compensator.calibrate_baseline(imu_pipe, ft_reader, warmup_sec=5.0)

    # 초기 bias 설정 (모든 모드에서 사용)
    f_bias_initial = None
    t_bias_initial = None
    settle_count = 0  # 초반 IMU 수렴 구간 프레임 카운트

    # 실시간 루프
    while not rospy.is_shutdown():
        if args.gravity_compensate:
            # 중력 보상 모드
            # process_single_frame 내부에서 RealSense / FT IMU를 선택하도록 use_ft_imu 플래그를 넘겨줌
            compensated_force, compensated_torque, debug_info = gravity_compensator.process_single_frame(
                imu_pipe=imu_pipe,
                ft_reader=ft_reader,
                use_ft_imu=gravity_compensator.use_ft_imu,
            )
            
            if compensated_force is not None:
                # 초기 bias 제거
                if f_bias_initial is None:
                    # FT IMU 모드일 때만 초반 N프레임을 버리면서 IMU 수렴을 기다림
                    if gravity_compensator.use_ft_imu:
                        settle_count += 1
                        if settle_count < 10000:  # 대략 1~2초 수준에서 한 번만 잡도록
                            # print(f"Settle count (FT IMU): {settle_count}")
                            continue
                        # 한 번 초기 bias를 잡은 뒤에는 다시 카운트할 필요 없음
                        settle_count = 0

                    f_bias_initial = compensated_force.copy()
                    t_bias_initial = compensated_torque.copy()
                
                # 초기 bias 제거
                f_final = compensated_force - f_bias_initial
                # fy 는 3N 더 빼기
                if args.robot:
                    f_final[1] -= 3.0
                else:
                    f_final[1] -= 0.0
                t_final = compensated_torque - t_bias_initial
                
                # # 결과 로깅 (디버깅용)
                # print("--------------------------------")
                # print(f"중력 보상: {'ON' if debug_info['gravity_compensation_on'] else 'OFF'}")
                # print(f"중력 보상 후 힘: {compensated_force}")
                # print(f"중력 보상 후 토크: {compensated_torque}")
                # print(f"최종 힘 (bias 제거): {f_final}")
                # print(f"최종 토크 (bias 제거): {t_final}")
                
                # 퍼블리시 (최종 값)
                msg = WrenchStamped()
                msg.header.stamp = rospy.Time.now()
                msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z   = f_final
                msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z = t_final
                pub.publish(msg)
            
            # # 주파수 통계 확인
            # stats = gravity_compensator.get_frequency_stats()
            # if stats:
            #     print("====== 주파수 모니터링 ======")
            #     print(f"측정 시간: {stats['elapsed']:.2f}초")
            #     print(f"IMU 데이터 수집 빈도: {stats['imu_freq']:.2f} Hz")
            #     print(f"FT 데이터 수집 빈도: {stats['ft_freq']:.2f} Hz") 
            #     print(f"메인 루프 실행 빈도: {stats['loop_freq']:.2f} Hz (목표: 100 Hz)")
            
            gravity_compensator.increment_loop_count()
            
        else:
            # Raw 값 모드 (bias 제거하여 0으로 만들기)
            try:
                ts, f_raw, t_raw = ft_reader.read_latest(timeout=0.001)
                
                # numpy array로 변환
                f_raw = np.array(f_raw)
                t_raw = np.array(t_raw)
                
                # 간단한 bias 제거 (첫 번째 값으로 초기화)
                if f_bias_initial is None:
                    f_bias_initial = f_raw.copy()
                    t_bias_initial = t_raw.copy()
                    print(f"Raw 초기 bias 설정: Force={f_bias_initial}, Torque={t_bias_initial}")
                
                # bias 제거
                f_zeroed = f_raw - f_bias_initial
                t_zeroed = t_raw - t_bias_initial
                
                # print("--------------------------------")
                # print(f"Raw Force: {f_raw}")
                # print(f"Raw Torque: {t_raw}")
                # print(f"Zeroed Force: {f_zeroed}")
                # print(f"Zeroed Torque: {t_zeroed}")
                
                # 퍼블리시 (bias 제거된 값)
                msg = WrenchStamped()
                msg.header.stamp = rospy.Time.now()
                msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z   = f_zeroed
                msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z = t_zeroed
                pub.publish(msg)
                
            except queue.Empty:
                continue
        
        rate.sleep()
        
        