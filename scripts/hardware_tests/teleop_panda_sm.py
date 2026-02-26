
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import time
import json
from multiprocessing.managers import SharedMemoryManager
import click
import cv2
import numpy as np
import scipy.spatial.transform as st
import socket
from utils.spacemouse_device import SpacemouseDevice as Spacemouse
from utils.precise_sleep import precise_wait
import sys
import termios
import tty
import select
import threading
from collections import deque
from utils.franka_api import FrankaAPI


class FTCollector:
    """FrankaAPI 로부터 힘 벡터를 원하는 Hz 로 계속 가져와 ring-buffer에 보관"""
    def __init__(self, api, rate_hz: float = 100., buf_len: int = 256, enable_ros=False):
        self.api = api
        self.rate = rate_hz
        # 윈도잉용
        self.buf = deque(maxlen=buf_len)      # [(timestamp,np.array(6)), ...]
        # 세션 풀 로그용 - raw 데이터
        self.full_ts = []
        self.full_ft_raw = []  # raw FT 데이터
        self.full_ft_normalized = []  # 정규화된 FT 데이터
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._th = None
        self.zero_force = np.zeros(6, dtype=np.float32)  # ← zero-force 보정값
        
        # FT 필터링을 위한 변수들
        self.filter_alpha = 0.3  # 저역통과필터 계수 (0.1~0.5 사이 권장, 작을수록 더 부드러움)
        self.filtered_ft = None  # 필터링된 FT 데이터
        
        # ROS 관련 변수들
        self.enable_ros = enable_ros
        self.ft_pub = None
        self.ft_filtered_pub = None
        if enable_ros:
            import rospy
            from std_msgs.msg import Float64MultiArray
            self.ft_pub = rospy.Publisher('/ft_sensor/raw', Float64MultiArray, queue_size=10)
            self.ft_filtered_pub = rospy.Publisher('/ft_sensor/filtered', Float64MultiArray, queue_size=10)
        
        # EMA 정규화기 추가 (zarr 코드와 동일한 파라미터)
        self.normalizer = RunningEMANormalizer(
            beta=0.95,          # zarr 코드와 동일
            warmup_steps=10,    # zarr 코드와 동일
            tanh_c=1.5,         # zarr 코드와 동일
            min_scale=1e-3      # zarr 코드와 동일
        )

    def _loop(self):
        period = 1.0 / self.rate
        nxt = time.perf_counter()
        last_normalize_time = time.time()
        normalize_interval = 1.0  # 1초마다 정규화 수행
        
        while not self._stop.is_set():
            ts = time.time()
            f = self.api.get_force_sync()      # REST 호출
            t = self.api.get_torque_sync()
            if f is not None and t is not None:
                # FT 데이터 결합
                ft_vec = np.concatenate([f, t])
                # zero_force 보정 적용
                ft_vec = ft_vec - self.zero_force
                # 좌표계 변환 적용
                ft_vec = convert_ft_axis(ft_vec)
                
                # 저역통과필터 적용
                if self.filtered_ft is None:
                    self.filtered_ft = ft_vec.copy()
                else:
                    self.filtered_ft = self.filter_alpha * ft_vec + (1 - self.filter_alpha) * self.filtered_ft
                
                # ROS로 publish (raw 데이터)
                if self.enable_ros and self.ft_pub:
                    try:
                        raw_msg = Float64MultiArray()
                        raw_msg.data = ft_vec.tolist()
                        self.ft_pub.publish(raw_msg)
                    except Exception as e:
                        pass  # ROS publish 실패시 무시
                
                # ROS로 publish (필터링된 데이터)
                if self.enable_ros and self.ft_filtered_pub:
                    try:
                        filtered_msg = Float64MultiArray()
                        filtered_msg.data = self.filtered_ft.tolist()
                        self.ft_filtered_pub.publish(filtered_msg)
                    except Exception as e:
                        pass  # ROS publish 실패시 무시
                
                with self._lock:
                    self.buf.append((ts, self.filtered_ft))  # 필터링된 데이터를 버퍼에 저장
                    # **풀 로그에도 append**
                    self.full_ts.append(ts)
                    self.full_ft_raw.append(ft_vec)  # raw 데이터는 별도 저장
                
                # 주기적으로 EMA 정규화 수행 (1초마다)
                if ts - last_normalize_time >= normalize_interval:
                    with self._lock:
                        if len(self.full_ft_raw) > 0:
                            # 최근 수집된 raw 데이터에 EMA 정규화 적용
                            raw_data = np.array(self.full_ft_raw)
                            try:
                                normalized_data = self.normalizer.normalize(raw_data)
                                # 정규화된 데이터 저장 (기존 데이터 덮어쓰기)
                                self.full_ft_normalized = normalized_data.tolist()
                            except Exception as e:
                                # 정규화 실패시 계속 진행
                                pass
                    last_normalize_time = ts
            
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
    
    def set_filter_alpha(self, alpha):
        """필터 강도 설정 (0.1~0.5 권장, 작을수록 더 부드러움)"""
        self.filter_alpha = np.clip(alpha, 0.01, 0.9)
        print(f"FT 필터 강도 설정: {self.filter_alpha:.2f}")
    
    def reset_filter(self):
        """필터 상태 초기화"""
        self.filtered_ft = None
        print("FT 필터 상태 초기화 완료")

class RunningEMANormalizer:
    def __init__(self, dim=6, beta=0.95, eps=1e-6, warmup_steps=10, tanh_c=1.5, min_scale=1e-3):
        self.beta = beta
        self.eps = eps
        self.warmup_steps = warmup_steps
        self.tanh_c = tanh_c
        self.min_scale = min_scale
        
        self.ema_scale = np.zeros(dim) + eps
        self.f0 = None
        self.step_count = 0

    def normalize(self, ft_data):
        """zarr 저장 코드와 동일한 EMA 정규화 적용"""
        if self.f0 is None:
            self.f0 = ft_data[0].copy()
        
        N = len(ft_data)
        normalized = np.zeros_like(ft_data)
        
        for t in range(N):
            f_c = ft_data[t] - self.f0
            r = np.abs(f_c)
            
            if self.step_count + t < self.warmup_steps:
                if self.step_count + t == 0:
                    self.ema_scale = r + self.eps
                else:
                    self.ema_scale = np.maximum(self.ema_scale, r)
            else:
                self.ema_scale = self.beta * self.ema_scale + (1 - self.beta) * r
            
            self.ema_scale = np.maximum(self.ema_scale, self.min_scale)
            r_norm = r / self.ema_scale
            mag = np.tanh(np.log1p(r_norm) / self.tanh_c)
            normalized[t] = np.sign(f_c) * mag
        
        self.step_count += N
        return normalized

def convert_ft_axis(ft):
    # ft: (N, 6) 또는 (6,)
    # 좌표계 변환: [x,y,z] → [-y, x, -z]
    # Force: [Fx, Fy, Fz] → [-Fy, Fx, -Fz]
    # Torque: [Tx, Ty, Tz] → [-Ty, Tx, -Tz]
    ft = np.asarray(ft)
    if ft.ndim == 1:
        return np.array([-ft[1], ft[0], -ft[2], -ft[4], ft[3], -ft[5]], dtype=ft.dtype)
    else:
        return np.stack([-ft[:,1], ft[:,0], -ft[:,2], -ft[:,4], ft[:,3], -ft[:,5]], axis=-1)

def save_ft_data(ft_collector, filename="teleop_ft_data.csv", save_normalized=True):
    """FTCollector의 수집된 데이터를 CSV로 저장"""
    ft_collector.stop()
    
    with ft_collector._lock:
        timestamps = ft_collector.full_ts.copy()
        raw_ft_data = ft_collector.full_ft_raw.copy()
        normalized_ft_data = ft_collector.full_ft_normalized.copy()
    
    if not timestamps or not raw_ft_data:
        print("No FT data collected.")
        return
    
    # Raw 데이터 저장
    arr_ts = np.array(timestamps, dtype=np.float64).reshape(-1, 1)
    arr_ft_raw = np.array(raw_ft_data, dtype=np.float32)
    
    # Raw 데이터 파일명
    base_name = filename.replace('.csv', '')
    raw_filename = f"{base_name}_raw.csv"
    
    csv_data_raw = np.hstack([arr_ts, arr_ft_raw])
    header = "timestamp,fx,fy,fz,tx,ty,tz"
    
    np.savetxt(raw_filename,
               csv_data_raw,
               delimiter=",",
               header=header,
               comments="")
    
    print(f"Raw 데이터 저장 완료 ▶ {raw_filename} ({csv_data_raw.shape[0]} samples)")
    
    # 정규화된 데이터 저장 (옵션)
    if save_normalized and normalized_ft_data:
        # 마지막 정규화 수행 (최신 데이터 반영)
        try:
            final_normalized = ft_collector.normalizer.normalize(arr_ft_raw)
            arr_ft_normalized = final_normalized.astype(np.float32)
            
            normalized_filename = f"{base_name}_normalized.csv"
            csv_data_normalized = np.hstack([arr_ts, arr_ft_normalized])
            
            np.savetxt(normalized_filename,
                       csv_data_normalized,
                       delimiter=",",
                       header=header,
                       comments="")
            
            print(f"정규화된 데이터 저장 완료 ▶ {normalized_filename} ({csv_data_normalized.shape[0]} samples)")
        except Exception as e:
            print(f"정규화된 데이터 저장 실패: {e}")
    
    # 기본 파일명으로 정규화된 데이터 저장 (학습용)
    if save_normalized and normalized_ft_data:
        try:
            np.savetxt(filename,
                       csv_data_normalized,
                       delimiter=",",
                       header=header,
                       comments="")
            print(f"학습용 데이터 저장 완료 ▶ {filename} (정규화됨)")
        except Exception as e:
            print(f"학습용 데이터 저장 실패: {e}")

def calibrate_ft_zero(ft_collector, num_samples=100):
    """FT 센서의 zero-force(잔류 힘) 값을 캘리브레이션"""
    print(f"[FT Calibration] {num_samples}개 샘플로 zero-force 캘리브레이션 중... (로봇에 힘이 가해지지 않게 하세요)")
    samples = []
    for i in range(num_samples):
        f = ft_collector.api.get_force_sync()
        t = ft_collector.api.get_torque_sync()
        if f is not None and t is not None:
            samples.append(np.concatenate([f, t]))
        time.sleep(0.01)
    if len(samples) == 0:
        print("[FT Calibration] 캘리브레이션 실패: 샘플이 없습니다.")
        return
    zero_force = np.mean(samples, axis=0)
    ft_collector.zero_force = zero_force
    print(f"[FT Calibration] 완료! zero-force: {zero_force.round(4)}")

def connect_to_server(host='localhost', port=5001, timeout=5):
    while True:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        client_socket.settimeout(timeout)
        try:
            client_socket.connect((host, port))
            print(f"서버 {host}:{port}에 연결되었습니다.")
            return client_socket
        except (ConnectionRefusedError, socket.timeout):
            print(f"서버 {host}:{port}에 연결 실패. 1초 후 재시도...")
            time.sleep(1)
            return None

def check_keyboard_input():
    """비차단 방식으로 키보드 입력 확인"""
    if select.select([sys.stdin], [], [], 0)[0]:
        key = sys.stdin.read(1)
        return key
    return None

@click.command()
@click.option('--frequency', '-f', default=50, type=float, help="제어 주기 (Hz)")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="명령 지연 시간 (초)")
@click.option('--server_ip', '-si', default='172.27.190.125', help="서버 IP 주소")
@click.option('--server_port', '-sp', default=4999, type=int, help="서버 포트 번호")
@click.option('--save_ft', '-ft', is_flag=True, help="FT 센서 데이터 수집 및 저장")
@click.option('--enable_ros', '-ros', is_flag=True, help="ROS로 FT 데이터 publish")
@click.option('--ft_filter_alpha', '-fa', default=0.3, type=float, help="FT 필터 강도 (0.1~0.5 권장, 작을수록 더 부드러움)")
def main(frequency, command_latency, server_ip, server_port, save_ft, enable_ros, ft_filter_alpha):
    dt = 1 / frequency
    max_pos_speed = 0.1  # m/s
    max_rot_speed = 0.3  # rad/s
    
    # 사전 계산된 상수
    pos_scale = max_pos_speed * dt
    rot_scale = max_rot_speed * dt
    
    # 1.초기 위치 설정
    initial_pose = np.array([0.5, 0.0162628, 0.16830516, 0.9986493, 0.01434065, 0.04985237, -0.00294379])
    # pos_bounds = np.array([[initial_pose[0]-0.2, initial_pose[1]-0.2], 
    #                       [initial_pose[0]+0.2, initial_pose[1]+0.2]], dtype=np.float32)

    # === FT 센서 초기화 (옵션) ===
    franka_api = None
    ft_collector = None
    if save_ft or enable_ros:
        print("=" * 60)
        print("🤖 FT 센서 초기화 중...")
        try:
            franka_api = FrankaAPI(server_ip)
            ft_collector = FTCollector(franka_api, rate_hz=100, enable_ros=enable_ros)
            
            # FT 필터 강도 설정
            ft_collector.set_filter_alpha(ft_filter_alpha)
            
            # 현재 로봇 상태 확인
            current_pose = franka_api.get_pose_sync()
            current_gripper = franka_api.get_gripper_sync()
            print(f"current_pose: {current_pose}")
            print(f"current_gripper: {current_gripper:.3f}")
            
            # FT zero-force 캘리브레이션
            print("\n⚠️  FT 센서 캘리브레이션을 시작합니다.")
            print("로봇에 외부 힘이 가해지지 않도록 하고 Enter를 누르세요...")
            input()
            calibrate_ft_zero(ft_collector, num_samples=100)
            
            # FT 데이터 수집 시작
            ft_collector.start()
            print("✅ FT 데이터 수집이 시작되었습니다.")
            if enable_ros:
                print("📡 ROS 토픽으로 FT 데이터를 publish합니다:")
                print("   - /ft_sensor/raw: 원본 FT 데이터")
                print("   - /ft_sensor/filtered: 필터링된 FT 데이터")
            
        except Exception as e:
            print(f"❌ FT 센서 초기화 중 오류 발생: {e}")
            print("FT 데이터 수집 없이 계속 진행합니다.")
            save_ft = False
        print("=" * 60)
    
    # 서버 연결
    client_socket = connect_to_server(server_ip, server_port)
    if client_socket is None:
        print("초기 연결 실패. 프로그램 종료.")
        # FT collector 정리
        if ft_collector:
            ft_collector.stop()
        return

    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager) as sm:
            print("🎮 SpaceMouse 텔레오퍼레이션 준비 완료!")
            if save_ft:
                print("📊 FT 센서 데이터 수집 활성화됨 - 프로그램 종료시 'teleop_ft_data.csv'로 저장됩니다.")
            if enable_ros:
                print("📡 ROS FT 데이터 publish 활성화됨")
            if not save_ft and not enable_ros:
                print("💡 FT 센서 데이터 수집을 원하면 '--save_ft' 또는 '-ft' 옵션을 사용하세요.")
                print("💡 ROS publish를 원하면 '--enable_ros' 또는 '-ros' 옵션을 사용하세요.")
            
            print("\n=== 조작 방법 ===")
            print("SpaceMouse: 6DOF 위치/회전 제어")
            print("버튼 0: 그리퍼 토글 (열기/닫기)")
            print("버튼 1: Z축 이동 허용")
            print("키보드 'q': 프로그램 종료")
            print("키보드 'r': 초기 위치로 리셋")
            if save_ft or enable_ros:
                print("키보드 '1': FT 필터 강도 증가 (더 부드럽게)")
                print("키보드 '2': FT 필터 강도 감소 (더 민감하게)")
                print("키보드 '0': FT 필터 리셋")
            print("=" * 60)
            
            target_pose = initial_pose.copy()
            current_pos = initial_pose[:3].copy()
            current_rot = st.Rotation.from_quat(initial_pose[3:])
            
            # 그리퍼 상태 추적
            gripper_state = 'open'
            last_btn0_state = False
            last_command_time = time.monotonic()
            
            t_start = time.monotonic()
            iter_idx = 0
            
            # 터미널 설정 저장
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                # 터미널을 raw 모드로 변경
                tty.setraw(sys.stdin.fileno())
                
                while True:
                    cycle_start = time.monotonic()
                    
                    # 스페이스마우스 상태 읽기
                    sm_state = sm.get_motion_state_transformed()
                    btn0 = sm.is_button_pressed(0)
                    btn1 = sm.is_button_pressed(1)
                    
                    # 그리퍼 토글 처리 (버튼 상태 변화 감지)
                    if btn0 and not last_btn0_state:
                        gripper_state = 'close' if gripper_state == 'open' else 'open'
                        last_command_time = cycle_start
                    last_btn0_state = btn0

                    # 입력 변환 및 스케일링
                    dpos = sm_state[:3] * pos_scale
                    drot_xyz = sm_state[3:] * rot_scale
                    
                    # 키보드 입력 체크
                    key = check_keyboard_input()
                    if key:
                        if key == 'q':
                            print("\n프로그램 종료")
                            # FT 데이터 저장
                            if save_ft and ft_collector:
                                print("FT 데이터 저장 중...")
                                save_ft_data(ft_collector, filename="teleop_ft_data.csv")
                            break
                        elif key == 'r':
                            print("\n초기 위치로 리셋")
                            
                            current_pos = initial_pose[:3].copy()
                            current_rot = st.Rotation.from_quat(initial_pose[3:])
                            
                            # Target pose [x, y, z, qx, qy, qz, qw] [1,0,0,0]
                            target_pose[:3] = current_pos
                            target_pose[3:] = current_rot.as_quat()
                            
                            
                            # 2. Server에 전송
                            # 리셋 명령 전송
                            try:
                                data = {
                                    'target_pose': target_pose.tolist(),
                                    'timestamp': cycle_start,
                                    'gripper_command': gripper_state,
                                    'reset': True
                                }
                                message = json.dumps(data, separators=(',', ':'))
                                client_socket.sendall(message.encode('utf-8') + b'\n')
                                time.sleep(0.1)
                            except (socket.error, BrokenPipeError) as e:
                                print(f"\n리셋 명령 전송 실패: {e}")
                            continue
                        elif key == '1' and (save_ft or enable_ros):
                            # FT 필터 강도 증가 (더 부드럽게)
                            new_alpha = max(0.01, ft_collector.filter_alpha - 0.05)
                            ft_collector.set_filter_alpha(new_alpha)
                            continue
                        elif key == '2' and (save_ft or enable_ros):
                            # FT 필터 강도 감소 (더 민감하게)
                            new_alpha = min(0.9, ft_collector.filter_alpha + 0.05)
                            ft_collector.set_filter_alpha(new_alpha)
                            continue
                        elif key == '0' and (save_ft or enable_ros):
                            # FT 필터 리셋
                            ft_collector.reset_filter()
                            continue

                    # # 위치 업데이트 (Z축)
                    # if btn1:  # Z축 이동 허용
                    #     current_pos += dpos
                    # else:  # XY 평면 이동만 허용
                    #     current_pos[0:2] += dpos[0:2]
                    
                    current_pos += dpos
                    # 회전 업데이트 - 항상 적용
                    if np.any(drot_xyz != 0):
                        drot = st.Rotation.from_euler('xyz', drot_xyz)
                        current_rot = drot * current_rot

                    # # 작업영역 제한
                    # current_pos[:2] = np.clip(current_pos[:2], pos_bounds[0], pos_bounds[1])
                    # current_pos[2] = np.clip(current_pos[2], 0.01, 0.5)  # Z축 제한

                    # 타겟 포즈 업데이트
                    target_pose[:3] = current_pos
                    target_pose[3:] = current_rot.as_quat()

                    # 일반 데이터 전송
                    try:
                        data = {
                            'target_pose': target_pose.tolist(),
                            'timestamp': cycle_start,
                            'gripper_command': gripper_state,
                            'reset': False  # 일반 명령임을 표시
                        }
                        message = json.dumps(data, separators=(',', ':'))
                        client_socket.sendall(message.encode('utf-8') + b'\n')
                    except (socket.error, BrokenPipeError) as e:
                        print(f"서버 연결 오류: {e}. 재연결 시도...")
                        client_socket.close()
                        client_socket = connect_to_server(server_ip, server_port)
                        if client_socket is None:
                            print("재연결 실패. 루프 유지.")
                            time.sleep(0.1)
                            continue

                    # 종료 조건 체크 (10회마다)
                    if iter_idx % 10 == 0:
                        if cv2.waitKey(1) == ord('q'):
                            break

                    # 주기 디버깅 (100회마다)
                    if iter_idx % 100 == 0 and iter_idx > 0:
                        elapsed = time.monotonic() - t_start
                        status_msg = f"Actual frequency: {iter_idx / elapsed:.2f} Hz"
                        if (save_ft or enable_ros) and ft_collector:
                            with ft_collector._lock:
                                ft_samples = len(ft_collector.full_ft_raw)
                                normalized_samples = len(ft_collector.full_ft_normalized)
                            status_msg += f" | FT samples: {ft_samples} (normalized: {normalized_samples})"
                            if enable_ros:
                                status_msg += f" | Filter alpha: {ft_collector.filter_alpha:.2f}"
                        print(status_msg)

                    # 정밀한 타이밍 제어
                    cycle_end = cycle_start + dt
                    sleep_time = cycle_end - time.monotonic()
                    if sleep_time > 0:
                        precise_wait(cycle_end)
                    
                    iter_idx += 1
                    
            finally:
                # FT 데이터 최종 저장 및 정리
                if save_ft and ft_collector:
                    print("\n최종 FT 데이터 저장 중...")
                    save_ft_data(ft_collector, filename="teleop_ft_data.csv")
                
                # FT collector 정리
                if ft_collector:
                    ft_collector.stop()
                
                # 터미널 설정 복원
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                client_socket.close()
                print("\n서버 연결을 종료합니다.")

if __name__ == "__main__":
    main()