
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import time
import json
import click
import cv2
import numpy as np
import scipy.spatial.transform as st
import socket
import sys
import termios
import tty
import select
import threading
from collections import deque
from utils.franka_api import FrankaAPI
import rospy
from std_msgs.msg import Float64MultiArray

def convert_ft_axis(ft):
    """Robot FT 축을 GUMI 축으로 변환 (실제 측정 기반)"""
    ft = np.asarray(ft)
    
    # 거리 비율에 따른 토크 스케일링 (GUMI: 12cm, Panda: 14cm)
    torque_scale = 12.0 / 14.0  # 0.857
    
    # 토크 스케일링 적용
    if ft.ndim == 2:
        ft[:,3] *= torque_scale  # Tx
        ft[:,4] *= torque_scale  # Ty  
        ft[:,5] *= torque_scale  # Tz
    else:
        ft[3] *= torque_scale  # Tx
        ft[4] *= torque_scale  # Ty  
        ft[5] *= torque_scale  # Tz

    if ft.ndim == 1:
        return np.array([ft[1], -ft[0], -ft[2], ft[4], -ft[3], -ft[5]], dtype=ft.dtype)
    else:
        # Force와 Torque 모두 같은 변환 적용
        return np.stack([ft[:,1],-ft[:,0], -ft[:,2], ft[:,4], -ft[:,3], -ft[:,5]], axis=-1)

class FTCollector:
    """FrankaAPI 로부터 힘 벡터를 원하는 Hz 로 계속 가져와 ring-buffer에 보관"""
    def __init__(self, api, rate_hz: float = 100., buf_len: int = 256, enable_ros=False):
        self.api = api
        self.rate = rate_hz
        # 윈도잉용
        self.buf  = deque(maxlen=buf_len)      # [(timestamp,np.array(6)), ...]
        # 세션 풀 로그용
        self.full_ts = []
        self.full_ft = []
        self.full_ts_list = []
        self.full_ft_list = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._th   = None
        self.zero_force = np.zeros(6, dtype=np.float32)  # ← zero-force 보정값
        
        # FT 필터링을 위한 변수들
        self.filter_alpha = 0.3  # 저역통과필터 계수 (0.1~0.5 사이 권장, 작을수록 더 부드러움)
        self.filtered_ft = None  # 필터링된 FT 데이터
        
        # ROS 관련 변수들
        self.enable_ros = enable_ros
        self.ft_pub = None
        self.ft_filtered_pub = None
        self.ft_axis_pubs = {}  # 개별 축별 publisher
        if enable_ros:
            self.ft_pub = rospy.Publisher('/ft_sensor/raw', Float64MultiArray, queue_size=10)
            self.ft_filtered_pub = rospy.Publisher('/ft_sensor/filtered', Float64MultiArray, queue_size=10)
            
            # 개별 축별 publisher 생성
            axis_names = ['force.x', 'force.y', 'force.z', 'torque.x', 'torque.y', 'torque.z']
            for i, axis in enumerate(axis_names):
                self.ft_axis_pubs[f'raw.{axis}'] = rospy.Publisher(f'/ft_sensor/raw.{axis}', Float64MultiArray, queue_size=10)
                self.ft_axis_pubs[f'filtered.{axis}'] = rospy.Publisher(f'/ft_sensor/filtered.{axis}', Float64MultiArray, queue_size=10)

    def _loop(self):
        period = 1.0 / self.rate
        nxt = time.perf_counter()
        while not self._stop.is_set():
            ts = time.time()
            f  = self.api.get_force_sync()      # REST 호출
            t = self.api.get_torque_sync()
            if f is not None and t is not None:
                # FT 데이터 결합
                ft_data = np.concatenate([f, t])
                
                # zero-force 보정 적용
                ft_data = ft_data - self.zero_force
                
                # 좌표계 변환 적용
                ft_data = convert_ft_axis(ft_data)
                
                # 저역통과필터 적용
                if self.filtered_ft is None:
                    self.filtered_ft = ft_data.copy()
                else:
                    self.filtered_ft = self.filter_alpha * ft_data + (1 - self.filter_alpha) * self.filtered_ft
                
                # ROS로 publish (raw 데이터)
                if self.enable_ros and self.ft_pub:
                    try:
                        raw_msg = Float64MultiArray()
                        raw_msg.data = ft_data.tolist()
                        self.ft_pub.publish(raw_msg)
                        
                        # 개별 축별 raw 데이터 publish
                        axis_names = ['force.x', 'force.y', 'force.z', 'torque.x', 'torque.y', 'torque.z']
                        for i, axis in enumerate(axis_names):
                            axis_msg = Float64MultiArray()
                            axis_msg.data = [ft_data[i]]
                            self.ft_axis_pubs[f'raw.{axis}'].publish(axis_msg)
                    except Exception as e:
                        pass  # ROS publish 실패시 무시
                
                # ROS로 publish (필터링된 데이터)
                if self.enable_ros and self.ft_filtered_pub:
                    try:
                        filtered_msg = Float64MultiArray()
                        filtered_msg.data = self.filtered_ft.tolist()
                        self.ft_filtered_pub.publish(filtered_msg)
                        
                        # 개별 축별 필터링된 데이터 publish
                        axis_names = ['force.x', 'force.y', 'force.z', 'torque.x', 'torque.y', 'torque.z']
                        for i, axis in enumerate(axis_names):
                            axis_msg = Float64MultiArray()
                            axis_msg.data = [self.filtered_ft[i]]
                            self.ft_axis_pubs[f'filtered.{axis}'].publish(axis_msg)
                    except Exception as e:
                        pass  # ROS publish 실패시 무시
                
                with self._lock:
                    self.buf.append((ts, self.filtered_ft))
                    # **풀 로그에도 append**
                    self.full_ts.append(ts)
                    self.full_ft.append(self.filtered_ft)
                    self.full_ts_list.append(ts)
                    self.full_ft_list.append(self.filtered_ft)
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

    def window(self, length: int):
        """최근 length 개 샘플 반환 (정규화된 FT 데이터)"""
        with self._lock:
            data = list(self.buf)[-length:]
        if len(data) < length:
            pad = [(0.0, np.zeros(6, np.float32))] * (length-len(data))
            data = pad + data
        ts, f = zip(*data)
        f = np.vstack(f)
        return f, np.array(ts, np.float64)

def save_ft_data(ft_collector, filename="ft_trace.csv"):
    """
    FTCollector.buf 안의 (timestamp, [fx,fy,fz,tx,ty,tz]) 전체를 CSV로 저장
    """
    # 1) collector 정지
    ft_collector.stop()

    # 2) 뮤텍트 잡고 버퍼 전체 복사
    with ft_collector._lock:
        data = list(ft_collector.buf)

    # 3) 빈 경우 처리
    if not data:
        print("No FT data collected.")
        return

    # 4) timestamp 와 force–torque 분리
    ts, ft = zip(*data)                     # ts: tuple of floats, ft: tuple of length-6 arrays
    arr_ts = np.array(ts, dtype=np.float64).reshape(-1, 1)   # (N,1)
    arr_ft = np.vstack(ft).astype(np.float32)               # (N,6)

    # 5) 합치기
    csv_data = np.hstack([arr_ts, arr_ft])  # (N,7)

    # 6) 헤더 정의
    header = "timestamp,fx,fy,fz,tx,ty,tz"

    # 7) 저장
    np.savetxt(filename,
               csv_data,
               delimiter=",",
               header=header,
               comments="")  # comments="" 로 '#' 제거

    print(f"저장 완료 ▶ {filename} ({csv_data.shape[0]} samples)")

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

def check_keyboard_input():
    """비차단 방식으로 키보드 입력 확인"""
    if select.select([sys.stdin], [], [], 0)[0]:
        key = sys.stdin.read(1)
        return key
    return None

@click.command()
@click.option('--server_ip', '-si', default='172.27.190.125', help="서버 IP 주소")
@click.option('--rate_hz', '-r', default=100, type=float, help="FT 데이터 수집 주기 (Hz)")
@click.option('--enable_ros', '-ros', is_flag=True, help="ROS로 FT 데이터 publish")
@click.option('--save_data', '-s', is_flag=True, help="FT 데이터를 CSV로 저장")
@click.option('--ft_filter_alpha', '-fa', default=0.3, type=float, help="FT 필터 강도 (0.1~0.5 권장, 작을수록 더 부드러움)")
@click.option('--output_file', '-o', default='ft_data.csv', help="저장할 파일명")
def main(server_ip, rate_hz, enable_ros, save_data, ft_filter_alpha, output_file):
    
    # ROS 초기화 (ROS 활성화시에만)
    if enable_ros:
        rospy.init_node('ft_sensor_publisher', anonymous=True)
        print("📡 ROS 노드 초기화 완료")
    
    # FrankaAPI 초기화
    print("=" * 60)
    print("🤖 FrankaAPI 초기화 중...")
    try:
        franka_api = FrankaAPI(server_ip)
        
        # 현재 로봇 상태 확인
        current_pose = franka_api.get_pose_sync()
        current_gripper = franka_api.get_gripper_sync()
        print(f"current_pose: {current_pose}")
        print(f"current_gripper: {current_gripper:.3f}")
        
    except Exception as e:
        print(f"❌ FrankaAPI 초기화 중 오류 발생: {e}")
        return
    
    # FTCollector 초기화
    ft_collector = FTCollector(franka_api, rate_hz=rate_hz, enable_ros=enable_ros)
    
    # FT 필터 강도 설정
    ft_collector.set_filter_alpha(ft_filter_alpha)
    
    # === FT zero-force 캘리브레이션 ===
    print("\n⚠️  FT 센서 캘리브레이션을 시작합니다.")
    print("로봇에 외부 힘이 가해지지 않도록 하고 Enter를 누르세요...")
    input()
    calibrate_ft_zero(ft_collector, num_samples=200)
    
    # FT 데이터 수집 시작
    ft_collector.start()
    print("✅ FT 데이터 수집이 시작되었습니다.")
    
    if enable_ros:
        print("📡 ROS 토픽으로 FT 데이터를 publish합니다:")
        print("   - /ft_sensor/raw: 원본 FT 데이터 (6차원)")
        print("   - /ft_sensor/filtered: 필터링된 FT 데이터 (6차원)")
        print("   - /ft_sensor/raw.force.x, /ft_sensor/raw.force.y, /ft_sensor/raw.force.z")
        print("   - /ft_sensor/raw.torque.x, /ft_sensor/raw.torque.y, /ft_sensor/raw.torque.z")
        print("   - /ft_sensor/filtered.force.x, /ft_sensor/filtered.force.y, /ft_sensor/filtered.force.z")
        print("   - /ft_sensor/filtered.torque.x, /ft_sensor/filtered.torque.y, /ft_sensor/filtered.torque.z")
    
    print("=" * 60)
    print("=== 조작 방법 ===")
    print("키보드 'q': 프로그램 종료")
    print("키보드 '1': FT 필터 강도 증가 (더 부드럽게)")
    print("키보드 '2': FT 필터 강도 감소 (더 민감하게)")
    print("키보드 '0': FT 필터 리셋")
    print("키보드 'c': FT 캘리브레이션 재실행")
    print("=" * 60)
    
    # 터미널 설정 저장
    old_settings = termios.tcgetattr(sys.stdin)
    
    try:
        # 터미널을 raw 모드로 변경
        tty.setraw(sys.stdin.fileno())
        
        start_time = time.time()
        sample_count = 0
        
        while True:
            # 키보드 입력 체크
            key = check_keyboard_input()
            if key:
                if key == 'q':
                    print("\n프로그램 종료")
                    break
                elif key == '1':
                    # FT 필터 강도 증가 (더 부드럽게)
                    new_alpha = max(0.01, ft_collector.filter_alpha - 0.05)
                    ft_collector.set_filter_alpha(new_alpha)
                elif key == '2':
                    # FT 필터 강도 감소 (더 민감하게)
                    new_alpha = min(0.9, ft_collector.filter_alpha + 0.05)
                    ft_collector.set_filter_alpha(new_alpha)
                elif key == '0':
                    # FT 필터 리셋
                    ft_collector.reset_filter()
                elif key == 'c':
                    # FT 캘리브레이션 재실행
                    print("\n⚠️  FT 센서 캘리브레이션을 재실행합니다.")
                    print("로봇에 외부 힘이 가해지지 않도록 하고 Enter를 누르세요...")
                    input()
                    calibrate_ft_zero(ft_collector, num_samples=200)
            
            # 현재 FT 데이터 출력 (1초마다)
            current_time = time.time()
            if current_time - start_time >= 1.0:
                with ft_collector._lock:
                    if len(ft_collector.buf) > 0:
                        latest_ft = ft_collector.buf[-1][1]  # 최신 FT 데이터
                        sample_count = len(ft_collector.full_ft)
                        elapsed = current_time - start_time
                        actual_rate = sample_count / elapsed
                        
                        print(f"\r[FT] Samples: {sample_count} | Rate: {actual_rate:.1f} Hz | "
                              f"Filter: {ft_collector.filter_alpha:.2f} | "
                              f"FT: [{latest_ft[0]:6.3f}, {latest_ft[1]:6.3f}, {latest_ft[2]:6.3f}, "
                              f"{latest_ft[3]:6.3f}, {latest_ft[4]:6.3f}, {latest_ft[5]:6.3f}]", 
                              end='', flush=True)
                
                start_time = current_time
            
            time.sleep(0.01)  # 10ms 대기
    
    finally:
        # FT 데이터 저장 (옵션)
        if save_data:
            print(f"\n📊 FT 데이터를 {output_file}로 저장 중...")
            save_ft_data(ft_collector, filename=output_file)
        
        # FT collector 정리
        ft_collector.stop()
        
        # 터미널 설정 복원
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print("\n프로그램을 종료합니다.")

if __name__ == "__main__":
    main()