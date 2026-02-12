import os
import sys

# Suppress Qt and OpenCV warnings
os.environ["QT_LOGGING_RULES"] = "qt.qpa.fonts=false;*.debug=false;driver.usb=false"
os.environ["QT_QPA_PLATFORM_NODEBUG"] = "1"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import cv2
import os
from datetime import datetime
import time
import numpy as np
import queue
import threading
from collections import deque
import pyrealsense2 as rs
from ahrs.filters import Madgwick
from tqdm import trange, tqdm
import argparse
import sys

from utils.rs_capture import RSCapture, DJICapture
from utils.ft_capture import AidinFTSensorUDP
from utils.gravity_compensation_utils import GravityCompensator

# 설정된 디렉토리 경로
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data', help='저장할 base 폴더명')
parser.add_argument('--add_cam', action='store_true', help='추가 카메라 사용 활성화')
parser.add_argument('--dji_device', type=str, default=None, help='DJI 액션캠이 연결된 /dev/video 경로')
args = parser.parse_args()
base_save_path = args.data_path
saving_images = False
episode_num = 1
handeye_save_path = None
pose_tracking_save_path = None
additional_cam_save_path = None  # 추가 카메라 경로
ft_data_save_path = None

# 추가 카메라 사용 여부
use_additional_cam = args.add_cam

# 스레드 제어 변수
stop_ft_thread = False
stop_image_save_thread = False
ft_thread = None
image_save_thread = None
ft_csv_file = None

# 이미지 저장 큐
image_queue = queue.Queue(maxsize=300)

# 핸드아이/DJI/추가 카메라가 없을 때 사용할 더미 프레임
DUMMY_HAND_FRAME = np.zeros((800, 1280, 3), dtype=np.uint8)
DUMMY_ADDITIONAL_FRAME = np.zeros((480, 640, 3), dtype=np.uint8)

# 주파수 모니터링 변수
MONITOR_INTERVAL = 10.0  # 10초마다 주파수 출력
image_count = 0
last_image_time = None
ft_count = 0
last_ft_time = None
ft_count_lock = threading.Lock()  # 스레드 안전한 카운팅을 위한 락
image_capture_count = 0
image_save_count = 0
capture_count_lock = threading.Lock()

# 진행 상황 표시 변수
episode_start_time = None
episode_image_count = 0
episode_ft_count = 0
episode_progress_bar = None
progress_lock = threading.Lock()

# 30Hz 설정 (이미지 캡처)
FRAME_RATE = 30  # Hz
FRAME_INTERVAL = 1.0 / FRAME_RATE  # 초 단위 (33.3ms)

# FT 데이터 수집 설정
FT_RATE = 200  # Hz (Aidin UDP 센서 최대 속도)
FT_INTERVAL = 1.0 / FT_RATE  # 초 단위 (5ms)
ft_dummy_mode = False  # FT 센서 미연결 시 더미 데이터 사용 여부

# 디렉토리 생성
os.makedirs(base_save_path, exist_ok=True)

def find_last_episode_number():
    """기존 에피소드 중 마지막 번호를 찾아서 반환"""
    if not os.path.exists(base_save_path):
        return 0
    
    episode_dirs = []
    for item in os.listdir(base_save_path):
        if os.path.isdir(os.path.join(base_save_path, item)) and item.startswith('episode_'):
            try:
                episode_num = int(item.split('_')[1])
                episode_dirs.append(episode_num)
            except (ValueError, IndexError):
                continue
    
    if not episode_dirs:
        return 0
    
    return max(episode_dirs)

# 초기화 시 마지막 에피소드 번호 찾기
episode_num = find_last_episode_number() + 1
print(f"기존 에피소드 중 마지막 번호: {episode_num - 1}, 새 에피소드는 {episode_num}부터 시작합니다.")

# 중력 보상기 초기화
gravity_compensator = GravityCompensator(
    mass_for_x=0.58,
    mass_for_y=0.53, 
    mass_for_z=0.7,
    com_ft=np.array([0.01, 0.01, 0.03]),
    g_const=9.81
)

def update_progress_bar():
    """진행 상황을 시각적으로 업데이트"""
    global episode_progress_bar, episode_start_time, episode_image_count, episode_ft_count
    
    if episode_progress_bar is None or episode_start_time is None:
        return
    
    current_time = time.time()
    elapsed_time = current_time - episode_start_time
    
    # 진행 상황 문자열 생성
    progress_str = f"📸 {episode_image_count} | 🔧 {episode_ft_count} | ⏱️ {elapsed_time:.1f}s"
    if ft_dummy_mode:
        progress_str += " (FT:DUMMY)"
    
    # 프로그레스 바 업데이트 (이미지 수를 기준으로 진행률 계산)
    # 30Hz로 1분간 수집한다고 가정하면 1800개가 목표
    target_images = 1000  # 1분 기준
    progress = min(episode_image_count, target_images)
    
    episode_progress_bar.total = target_images
    episode_progress_bar.n = progress
    episode_progress_bar.set_description(progress_str)
    episode_progress_bar.refresh()

def print_episode_summary():
    """에피소드 완료 시 통계 출력"""
    global episode_start_time, episode_image_count, episode_ft_count, ft_dummy_mode
    
    if episode_start_time is None:
        return
    
    total_time = time.time() - episode_start_time
    image_hz = episode_image_count / total_time if total_time > 0 else 0
    ft_hz = episode_ft_count / total_time if total_time > 0 else 0
    
    # 실제 저장된 파일 개수 확인
    save_counts = {}
    current_ep_dir = f"{base_save_path}/episode_{episode_num}"
    for cam in ['handeye', 'pose_tracking', 'additional_cam']:
        path = os.path.join(current_ep_dir, "images", cam)
        if os.path.exists(path):
            save_counts[cam] = len([f for f in os.listdir(path) if f.endswith('.jpg')])
        else:
            save_counts[cam] = 0

    print("\n" + "="*60)
    print(f"📊 Episode {episode_num} 완료! {'[FT DUMMY]' if ft_dummy_mode else ''}")
    print("="*60)
    print(f"⏱️  총 수집 시간: {total_time:.1f}초")
    print(f"📸 획득 이미지 (Base): {episode_image_count}개 ({image_hz:.1f} Hz)")
    print(f"💾 실제 저장된 파일:")
    print(f"   - Handeye: {save_counts['handeye']}개")
    print(f"   - DJI (Pose): {save_counts['pose_tracking']}개")
    if use_additional_cam:
        print(f"   - Additional: {save_counts['additional_cam']}개")
    print(f"🔧 FT 데이터: {episode_ft_count}개 ({ft_hz:.1f} Hz)")
    print(f"📁 저장 위치: {current_ep_dir}")
    print("="*60 + "\n")

def start_new_episode():
    """새 episode 디렉토리 생성."""
    global handeye_save_path, pose_tracking_save_path, additional_cam_save_path, ft_data_save_path, episode_num, ft_csv_file
    global image_count, last_image_time, ft_count, last_ft_time, image_capture_count, image_save_count
    global episode_start_time, episode_image_count, episode_ft_count, episode_progress_bar
    
    # 주파수 모니터링 변수 초기화
    image_count = 0
    last_image_time = time.time()
    ft_count = 0
    last_ft_time = time.time()
    image_capture_count = 0
    image_save_count = 0
    
    # 에피소드 진행 상황 초기화
    episode_start_time = time.time()
    episode_image_count = 0
    episode_ft_count = 0
    
    # 진행 상황 표시 초기화
    if episode_progress_bar is not None:
        episode_progress_bar.close()
    
    desc = "데이터 수집 중"
    if ft_dummy_mode:
        desc += " (FT:DUMMY)"
        
    episode_progress_bar = tqdm(total=1800, desc=desc, 
                               bar_format='{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]', 
                               ncols=85, leave=True)
    
    episode_dir = os.path.join(base_save_path, f'episode_{episode_num}')
    
    # 디렉토리가 이미 존재하는 경우 처리
    if os.path.exists(episode_dir):
        print(f"경고: episode_{episode_num} 디렉토리가 이미 존재합니다. 기존 데이터를 덮어쓰게 됩니다.")
        # 기존 디렉토리 삭제 후 새로 생성
        import shutil
        shutil.rmtree(episode_dir)
        print(f"기존 episode_{episode_num} 디렉토리를 삭제했습니다.")
    
    # 디렉토리 생성
    handeye_save_path = os.path.join(episode_dir, 'images', 'handeye')
    pose_tracking_save_path = os.path.join(episode_dir, 'images', 'pose_tracking')
    additional_cam_save_path = os.path.join(episode_dir, 'images', 'additional_cam')
    ft_data_save_path = os.path.join(episode_dir, 'ft_data')
    
    os.makedirs(handeye_save_path, exist_ok=True)
    os.makedirs(pose_tracking_save_path, exist_ok=True)
    os.makedirs(additional_cam_save_path, exist_ok=True)  # 추가 카메라 디렉토리
    os.makedirs(ft_data_save_path, exist_ok=True)
    
    # FT 데이터를 위한 CSV 파일 생성
    ft_csv_path = os.path.join(episode_dir, f'ft_data_episode_{episode_num}.csv')
    ft_csv_file = open(ft_csv_path, 'w')
    
    # CSV 헤더 작성
    ft_csv_file.write("timestamp,force_x,force_y,force_z,torque_x,torque_y,torque_z\n")
    
    print(f"새로운 에피소드 저장 위치 - handeye: {handeye_save_path}")
    print(f"새로운 에피소드 저장 위치 - pose_tracking: {pose_tracking_save_path}")
    print(f"새로운 에피소드 저장 위치 - additional_cam: {additional_cam_save_path}")  # 추가 출력
    print(f"FT 데이터 CSV 파일: {ft_csv_path}")

def read_latest_imu(pipe):
    """
    RealSense pipeline에서 남아 있는 프레임을 다 버리고
    accel + gyro Motion Frame 한 쌍을 numpy 벡터로 돌려준다.
    새 프레임이 없으면 None 반환
    """
    frames = pipe.poll_for_frames()
    if not frames:
        return None

    # 버퍼에 더 남아 있으면 끝까지 빼낸다
    while True:
        more = pipe.poll_for_frames()
        if not more:
            break
        frames = more  

    accel = frames.first_or_default(rs.stream.accel)
    gyro  = frames.first_or_default(rs.stream.gyro)
    if accel is None or gyro is None:
        return None 

    a = accel.as_motion_frame().get_motion_data()
    g = gyro .as_motion_frame().get_motion_data()
    acc_vec  = np.array([a.x, a.y, a.z], dtype=float)
    gyro_vec = np.array([g.x, g.y, g.z], dtype=float)
    return acc_vec, gyro_vec

def check_ft_frequency():
    """FT 데이터 수집 주파수 확인 및 출력"""
    global ft_count, last_ft_time
    
    current_time = time.time()
    elapsed = current_time - last_ft_time
    
    if elapsed >= MONITOR_INTERVAL:
        with ft_count_lock:
            local_ft_count = ft_count
            ft_count = 0
        
        ft_frequency = local_ft_count / elapsed
        print(f"FT 데이터 수집 주파수: {ft_frequency:.2f} Hz (목표: {FT_RATE} Hz) - {local_ft_count}개 샘플 / {elapsed:.2f}초")
        last_ft_time = current_time
        return True
    
    return False

def check_image_frequency():
    """이미지 수집 주파수 확인 및 출력"""
    global image_count, last_image_time, image_capture_count, image_save_count
    
    current_time = time.time()
    elapsed = current_time - last_image_time
    
    if elapsed >= MONITOR_INTERVAL:
        local_image_count = image_count
        image_count = 0
        
        with capture_count_lock:
            local_capture_count = image_capture_count
            local_save_count = image_save_count
        
        image_frequency = local_image_count / elapsed
        print(f"이미지 수집 주파수: {image_frequency:.2f} Hz (목표: {FRAME_RATE} Hz) - {local_image_count}개 이미지 / {elapsed:.2f}초")
        print(f"이미지 캡처/저장 현황: 캡처 {local_capture_count}개, 저장 {local_save_count}개, 큐 크기: {image_queue.qsize()}")
        last_image_time = current_time
        return True
    
    return False

def format_timestamp_from_float(timestamp_float):
    """float 타입의 시점(time.time())을 저장용 문자열로 변환"""
    dt = datetime.fromtimestamp(timestamp_float)
    return dt.strftime("%Y%m%d_%H%M%S_%f")[:-3]

def image_save_thread_func():
    """개별 프레임을 독립적으로 저장하는 스레드"""
    global stop_image_save_thread, image_save_count, episode_image_count
    
    print("이미지 개별 저장 스레드 시작")
    total_save_count = 0
    start_time = time.time()
    
    # JPEG 저장 설정
    jpeg_params = {
        'handeye': [cv2.IMWRITE_JPEG_QUALITY, 90],
        'pose_tracking': [cv2.IMWRITE_JPEG_QUALITY, 100],
        'additional_cam': [cv2.IMWRITE_JPEG_QUALITY, 90]
    }
    
    while not stop_image_save_thread or not image_queue.empty():
        try:
            item = image_queue.get(block=True, timeout=0.1)
            if item is None:
                image_queue.task_done()
                break
                
            subfolder_name, timestamp_str, frame = item
            
            # 각 에피소드 디렉토리 내의 정확한 경로 찾기
            if subfolder_name == 'handeye':
                save_dir = handeye_save_path
            elif subfolder_name == 'pose_tracking':
                save_dir = pose_tracking_save_path
            else:
                save_dir = additional_cam_save_path

            if save_dir is None:
                image_queue.task_done()
                continue

            save_path = os.path.join(save_dir, f"{timestamp_str}.jpg")
            cv2.imwrite(save_path, frame, jpeg_params.get(subfolder_name, [cv2.IMWRITE_JPEG_QUALITY, 90]))
            
            with capture_count_lock:
                image_save_count += 1
            if subfolder_name == 'handeye': # 기준 카메라 수만 프로그레스 바에 반영
                with progress_lock:
                    episode_image_count += 1
                    update_progress_bar()
            
            total_save_count += 1
            image_queue.task_done()
            
        except queue.Empty:
            continue
        except Exception as e:
            print(f"이미지 저장 오류: {e}")
    
    elapsed = time.time() - start_time
    print(f"이미지 저장 스레드 종료: 총 {total_save_count}개 저장, 평균 {total_save_count/elapsed:.1f}Hz")

def ft_collection_thread(ft_reader, imu_pipe, gravity_compensator):
    """FT 데이터를 독립적으로 수집하는 스레드 함수"""
    global stop_ft_thread, ft_csv_file, ft_count, episode_ft_count, ft_dummy_mode
    
    if ft_dummy_mode:
        print("FT Dummy 모드 실행 중 - 0으로 채워진 데이터를 저장합니다.")
    else:
        print("FT 데이터 수집 스레드 시작 (목표: 200Hz)")
    
    # FT 센서 버퍼 비우기 - 이전 데이터 제거
    if not ft_dummy_mode:
        print("FT 센서 버퍼 비우기 중...")
        for _ in range(50):  # 충분히 많은 데이터를 버림
            try:
                ft_reader.get_frame(timeout=0.001)
            except:
                break
    
    total_ft_count = 0
    start_collection_time = time.time()
    
    try:
        while not stop_ft_thread:
            if ft_dummy_mode:
                # Dummy 모드: 200Hz 주기로 0 데이터 생성
                time.sleep(FT_INTERVAL)
                compensated_force = np.zeros(3)
                compensated_torque = np.zeros(3)
            else:
                # IMU 업데이트
                if imu_pipe is not None:
                    gravity_compensator.update_imu(imu_pipe)

                # FT 데이터 읽기
                try:
                    ts, f_raw, t_raw = ft_reader.get_frame(timeout=0.001)
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"FT 데이터 읽기 오류: {str(e)}")
                    continue

                # FT 데이터 처리 및 중력 보상
                forces_filt, torques_filt = gravity_compensator.process_ft_data(f_raw, t_raw)
                compensated_force, compensated_torque = gravity_compensator.compensate_gravity(
                    forces_filt, torques_filt, gravity_compensation_on=True
                )

            ft_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            if ft_csv_file is not None:
                ft_csv_file.write(f"{ft_timestamp}," +
                                  f"{compensated_force[0]},{compensated_force[1]},{compensated_force[2]}," +
                                  f"{compensated_torque[0]},{compensated_torque[1]},{compensated_torque[2]}\n")
                ft_csv_file.flush()

            with ft_count_lock:
                ft_count += 1
                total_ft_count += 1
            
            # 에피소드 진행 상황 업데이트
            with progress_lock:
                episode_ft_count += 1
                update_progress_bar()

    finally:
        if not ft_dummy_mode:
            print("FT 스레드 종료 - 남은 데이터 저장 시도")
            while True:
                try:
                    ts, f_raw, t_raw = ft_reader.get_frame(timeout=0.001)
                    
                    # FT 데이터 처리 및 중력 보상
                    forces_filt, torques_filt = gravity_compensator.process_ft_data(f_raw, t_raw)
                    compensated_force, compensated_torque = gravity_compensator.compensate_gravity(
                        forces_filt, torques_filt, gravity_compensation_on=True
                    )

                    ft_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    if ft_csv_file is not None:
                        ft_csv_file.write(f"{ft_timestamp}," +
                                          f"{compensated_force[0]},{compensated_force[1]},{compensated_force[2]}," +
                                          f"{compensated_torque[0]},{compensated_torque[1]},{compensated_torque[2]}\n")
                        ft_csv_file.flush()

                    with ft_count_lock:
                        ft_count += 1
                        total_ft_count += 1
                    
                    # 에피소드 진행 상황 업데이트
                    with progress_lock:
                        episode_ft_count += 1
                        update_progress_bar()
                except:
                    break
        print(f"FT 스레드 종료 완료 - 총 {total_ft_count}개 저장됨")
    
    # 전체 평균 주파수 계산
    total_time = time.time() - start_collection_time
    avg_frequency = total_ft_count / total_time if total_time > 0 else 0
    print(f"FT 데이터 수집 스레드 종료: 총 {total_ft_count}개 데이터, 평균 {avg_frequency:.2f} Hz")

print("카메라와 센서 초기화 중...")

# Realsense D455i (camera1, handeye)
handeye_cam = None
try:
    handeye_cam = RSCapture(
        name='wrist_1',
        serial_number='241122306040',
        dim=(1280, 800), # 원본 해상도 유지
        fps=30,
        depth=False
    )
except Exception as e:
    print(f"[경고] Handeye 카메라 초기화 실패, 더미 프레임으로 대체합니다: {e}")

# DJI Action Cam (camera2, pose_tracking)
dji_cam = None
try:
    dji_cam = DJICapture(
        name='pose_tracking_cam',
        device=args.dji_device,
        dim=(1280, 720), # Calibration 해상도(720p)에 맞춰 형상 보존
        fps=30,
        zero_config=False # 명시적으로 720p MJPG 설정을 적용하도록 수정
    )
except Exception as e:
    print(f"\n[❌ ERROR] DJI Action Cam 초기화 실패: {e}")
    print("DJI 카메라가 연결되지 않았거나 다른 프로그램에서 사용 중일 수 있습니다.")
    print("더미 프레임으로 계속 진행합니다.\n")

# 추가 Realsense 카메라 (camera3, additional) - 옵션
additional_cam = None
if use_additional_cam:
    # 시리얼 번호는 실제 카메라에 맞게 변경 필요
    additional_cam = RSCapture(
        name='additional_camera',
        serial_number='427622273372',
        dim=(640, 480),
        fps=30, # 90fps는 버스 대역폭 부하가 너무 큼
        depth=False
    )

# ====== FT 센서 시작 ======
# ft_reader = FT300Reader('/dev/ttyUSB0', 19200)  # 기존 시리얼 방식
ft_reader = AidinFTSensorUDP('172.27.190.4', 8999)  # 새로운 UDP 방식
ft_reader.start()

# FT 센서 연결 확인 로직 (Dummy 모드 전환 지원)
print("FT 센서 연결 테스트 중...")
try:
    # 1초 동안 데이터를 기다려봄
    ft_reader.get_frame(timeout=1.0)
    print("FT 센서 연결 성공!")
    ft_dummy_mode = False
except queue.Empty:
    print("\n" + "!"*40)
    print("[⚠ WARNING] FT 센서로부터 데이터를 수신할 수 없습니다.")
    print("!"*40)
    user_choice = input("FT 연결이 안됐는데, dummy로 저장할까요? (y/n): ").strip().lower()
    if user_choice == 'y':
        print(">> FT Dummy 모드로 진행합니다. (FT 데이터는 0으로 저장됨)")
        ft_dummy_mode = True
    else:
        print(">> 프로그램을 종료합니다.")
        sys.exit(1)

# IMU 설정
imu_pipe = None
try:
    imu_pipe = rs.pipeline()
    imu_cfg = rs.config()
    imu_cfg.enable_stream(rs.stream.accel)
    imu_cfg.enable_stream(rs.stream.gyro)
    imu_pipe.start(imu_cfg)
    print("IMU 초기화 완료")
except Exception as e:
    print(f"\n[❌ ERROR] RealSense IMU 초기화 실패: {e}")
    imu_pipe = None

# 중력 보상기 베이스라인 캘리브레이션
if not ft_dummy_mode:
    if imu_pipe is not None:
        try:
            gravity_compensator.calibrate_baseline(imu_pipe, ft_reader, warmup_sec=5.0)
        except Exception as e:
            print(f"[❌ ERROR] 베이스라인 캘리브레이션 실패: {e}")
            user_choice = input("캘리브레이션 실패. Dummy 모드로 전환하시겠습니까? (y/n): ").strip().lower()
            if user_choice == 'y':
                ft_dummy_mode = True
            else:
                sys.exit(1)
    else:
        print("[경고] IMU가 없어 베이스라인 캘리브레이션을 건너뜁니다.")
else:
    print("[INFO] FT Dummy 모드이므로 베이스라인 캘리브레이션을 건너뜁니다.")

frame_count = 0
start_time = None
total_image_count = 0
program_start_time = time.time()

# 주파수 조절을 위한 변수
last_acq_saved = {'handeye': 0, 'pose_tracking': 0, 'additional_cam': 0}

# 더미 디스플레이 프레임
DUMMY_DISP_640x480 = np.zeros((480, 640, 3), dtype=np.uint8)
DUMMY_DISP_640x360 = np.zeros((360, 640, 3), dtype=np.uint8)

while True:
    # 1. 카메라 프레임 읽기 (Threaded - Non-blocking)
    res1, res2, res3 = None, None, None

    if handeye_cam is not None:
        ok1, res1 = handeye_cam.read() # (원본, 디스플레이, 획득시간)
    if dji_cam is not None:
        ok2, res2 = dji_cam.read()
    if use_additional_cam and additional_cam is not None:
        ok3, res3 = additional_cam.read()

    # 2. 화면 표시 (Composite Display - 극강의 응답성)
    disp1 = res1[1] if res1 else DUMMY_DISP_640x480
    disp2 = res2[1] if res2 else DUMMY_DISP_640x360
    
    combined = np.vstack([disp1, disp2])
    cv2.imshow("RAW Capture - q:exit, s:save, x:stop", combined)
    
    key = cv2.waitKey(1) & 0xFF

    # 3. 제어 명령 처리
    if key == ord('s') and not saving_images:
        saving_images = True
        start_new_episode()
        last_acq_saved = {'handeye': 0, 'pose_tracking': 0, 'additional_cam': 0}
        
        while not image_queue.empty():
            try: image_queue.get_nowait(); image_queue.task_done()
            except: break
        
        stop_image_save_thread = False
        image_save_thread = threading.Thread(target=image_save_thread_func, daemon=True)
        image_save_thread.start()
        
        stop_ft_thread = False
        ft_thread = threading.Thread(target=ft_collection_thread, args=(ft_reader, imu_pipe, gravity_compensator), daemon=True)
        ft_thread.start()
        print(f"독립적 Raw Capture 시작 (Episode {episode_num})")

    elif (key == ord('x') or key == ord('q')) and saving_images:
        print("데이터 저장 중지 요청... (큐 처리 중)")
        saving_images = False # 1. 새로운 캡처 중단
        
        # 2. 이미지 저장 큐가 비워질 때까지 대기
        print(f"이미지 큐 처리 대기 중... (남은 수: {image_queue.qsize()})")
        image_queue.join()
        print("모든 이미지 저장 완료.")

        stop_ft_thread = True
        if ft_thread is not None: ft_thread.join(timeout=2.0)
        
        stop_image_save_thread = True
        if image_save_thread is not None: image_save_thread.join(timeout=2.0)
        
        print_episode_summary()
        if episode_progress_bar is not None: episode_progress_bar.close()
        if ft_csv_file is not None: ft_csv_file.close()
        episode_num = find_last_episode_number() + 1
        if key == ord('q'): break

    elif key == ord('q'):
        break

    # 4. 개별 프레임 독립 저장 로직 (모든 고유 프레임 캡처)
    if saving_images:
        # Handeye
        if res1 and res1[0] is not None and res1[2] > last_acq_saved['handeye']:
            ts_str = format_timestamp_from_float(res1[2])
            image_queue.put(('handeye', ts_str, res1[0].copy()))
            last_acq_saved['handeye'] = res1[2]
            with capture_count_lock: image_capture_count += 1
            total_image_count += 1

        # DJI (Pose Tracking)
        if res2 and res2[0] is not None and res2[2] > last_acq_saved['pose_tracking']:
            ts_str = format_timestamp_from_float(res2[2])
            image_queue.put(('pose_tracking', ts_str, res2[0].copy()))
            last_acq_saved['pose_tracking'] = res2[2]
            with capture_count_lock: image_capture_count += 1

        # Additional
        if use_additional_cam and res3 and res3[0] is not None and res3[2] > last_acq_saved['additional_cam']:
            ts_str = format_timestamp_from_float(res3[2])
            image_queue.put(('additional_cam', ts_str, res3[0].copy()))
            last_acq_saved['additional_cam'] = res3[2]
            with capture_count_lock: image_capture_count += 1

    # 루프 과열 방지를 위해 아주 미세한 대기
    # cv2.waitKey(1)이 이미 1ms 대기를 수행하므로 추가 대기는 불필요할 수 있음
    
    # # 루프 실행 시간 모니터링 (필요 시 경고)
    # loop_time = time.time() - loop_start
    # if loop_time > FRAME_INTERVAL * 0.8 and saving_images:  # 목표 간격의 80% 이상 소요되면 경고
    #     print(f"경고: 루프 실행 시간이 김: {loop_time*1000:.1f}ms (목표 간격: {FRAME_INTERVAL*1000:.1f}ms)")

# 종료 시 모든 카메라 및 윈도우 해제
stop_image_save_thread = True
stop_ft_thread = True

# 진행 상황 표시 정리
if episode_progress_bar is not None:
    episode_progress_bar.close()
    episode_progress_bar = None

if image_save_thread is not None:
    image_save_thread.join(timeout=5.0)
    
if ft_thread is not None:
    ft_thread.join(timeout=2.0)
    
if ft_csv_file is not None:
    ft_csv_file.close()
    
if handeye_cam is not None:
    handeye_cam.close()     # Realsense 카메라 닫기
dji_cam.close()        # DJI 카메라 닫기
if 'additional_cam' in globals() and additional_cam is not None:
    additional_cam.close()  # 추가 Realsense 카메라 닫기
ft_reader.stop()        # FT 센서 닫기
imu_pipe.stop()         # IMU 파이프라인 닫기
cv2.destroyAllWindows()