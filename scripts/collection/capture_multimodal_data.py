
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import cv2
import os
from datetime import datetime
import time
import numpy as np
import queue
import json
import threading
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, lfilter
from collections import deque
import pyrealsense2 as rs
from ahrs.filters import Madgwick
from tqdm import trange, tqdm
import argparse
import sys

from utils.rs_capture import RSCapture, AzureImageCapture
from utils.ft_capture import FT300Reader, AidinFTSensorUDP
from utils.gravity_compensation_utils import GravityCompensator

# 설정된 디렉토리 경로
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default='data', help='저장할 base 폴더명')
args = parser.parse_args()
base_save_path = args.data_path
saving_images = False
episode_num = 1
handeye_save_path = None
pose_tracking_save_path = None
additional_cam_save_path = None  # 추가 카메라 경로
ft_data_save_path = None

# 스레드 제어 변수
stop_ft_thread = False
stop_image_save_thread = False
ft_thread = None
image_save_thread = None
ft_csv_file = None

# 이미지 저장 큐
image_queue = queue.Queue(maxsize=300)

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
    global episode_start_time, episode_image_count, episode_ft_count
    
    if episode_start_time is None:
        return
    
    total_time = time.time() - episode_start_time
    image_hz = episode_image_count / total_time if total_time > 0 else 0
    ft_hz = episode_ft_count / total_time if total_time > 0 else 0
    
    print("\n" + "="*60)
    print(f"📊 Episode {episode_num-1} 완료!")
    print("="*60)
    print(f"⏱️  총 수집 시간: {total_time:.1f}초")
    print(f"📸 이미지: {episode_image_count}개 ({image_hz:.1f} Hz)")
    print(f"🔧 FT 데이터: {episode_ft_count}개 ({ft_hz:.1f} Hz)")
    print(f"📁 저장 위치: {base_save_path}/episode_{episode_num-1}")
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
    episode_progress_bar = tqdm(total=1800, desc="데이터 수집 중", 
                               bar_format='{desc} |{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]', 
                               ncols=80, leave=True)
    
    # date_str = datetime.now().strftime("%m%d")
    # episode_dir = os.path.join(base_save_path, date_str, f'episode_{episode_num}')
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
    additional_cam_save_path = os.path.join(episode_dir, 'images', 'additional_cam')  # 추가 카메라 경로
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

def image_save_thread_func():
    """이미지를 별도 스레드에서 저장하는 함수"""
    global stop_image_save_thread, image_save_count, episode_image_count
    
    print("이미지 저장 스레드 시작")
    total_save_count = 0
    start_time = time.time()
    last_debug_time = time.time()
    
    # 간단한 JPEG 저장 설정
    jpeg_params_for_handeye = [cv2.IMWRITE_JPEG_QUALITY, 90]
    jpeg_params_for_pose = [cv2.IMWRITE_JPEG_QUALITY, 100]
    
    while not stop_image_save_thread:
        # 디버그 정보 주기적 출력
        current_time = time.time()
        if current_time - last_debug_time > 5.0:  # 5초마다
            print(f"이미지 저장 스레드: 활성 상태, 저장 {total_save_count}개, 큐 크기 {image_queue.qsize()}")
            last_debug_time = current_time
        
        try:
            # 큐에서 이미지 가져오기
            item = image_queue.get(block=True, timeout=0.1)
            
            if item is None:
                image_queue.task_done()
                break
                
            timestamp, frame1, frame2, frame3 = item  # 세 개의 프레임 언패킹
            
            # 디렉토리 존재 확인 및 생성
            if not os.path.exists(handeye_save_path):
                os.makedirs(handeye_save_path, exist_ok=True)
            if not os.path.exists(pose_tracking_save_path):
                os.makedirs(pose_tracking_save_path, exist_ok=True)
            if not os.path.exists(additional_cam_save_path):
                os.makedirs(additional_cam_save_path, exist_ok=True)
            
            # 파일 경로 생성 (jpg 사용)
            handeye_path = os.path.join(handeye_save_path, f"{timestamp}.jpg")
            pose_path = os.path.join(pose_tracking_save_path, f"{timestamp}.jpg")
            additional_path = os.path.join(additional_cam_save_path, f"{timestamp}.jpg")  # 추가 카메라 경로

            # 이미지 저장 시도
            try:
                cv2.imwrite(handeye_path, frame1, jpeg_params_for_handeye)
            except Exception as e:
                print(f"handeye 이미지 저장 실패: {e}")
            
            try:
                cv2.imwrite(pose_path, frame2, jpeg_params_for_pose)
            except Exception as e:
                print(f"pose 이미지 저장 실패: {e}")
                
            # 추가 카메라 이미지 저장
            try:
                cv2.imwrite(additional_path, frame3, jpeg_params_for_handeye)
            except Exception as e:
                print(f"additional 이미지 저장 실패: {e}")
            
            # 카운터 증가
            with capture_count_lock:
                image_save_count += 1
            
            # 에피소드 진행 상황 업데이트
            with progress_lock:
                episode_image_count += 1
                update_progress_bar()
            
            total_save_count += 1
            
            # 큐 작업 완료 표시
            image_queue.task_done()
            
            
        except queue.Empty:
            # 타임아웃 - 정상, 계속 진행
            continue
        except Exception as e:
            print(f"심각한 오류 발생: {str(e)}")
            # 치명적인 오류 발생 시에도 계속 진행 시도
            time.sleep(0.1)  # 약간 대기
    
    elapsed = time.time() - start_time
    avg_rate = total_save_count / elapsed if elapsed > 0 else 0
    print(f"이미지 저장 스레드 종료: 총 {total_save_count}개 저장, 평균 {avg_rate:.2f} Hz")

def ft_collection_thread(ft_reader, imu_pipe, gravity_compensator):
    """FT 데이터를 독립적으로 수집하는 스레드 함수"""
    global stop_ft_thread, ft_csv_file, ft_count, episode_ft_count
    
    print("FT 데이터 수집 스레드 시작 (목표: 200Hz)")
    
    # FT 센서 버퍼 비우기 - 이전 데이터 제거
    print("FT 센서 버퍼 비우기 중...")
    for _ in range(50):  # 충분히 많은 데이터를 버림
        try:
            ft_reader.get_frame(timeout=0.001)
        except:
            break
    
    total_ft_count = 0
    start_collection_time = time.time()
    last_debug_time = time.time()
    debug_interval = 5.0  # 5초마다 디버그 정보 출력
    
    try:
        while not stop_ft_thread:
            cycle_start = time.time()
            
            # IMU 업데이트
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
handeye_cam = RSCapture(
    name='wrist_1',
    serial_number='339522300864',
    dim=(1280, 800),
    fps=30,
    depth=False
)

# Azure Kinect (camera2, pose_tracking)
azure_cam = AzureImageCapture()
azure_cam = azure_cam.start()

# 추가 Realsense 카메라 (camera3, additional)
# 시리얼 번호는 실제 카메라에 맞게 변경 필요
additional_cam = RSCapture(
    name='additional_camera',
    serial_number='130322271079',  # 130322271079, 427622273372
    dim=(640, 480) ,  # 필요에 따라 해상도 조정
    fps=90,
    depth=False
)

# ====== FT 센서 시작 ======
# ft_reader = FT300Reader('/dev/ttyUSB0', 19200)  # 기존 시리얼 방식
ft_reader = AidinFTSensorUDP('172.27.190.4', 8890)  # 새로운 UDP 방식
ft_reader.start()

# IMU 설정
imu_pipe = rs.pipeline()
imu_cfg = rs.config()
imu_cfg.enable_stream(rs.stream.accel)
imu_cfg.enable_stream(rs.stream.gyro)
imu_pipe.start(imu_cfg)
print("IMU 초기화 완료")

# 중력 보상기 베이스라인 캘리브레이션
gravity_compensator.calibrate_baseline(imu_pipe, ft_reader, warmup_sec=5.0)

frame_count = 0
start_time = None
total_image_count = 0
program_start_time = time.time()

print("초기화 완료. s를 눌러 이미지 저장을 시작하세요, x를 눌러 중지하세요, 종료하려면 q를 누르세요.")

while True:
    loop_start = time.time()
    
    # 세 카메라에서 프레임 읽기
    ret1, frame1 = handeye_cam.read()  # Realsense D455i
    ret2, frame2 = azure_cam.read()    # Azure Kinect
    ret3, frame3 = additional_cam.read()  # 추가 Realsense 카메라

    if not ret1 or not ret2 or not ret3:
        print("카메라 프레임을 읽을 수 없습니다.")
        break

    # 각 카메라 영상을 별도의 창으로 표시
    resize_frame1 = cv2.resize(frame1, (0, 0), fx=0.5, fy=0.5)
    resize_frame2 = cv2.resize(frame2, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow("Camera 1 (Handeye)", resize_frame1)
    cv2.imshow("Camera 2 (Pose Tracking)", resize_frame2)
    # cv2.imshow("Camera 3 (Additional)", frame3)  # 추가 카메라 화면
    
    # 키 입력
    key = cv2.waitKey(1) & 0xFF

    # s를 눌러 이미지 저장 시작
    if key == ord('s') and not saving_images:
        saving_images = True
        start_new_episode()
        start_time = time.time()
        frame_count = 0
        
        # 이미지 저장 변수 초기화
        with capture_count_lock:
            image_capture_count = 0
            image_save_count = 0
        
        # 이미지 큐 비우기
        while not image_queue.empty():
            try:
                image_queue.get_nowait()
                image_queue.task_done()
            except:
                break
        
        # 이미지 저장 스레드 시작 (하나만 사용)
        stop_image_save_thread = False
        image_save_thread = threading.Thread(
            target=image_save_thread_func,
            daemon=True,
            name="ImageSaver"
        )
        image_save_thread.start()
        
        # FT 데이터 수집 스레드 시작
        stop_ft_thread = False
        
        # FT 센서 버퍼 완전 비우기 - s 키를 누른 시점의 데이터부터 시작하기 위해
        print("FT 센서 버퍼 완전 비우기 중...")
        ft_buffer_cleared = False
        while not ft_buffer_cleared:
            try:
                ft_reader.get_frame(timeout=0.001)
            except queue.Empty:
                ft_buffer_cleared = True
                break
            except Exception:
                ft_buffer_cleared = True
                break
        print("FT 센서 버퍼 비우기 완료")
        
        ft_thread = threading.Thread(
            target=ft_collection_thread,
            args=(ft_reader, imu_pipe, gravity_compensator),
            daemon=True,
            name="FTCollector"
        )
        ft_thread.start()
        
        print(f"이미지 및 FT 데이터 저장 시작 (Episode {episode_num})")
        

    # x를 눌러 이미지 저장 중지 및 에피소드 번호 증가
    elif key == ord('x') and saving_images:
        print("데이터 저장 중지 요청... (큐에 있는 데이터 처리 중)")
        
        # 1단계: 새로운 데이터 캡처 중단 (saving_images는 아직 True 유지)
        # 2단계: 이미지 저장 스레드가 큐를 비울 때까지 대기
        print("이미지 큐 처리 대기 중...")
        image_queue.join()  # 큐가 완전히 비워질 때까지 대기
        
        # 3단계: 이제 saving_images를 False로 설정
        saving_images = False
        print("모든 이미지 저장 완료")
        
        # 4단계: FT 스레드 종료
        print("FT 스레드 종료 대기 중...")
        stop_ft_thread = True
        
        # FT 스레드가 완전히 종료될 때까지 충분히 대기
        if ft_thread is not None:
            ft_thread.join(timeout=10.0)
            if ft_thread.is_alive():
                print("경고: FT 스레드가 완전히 종료되지 않았습니다.")
            ft_thread = None
        
        # 5단계: 이미지 저장 스레드 종료
        stop_image_save_thread = True
        if image_save_thread is not None:
            image_save_thread.join(timeout=5.0)
            image_save_thread = None
        
        # 에피소드 완료 통계 출력
        print_episode_summary()
        
        # 진행 상황 표시 정리
        if episode_progress_bar is not None:
            episode_progress_bar.close()
            episode_progress_bar = None
        
        if ft_csv_file is not None:
            ft_csv_file.close()
            ft_csv_file = None
            
        print(f"이미지 및 FT 데이터 저장 종료 (Episode {episode_num})")
        # 다음 에피소드 번호 설정
        episode_num = find_last_episode_number() + 1
        print(f"다음 에피소드 번호: {episode_num}")

    # q를 눌러 종료
    elif key == ord('q'):
        # 진행 상황 표시 정리
        if episode_progress_bar is not None:
            episode_progress_bar.close()
            episode_progress_bar = None
        
        # 이미지 저장 스레드 종료
        stop_image_save_thread = True
        if image_save_thread is not None:
            image_save_thread.join(timeout=5.0)
        
        # FT 데이터 수집 스레드 종료
        stop_ft_thread = True
        if ft_thread is not None:
            ft_thread.join(timeout=2.0)
            
        if ft_csv_file is not None:
            ft_csv_file.close()
            
        # 프로그램 전체 실행 통계 출력
        total_time = time.time() - program_start_time
        avg_image_rate = total_image_count / total_time if total_time > 0 else 0
        print(f"\n프로그램 수행 통계:")
        print(f"총 실행 시간: {total_time:.2f}초")
        print(f"총 이미지 수집: {total_image_count}개 ({avg_image_rate:.2f} Hz 평균)")
        
        print("프로그램 종료")
        break

    if saving_images:
        current_time = time.time()
        
        # 이미지 캡처 시점의 타임스탬프 생성
        image_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        try:
            # 안전하게 이미지 복사
            frame1_copy = frame1.copy() if frame1 is not None else None
            frame2_copy = frame2.copy() if frame2 is not None else None
            frame3_copy = frame3.copy() if frame3 is not None else None  # 추가 카메라
            
            if frame1_copy is not None and frame2_copy is not None and frame3_copy is not None:
                # 큐가 거의 가득 찼을 때 경고
                if image_queue.qsize() > image_queue.maxsize * 0.8:
                    print(f"경고: 이미지 큐가 거의 가득 참 ({image_queue.qsize()}/{image_queue.maxsize})")
                
                # 큐에 넣기 (세 개의 프레임 넣음)
                image_queue.put((image_timestamp, frame1_copy, frame2_copy, frame3_copy), 
                              block=True, timeout=0.5)
                
                # 캡처 카운터 증가
                with capture_count_lock:
                    image_capture_count += 1
                
                frame_count += 1
                image_count += 1
                total_image_count += 1
                
                # 30개마다 상태 출력
                if image_capture_count % 30 == 0:
                    print(f"이미지 큐 상태: 사이즈={image_queue.qsize()}, 캡처={image_capture_count}, 저장={image_save_count}")
                
        except queue.Full:
            print(f"경고: 이미지 큐가 가득 참 - 이미지 저장 건너뜀!")
        except Exception as e:
            print(f"이미지 큐 추가 오류: {str(e)}")

    # 프레임 속도 유지 (필요 시 대기)
    if saving_images and start_time is not None:
        time_to_next_frame = (start_time + frame_count * FRAME_INTERVAL) - time.time()
        if time_to_next_frame > 0:
            time.sleep(time_to_next_frame)  # 다음 프레임까지 대기
    
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
    
handeye_cam.close()     # Realsense 카메라 닫기
azure_cam.stop()        # Azure 카메라 닫기
additional_cam.close()  # 추가 Realsense 카메라 닫기
ft_reader.stop()        # FT 센서 닫기
imu_pipe.stop()         # IMU 파이프라인 닫기
cv2.destroyAllWindows()