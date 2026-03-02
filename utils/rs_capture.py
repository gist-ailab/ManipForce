import os
import threading
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
import pyk4a
from pyk4a import Config, PyK4A
from scipy.signal import butter, filtfilt
from scipy.signal import butter, lfilter
from scipy.spatial.transform import Rotation as R
import time

class IMUCapture:
    def get_device_serial_numbers(self):
        devices = rs.context().devices
        for d in devices:
            print(f"🔹 장치 시리얼 번호: {d.get_info(rs.camera_info.serial_number)}")
        return [d.get_info(rs.camera_info.serial_number) for d in devices]
  
    def __init__(self, name, serial_number, dim=(640, 480), fps=15, depth=False):
        self.name = name
        assert serial_number in self.get_device_serial_numbers()
        self.serial_number = serial_number
        self.depth = depth
        self.pipe = rs.pipeline()
        self.cfg = rs.config()

        self.cfg.enable_device(self.serial_number)
        self.cfg.enable_stream(rs.stream.color, dim[0], dim[1], rs.format.bgr8, fps)
        self.cfg.enable_stream(rs.stream.accel)
        self.cfg.enable_stream(rs.stream.gyro)

        # if self.depth:
        #     self.cfg.enable_stream(rs.stream.depth, dim[0], dim[1], rs.format.z16, fps)
        self.profile = self.pipe.start(self.cfg)

        self.orientation = R.from_quat([0, 0, 0, 1])  # 초기 단위 쿼터니언 (기본 방향)
        self.prev_time = time.time()

        self.accel_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.gravity = np.array([0, 0, -9.81])  # 중력 가속도

        self.gyro_buffer = []
        self.accel_buffer = []
        self.min_samples = 20

        # Create an align object
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def read(self):
        frames = self.pipe.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if not color_frame:
            return False, None
        image = np.asarray(color_frame.get_data())
        if self.depth:
            depth_frame = aligned_frames.get_depth_frame()
            if depth_frame:
                depth_image = np.expand_dims(np.asarray(depth_frame.get_data()), axis=2)
                return True, (image, depth_image)
        return True, image

    def gyro_data(self, gyro):
        return np.asarray([gyro.x, gyro.y, gyro.z])

    def accel_data(self, accel):
        return np.asarray([accel.x, accel.y, accel.z])

    def get_imu_data(self):
        """ IMU 데이터를 읽어 반환 (가속도 및 자이로) """
        try:
            frames = self.pipe.wait_for_frames()
            accel_frame, gyro_frame = None, None

            for frame in frames:
                if frame.is_motion_frame():
                    motion_frame = frame.as_motion_frame()
                    if motion_frame.get_profile().stream_type() == rs.stream.accel:
                        accel_frame = motion_frame
                    elif motion_frame.get_profile().stream_type() == rs.stream.gyro:
                        gyro_frame = motion_frame

            if accel_frame and gyro_frame:
                raw_accel = self.accel_data(accel_frame.get_motion_data())
                raw_gyro = self.gyro_data(gyro_frame.get_motion_data())

                # 보정된 값 적용
                accel = raw_accel - self.gravity
                gyro = raw_gyro - self.gyro_bias

                # 버퍼에 데이터 추가
                self.accel_buffer.append(accel)
                self.gyro_buffer.append(gyro)

                # 버퍼 크기 관리
                if len(self.accel_buffer) > self.min_samples:
                    self.accel_buffer.pop(0)
                if len(self.gyro_buffer) > self.min_samples:
                    self.gyro_buffer.pop(0)

                return accel, gyro

            return None, None
    
        except Exception as e:
            print(f"IMU 데이터 읽기 실패: {e}")
            return None, None

    def butter_lowpass_filter(self, data, cutoff=5, fs=100, order=3):
        """Butterworth Low-pass Filter 적용"""
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)

        # 데이터 길이가 필터링에 필요한 최소 길이보다 작으면 필터링하지 않음
        if len(data) < 12:
            return np.array(data[-1])  # 가장 최근 데이터 반환
        
        return filtfilt(b, a, np.array(data), axis=0)

    def calibrate_gyro(self, duration=5):
        """Gyro 바이어스 보정"""
        gyro_data = []
        print("🔹 Gyro 바이어스 보정 중...")

        start_time = time.time()
        while time.time() - start_time < duration:
            _, gyro = self.get_imu_data()
            if gyro is not None:
                gyro_data.append(gyro)
            time.sleep(0.01)

        self.gyro_bias = np.mean(gyro_data, axis=0)
        print(f"✅ Gyro 바이어스 보정 완료: {self.gyro_bias}")

    def calibrate_accel(self, duration=5):
        """초기 중력 벡터 보정"""
        accel_data = []
        print("🔹 Accel 중력 벡터 보정 중...")

        # realsense 의 imu는 -y 방향으로의 중력이 발생
        

        start_time = time.time()
        while time.time() - start_time < duration:
            accel, _ = self.get_imu_data()
            if accel is not None:
                accel_data.append(accel)
            time.sleep(0.01)

        self.gravity = np.mean(accel_data, axis=0)

        norm_gravity = np.linalg.norm(self.gravity)
        if norm_gravity > 0:
            self.gravity = self.gravity / norm_gravity * 9.81

        print(f"✅ Accel 중력 벡터 보정 완료: {self.gravity}")

    def update_orientation(self, gyro, dt):
        """자이로 데이터를 적분하여 회전(오일러 각) 추정"""
        rotation_vector = gyro * dt  # Δθ = ω * dt

        if np.linalg.norm(rotation_vector) < 1e-6:
            return self.orientation.as_euler('zyx', degrees=True)  # 오일러 각으로 반환

        delta_rotation = R.from_rotvec(rotation_vector)  # 회전 벡터 → 회전 변환
        self.orientation = delta_rotation * self.orientation  # 회전 누적 적용

        return self.orientation.as_euler('zyx', degrees=True)  # 최종 회전 결과 반환 (Yaw, Pitch, Roll)


    def correct_gravity(self, accel):
        """가속도 데이터를 기반으로 중력 방향 보정"""
        g_estimate = accel / np.linalg.norm(accel)  # 가속도 벡터 정규화
        g_real = np.array([0, 0, -1])  # 실제 중력 벡터

        correction_quat, _ = R.align_vectors([g_real], [g_estimate])
        self.orientation = correction_quat * self.orientation

        # **수정된 부분: 직접 쿼터니언 정규화**
        orientation_quat = self.orientation.as_quat()
        self.orientation = R.from_quat(orientation_quat / np.linalg.norm(orientation_quat))

        return self.orientation.as_quat()

    def close(self):
        self.pipe.stop()
        self.cfg.disable_all_streams()

class RSCapture:
    def __init__(self, name, serial_number, dim=(1280, 800), fps=30, depth=False, auto_start=True):
        self.name = name
        self.serial_number = serial_number
        self.depth = depth
        self.dim = dim
        self.fps = fps
        self.pipe = rs.pipeline()
        self.cfg = rs.config()

        self.cfg.enable_device(self.serial_number)
        self.cfg.enable_stream(rs.stream.color, dim[0], dim[1], rs.format.bgr8, fps)
        if self.depth:
            self.cfg.enable_stream(rs.stream.depth, dim[0], dim[1], rs.format.z16, fps)
        
        # Threading
        self.running = False
        self.thread = None
        self.latest_frame = None
        self.latest_display_frame = None
        self.lock = threading.Lock()

        if auto_start:
            self.start()

    def start(self):
        if self.running: return self
        self.profile = self.pipe.start(self.cfg)
        self.sensor = self.profile.get_device().first_depth_sensor()
        self.sensor.set_option(rs.option.enable_auto_exposure, 0)
        self.sensor.set_option(rs.option.exposure, 10000)
        
        align_to = rs.stream.color
        self.align = rs.align(align_to)
        
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        while self.running:
            try:
                frames = self.pipe.wait_for_frames(timeout_ms=1000)
                acq_time = time.time() # 프레임 획득 즉시 시간 기록
                if not frames: continue
                
                aligned_frames = self.align.process(frames)
                color_frame = aligned_frames.get_color_frame()
                if not color_frame: continue
                
                image = np.asarray(color_frame.get_data())
                
                result = image
                if self.depth:
                    depth_frame = aligned_frames.get_depth_frame()
                    if depth_frame:
                        depth_image = np.expand_dims(np.asarray(depth_frame.get_data()), axis=2)
                        result = (image, depth_image)
                
                with self.lock:
                    self.latest_frame = result
                    self.latest_acq_time = acq_time
            except Exception as e:
                if self.running: # 종료 중이 아닐 때만 출력
                    print(f"[RS {self.name}] Error: {e}")
                time.sleep(0.1)

    def read(self):
        with self.lock:
            frame = self.latest_frame
            acq_time = getattr(self, 'latest_acq_time', None)
        
        if frame is None:
            return False, None
            
        # 화면 표시용 리사이즈는 read() 시점에 수행 (Lazy resizing)
        img_to_resize = frame[0] if isinstance(frame, tuple) else frame
        display = cv2.resize(img_to_resize, (640, 400), interpolation=cv2.INTER_NEAREST)
            
        return True, (frame, display, acq_time)

    def close(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.pipe.stop()
        self.cfg.disable_all_streams()
        
class AzureImageCapture:
    def __init__(self):
        self.k4a = None
        self.config = Config(
            color_resolution=pyk4a.ColorResolution.RES_1080P,
            camera_fps=pyk4a.FPS.FPS_30,
            depth_mode=pyk4a.DepthMode.OFF,  # 더 빠른 처리를 위해 binning 적용
            synchronized_images_only=False,  # 동기화 비활성화로 성능 향상
        )
    
    def start(self):
        """Azure Kinect 카메라를 시작합니다."""
        try:
            self.k4a = PyK4A(self.config)
            self.k4a.start()

            # 하드웨어 노출/게인/화이트밸런스 설정 (가능한 경우)
            try:
                set_cc = None
                if hasattr(self.k4a, 'set_color_control'):
                    set_cc = self.k4a.set_color_control
                elif hasattr(self.k4a, 'device') and hasattr(self.k4a.device, 'set_color_control'):
                    set_cc = self.k4a.device.set_color_control
                elif hasattr(self.k4a, '_device') and hasattr(self.k4a._device, 'set_color_control'):
                    set_cc = self.k4a._device.set_color_control
                if set_cc is not None:
                    from pyk4a import ColorControlCommand, ColorControlMode
                    set_cc(ColorControlCommand.EXPOSURE_TIME_ABSOLUTE, ColorControlMode.MANUAL, 200)
                    set_cc(ColorControlCommand.GAIN, ColorControlMode.MANUAL, 16)
                    set_cc(ColorControlCommand.WHITEBALANCE, ColorControlMode.MANUAL, 4500)
                    print("[INFO] Azure exposure set: 200us, gain:16, wb:4500K")
                else:
                    print('[INFO] set_color_control not available; skipping hardware exposure setup.')
            except Exception as e:
                print(f"[INFO] Hardware exposure control unsupported or failed: {e}")
            
            # 처음 몇 프레임은 버림 (안정화)
            for _ in range(30):
                self.k4a.get_capture()
                
            print("Azure Kinect 카메라가 성공적으로 시작되었습니다.")
            return self
            
        except Exception as e:
            print(f"Azure Kinect 시작 실패: {e}")
            return None
        
    def get_intrinsics(self):
        """카메라 내부 파라미터를 가져옵니다. 실제 카메라에서 실패하면 기본값을 사용합니다."""
        if self.k4a is None:
            print("카메라가 연결되지 않아 기본값을 사용합니다.")
            return self.get_default_intrinsics()
            
        calibration = self.k4a.calibration
        intrinsics = calibration.get_camera_calibration(pyk4a.CalibrationType.COLOR)
        
        K_azure = np.array([
            [intrinsics.intrinsics.fx, 0, intrinsics.intrinsics.cx],
            [0, intrinsics.intrinsics.fy, intrinsics.intrinsics.cy],
            [0, 0, 1]
        ], dtype=np.float32)
        
        D_azure = np.array([
            intrinsics.intrinsics.k1, intrinsics.intrinsics.k2,
            intrinsics.intrinsics.p1, intrinsics.intrinsics.p2,
            intrinsics.intrinsics.k3, 0, 0, 0
        ], dtype=np.float32).ravel()
        # 왜곡 벡터 길이 가드
        if D_azure.size not in (4,5,8,12,14):
            D_azure = np.zeros(5, np.float32)
        
        print("Azure Kinect 실제 내부 파라미터:")
        print(f"카메라 행렬:\n{K_azure}")
        print(f"왜곡 계수: {D_azure}")
        
        return K_azure, D_azure

    def get_default_intrinsics(self):
        """Azure Kinect의 1080p 해상도에 대한 기본 내부 파라미터를 반환합니다."""
        # 1920x1080 해상도에 대한 기본값
        fx, fy = 950.0, 950.0  # 초점 거리
        cx, cy = 960.0, 540.0  # 주점 (1920x1080 해상도의 중심)
        
        # 카메라 행렬 구성
        K_azure = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        # 왜곡 계수 (k1, k2, p1, p2, k3, k4, k5, k6)
        D_azure = np.zeros(8)  # 왜곡 없다고 가정
        
        print("Azure Kinect 기본 내부 파라미터:")
        print(f"카메라 행렬:\n{K_azure}")
        print(f"왜곡 계수: {D_azure}")
        
        return K_azure, D_azure

    def read(self):
        """프레임을 읽어옵니다."""
        if self.k4a is None:
            return False, None
            
        try:
            capture = self.k4a.get_capture()
            if capture.color is not None:
                color_image = cv2.cvtColor(capture.color, cv2.COLOR_BGRA2BGR)
                return True, color_image
            return False, None
        except Exception as e:
            print(f"프레임 읽기 실패: {e}")
            return False, None
    
    def stop(self):
        """카메라를 정지합니다."""
        if self.k4a is not None:
            self.k4a.stop()
            self.k4a = None


class DJICapture:
    """
    DJI Action 4 (또는 기타 USB UVC 액션캠)용 간단 캡처 래퍼.
    RealSense / Azure 캡처 클래스와 동일한 read/close API를 제공한다.
    """

    def __init__(
        self,
        name: str = "dji_action_cam",
        device: Optional[str] = None,
        dim: Tuple[int, int] = (1280, 720),
        fps: int = 30,
        backend: int = cv2.CAP_V4L2,
        auto_start: bool = True,
        zero_config: bool = True,
        threaded: bool = True,
    ):
        self.name = name
        self.backend = backend
        self.device = self._resolve_device(device, self.backend)
        self.dim = dim
        self.fps = fps
        self.zero_config = zero_config
        self.threaded = threaded
        self.cap: Optional[cv2.VideoCapture] = None

        # 스레드 관련 변수
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.latest_frame = None
        self.latest_display_frame = None # 화면 출력용 축소 프레임
        self.lock = threading.Lock()
        self.read_count = 0 

        # 저장 제어 상태
        self._saving = False
        self._save_dir: Optional[str] = None
        self._saved_count = 0

        self._fourcc = cv2.VideoWriter_fourcc(*"MJPG")

        if auto_start:
            self.start()

    def _resolve_device(self, device: Optional[str], backend: int) -> str:
        if device:
            return device
        env_device = os.environ.get("DJI_DEVICE")
        if env_device:
            return env_device
        
        # 1. /dev/v4l/by-id 에서 "DJI"가 포함된 장치 찾기
        import glob
        by_id_paths = sorted(glob.glob("/dev/v4l/by-id/*DJI*"))
        
        if by_id_paths:
            target_path = None
            for path in by_id_paths:
                if "index0" in path:
                    target_path = path
                    break
            if target_path is None:
                target_path = by_id_paths[0]
            real_device_path = os.path.realpath(target_path)
            print(f"[DJI] 감지된 DJI 장치 (by-id): {target_path} -> {real_device_path}")
            return real_device_path

        devices = sorted(glob.glob("/dev/video*"), key=lambda x: int(x.split('video')[-1]))
        if not devices:
            raise RuntimeError(
                "연결된 UVC 카메라를 찾을 수 없습니다. DJI 액션캠을 UVC 모드로 전환했는지 확인하세요."
            )
        
        for dev in devices:
            try:
                test_cap = cv2.VideoCapture(dev, backend)
                if test_cap.isOpened():
                    test_cap.release()
                    print(f"[DJI] 장치 감지 및 테스트 성공: {dev}")
                    return dev
            except:
                continue
        
        return devices[0]

    def start(self) -> "DJICapture":
        """VideoCapture를 초기화하고 스레드를 시작."""
        if self.running:
            return self

        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.device, self.backend)
            if not self.cap.isOpened():
                raise RuntimeError(f"DJI 카메라를 열 수 없습니다: {self.device}")

            # 'Fast' 스크립트 특성 반영
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 1.0)

            if not self.zero_config:
                # FOURCC를 먼저 설정해야 해상도/FPS가 올바르게 적용되는 경우가 많음
                self.cap.set(cv2.CAP_PROP_FOURCC, self._fourcc)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.dim[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.dim[1])
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                
                # 실제 적용된 값 확인
                rw = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                rh = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                rfps = self.cap.get(cv2.CAP_PROP_FPS)
                rfcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
                fcc_str = "".join([chr((rfcc >> 8 * i) & 0xFF) for i in range(4)])
                print(f"[DJI] 하드웨어 설정 결과: {rw}x{rh} @ {rfps}FPS (FourCC: {fcc_str})")
            else:
                print(f"[DJI] Zero-Config 모드 (드라이버 기본값 사용)")

        # 스레드 시작
        if self.threaded:
            self.running = True
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
            print(f"[DJI] 캡처 스레드 시작됨 (장치: {self.device})")
        else:
            print(f"[DJI] Non-Threaded 모드 (장치: {self.device})")
        
        return self

    def _update(self):
        """별도 스레드에서 계속 프레임을 읽어 최신 상태 유지
        """
        # 단순 Read가 가장 응답성이 좋음 (Grab 셔플링 제거)
        fps_start_time = time.time()
        fps_counter = 0
        
        try:
            while self.running:
                if not self.running:
                    break
                    
                ret, frame = self.cap.read()
                acq_time = time.time() # 프레임 획득 즉시 시간 기록
                if ret:
                    with self.lock:
                        self.latest_frame = frame
                        self.latest_acq_time = acq_time
                        self.read_count += 1
                    
                    fps_counter += 1
                    now = time.time()
                    if now - fps_start_time >= 2.0:
                        actual_fps = fps_counter / (now - fps_start_time)
                        print(f"[DJI] Capture FPS: {actual_fps:.2f}")
                        fps_start_time = now
                        fps_counter = 0
                else:
                    time.sleep(0.001)
        except cv2.error:
            # 종료 시점에 발생할 수 있는 에러 무시
            pass
        except Exception:
            pass

    def read(self):
        """프레임을 반환."""
        if self.threaded:
            if not self.running:
                return False, None

            with self.lock:
                frame = self.latest_frame
                acq_time = getattr(self, 'latest_acq_time', None)
            
            if frame is None:
                return False, None
                
            # 화면 표시용 리사이즈는 read() 시점에 수행 (Lazy resizing)
            display = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_NEAREST)
                
            return True, (frame, display, acq_time)
        else:
            # Non-threaded: Blocking read
            if self.cap is None: return False, None
            ret, frame = self.cap.read()
            if not ret: return False, None
            acq_time = time.time()
            display = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_NEAREST)
            return True, (frame, display, acq_time)

    def close(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        if self.cap:
            self.cap.release()

    def start_recording(self, save_dir: Optional[str] = None):
        """간단한 로컬 저장을 활성화한다."""
        base_dir = (
            save_dir
            if save_dir
            else os.path.join(os.path.dirname(os.path.abspath(__file__)), "test", self.name)
        )
        os.makedirs(base_dir, exist_ok=True)
        self._save_dir = base_dir
        self._saving = True
        self._saved_count = 0
        print(f"[DJI:{self.name}] 저장 시작 → {self._save_dir}")

    def stop_recording(self):
        """로컬 저장 중지."""
        if self._saving:
            print(f"[DJI:{self.name}] 저장 중지 (총 {self._saved_count}장)")
        self._saving = False
        self._save_dir = None

    def _save_frame(self, frame):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.name}_{self._saved_count:06d}_{ts}.jpg"
        filepath = os.path.join(self._save_dir, filename)
        try:
            cv2.imwrite(filepath, frame)
            self._saved_count += 1
        except Exception as exc:  # pragma: no cover - 단순 로그
            print(f"[DJI:{self.name}] 프레임 저장 실패: {exc}")

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def stop(self):
        """Azure와 인터페이스를 맞추기 위한 별칭."""
        self.close()
        