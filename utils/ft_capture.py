import time
from scipy.spatial.transform import Rotation as R
import serial
import threading
import queue
import struct
import socket
import numpy as np
from collections import deque
            
def modbus_crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF

class FT300Reader:
    def __init__(self, device='/dev/ttyUSB0', baudrate=19200, queue_size=500):
        self.ser = serial.Serial(
            device,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=None,
            inter_byte_timeout=0
        )
        self.q = queue.Queue(maxsize=queue_size)
        self._latest = None            # 마지막으로 받은 (ts, forces, torques)
        self._lock   = threading.Lock()
        
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._read_loop, daemon=True)

    def start(self):
        self.ser.write(b'\xff' * 50)
        time.sleep(0.05)
        slave, func = 9, 16
        addr, nregs, data = 410, 1, 0x0200
        frame = bytes([
            slave, func,
            (addr>>8)&0xFF, addr&0xFF,
            (nregs>>8)&0xFF, nregs&0xFF,
            2,
            (data>>8)&0xFF, data&0xFF
        ])
        crc = modbus_crc16(frame)
        self.ser.write(frame + bytes([crc&0xFF, (crc>>8)&0xFF]))
        self._t.start()

    def stop(self):
        self._stop.set()
        self._t.join(timeout=1.0)
        self.ser.close()

    def _read_loop(self):
        buf = bytearray()
        while not self._stop.is_set():
            data = self.ser.read(self.ser.in_waiting or 1)
            if data:
                buf.extend(data)
            while len(buf) >= 16:
                idx = buf.find(b'\x20\x4E')
                if idx < 0:
                    buf.clear()
                    break
                if len(buf) - idx < 16:
                    break
                frame = buf[idx:idx+16]
                del buf[:idx+16]
                if modbus_crc16(frame[:14]) != (frame[14] + (frame[15]<<8)):
                    continue
                regs = [struct.unpack_from('<h', frame, 2+2*i)[0] for i in range(6)]
                forces  = [regs[i]/100.0   for i in range(3)]
                torques = [regs[i]/1000.0  for i in range(3,6)]
                ts = time.time()
                # 2‑라인만 추가
                with self._lock:
                    self._latest = (ts, forces, torques)
                # 큐에는 그대로 넣어‑두면, 필요할 때 과거 프레임도 가져다 쓸 수 있음
                try:
                    self.q.put_nowait((ts, forces, torques))
                except queue.Full:
                    self.q.get_nowait()          # 가장 오래된 것 버리고
                    self.q.put_nowait((ts, forces, torques))


    def get_frame(self, timeout=None):
        return self.q.get(timeout=timeout)

    def read(self, timeout=None):
        _, forces, torques = self.get_frame(timeout=timeout)
        return forces, torques

    def read_latest(self, timeout=0.0):
        """
        가장 최근에 들어온 (ts, forces, torques) 튜플을 돌려준다.
        * timeout>0 인 경우, 지정 시간 동안 아무 프레임도 안 오면 queue.Empty 예외 발생
        * FT 프레임이 한 번도 들어오지 않았으면 역시 queue.Empty
        """
        start = time.time()
        while True:
            with self._lock:
                if self._latest is not None:
                    return self._latest
            if timeout and (time.time() - start) > timeout:
                raise queue.Empty
            time.sleep(0.001)   # 1 ms 폴링


class AidinFTSensorUDP:
    """Aidin FT 센서 UDP 통신 클래스 - FT300Reader와 호환 인터페이스"""
    
    def __init__(self, sensor_ip='172.27.190.4', sensor_port=8890, queue_size=500):
        self.sensor_ip = sensor_ip
        self.sensor_port = sensor_port
        
        self.socket = None
        self.connected = False
        self.data_buffer = deque(maxlen=1000)
        self.q = queue.Queue(maxsize=queue_size)  # FT300Reader와 호환
        self._latest = None
        self._lock = threading.Lock()
        self.running = False
        self.thread = None
        
        # 명령어
        self.CMD_START = bytes([0x00, 0x03, 0x02])
        self.CMD_STOP = bytes([0x00, 0x03, 0x03])
        self.CMD_BIAS = bytes([0x00, 0x03, 0x04])
        self.DATA_SIZE = 52
        
        # Hz 측정용
        self.packet_count = 0
        self.start_time = None
        
    def start(self):
        """FT300Reader와 호환 - 센서 시작"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # 소켓 버퍼 크기 늘리기 (1MB)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
            self.socket.settimeout(5.0)
            
            # 0 대신 sensor_port(8890)로 바인딩하여 특정 포트로 들어오는 데이터 수신
            try:
                self.socket.bind(('', 0))
                print(f"[Aidin FT Debug] 로컬 포트 {self.socket.getsockname()[1]}에 바인딩 성공")
            except Exception as e:
                print(f"[Aidin FT Debug] 바인딩 실패 ({e})")
                self.socket.bind(('', 0))
                
            self.connected = True
            print(f"Aidin FT 센서 연결됨: {self.sensor_ip}:{self.sensor_port}")
            
            # 혹시 모를 스트리밍 강제 종료 및 안정화
            self.stop_transmit()
            time.sleep(0.2)
            
            # 바이어스 모드 실행
            self.bias_mode()
            time.sleep(1.0)
            
            # 전송 모드 시작
            if self.start_transmit():
                self.running = True
                self.thread = threading.Thread(target=self._data_receiver_thread, daemon=True)
                self.thread.start()
                print("Aidin FT 센서 스트리밍 시작")
            else:
                print("Aidin FT 센서 시작 명령 전송 실패")
                
        except Exception as e:
            print(f"Aidin FT 센서 시작 실패: {e}")
            self.connected = False
    
    def stop(self):
        """FT300Reader와 호환 - 센서 정지"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.connected:
            self.stop_transmit()
        if self.socket:
            self.socket.close()
        self.connected = False
        print("Aidin FT 센서 정지됨")
    
    def send_command(self, command):
        """명령어 전송"""
        if not self.connected or not self.socket:
            return False
        try:
            self.socket.sendto(command, (self.sensor_ip, self.sensor_port))
            time.sleep(0.1)
            return True
        except Exception as e:
            print(f"Aidin FT 명령어 전송 실패: {e}")
            return False

    def start_transmit(self):
        """전송 모드 시작"""
        return self.send_command(self.CMD_START)
    
    def stop_transmit(self):
        """전송 모드 정지"""
        return self.send_command(self.CMD_STOP)
    
    def bias_mode(self):
        """바이어스 모드"""
        return self.send_command(self.CMD_BIAS)
    
    def parse_data(self, raw_data):
        """52바이트 raw 데이터를 파싱"""
        if len(raw_data) != self.DATA_SIZE:
            return None
        try:
            values = struct.unpack('>13f', raw_data)
            return {
                'timestamp': time.time(),
                'fx': values[0], 'fy': values[1], 'fz': values[2],
                'tx': values[3], 'ty': values[4], 'tz': values[5],
                'ax': values[6], 'ay': values[7], 'az': values[8],
                'gx': values[9], 'gy': values[10], 'gz': values[11],
                'temp': values[12]
            }
        except:
            return None
    
    def _data_receiver_thread(self):
        """데이터 수신 스레드"""
        self.socket.settimeout(2.0)
        self.start_time = time.time()
        self.packet_count = 0
        
        print(f"[Aidin FT Debug] 수신 스레드 시작, 타켓: {self.sensor_ip}")
        
        first_packet = False
        while self.running and self.connected:
            try:
                data, addr = self.socket.recvfrom(1024)
                
                if not first_packet:
                    print(f"[Aidin FT Debug] 첫 번째 데이터 패킷 수신됨! 크기: {len(data)}")
                    first_packet = True
                
                if len(data) == self.DATA_SIZE:
                    parsed_data = self.parse_data(data)
                    if parsed_data:
                        # FT 데이터 추출
                        forces = [parsed_data['fx'], parsed_data['fy'], parsed_data['fz']]
                        torques = [parsed_data['tx'], parsed_data['ty'], parsed_data['tz']]
                        ts = parsed_data['timestamp']
                        
                        with self._lock:
                            self.data_buffer.append(parsed_data)
                            self._latest = (ts, forces, torques)  # FT300Reader 호환
                        
                        # 큐에 넣기 (FT300Reader 호환)
                        try:
                            self.q.put_nowait((ts, forces, torques))
                        except queue.Full:
                            try:
                                self.q.get_nowait()  # 가장 오래된 것 버리고
                                self.q.put_nowait((ts, forces, torques))
                            except queue.Empty:
                                pass
                        
                        self.packet_count += 1
                else:
                    if len(data) > 0:
                        print(f"[Aidin FT Debug] 잘못된 데이터 크기: {len(data)} (기대값: {self.DATA_SIZE})")
                
            except socket.timeout:
                if not first_packet:
                    print(f"[Aidin FT Debug] 데이터 수신 타임아웃 (2초 동안 데이터 없음)")
                continue
            except Exception as e:
                if self.running:  # 정상적인 종료가 아닌 경우만 출력
                    print(f"[Aidin FT Debug] 수신 오류: {e}")
                break
        print(f"[Aidin FT Debug] 수신 스레드 종료. 총 패킷: {self.packet_count}")
    
    def get_frame(self, timeout=None):
        """FT300Reader와 호환 - (timestamp, forces, torques) 반환"""
        return self.q.get(timeout=timeout)
    
    def get_frame_with_imu(self, timeout=None):
        """FT + IMU 데이터 모두 반환 - (timestamp, forces, torques, accel, gyro)"""
        start = time.time()
        while True:
            with self._lock:
                if self.data_buffer:
                    latest_data = self.data_buffer[-1]
                    forces = [latest_data['fx'], latest_data['fy'], latest_data['fz']]
                    torques = [latest_data['tx'], latest_data['ty'], latest_data['tz']]
                    accel = [latest_data['ax'], latest_data['ay'], latest_data['az']]
                    gyro = [latest_data['gx'], latest_data['gy'], latest_data['gz']]
                    return (latest_data['timestamp'], forces, torques, accel, gyro)
            if timeout and (time.time() - start) > timeout:
                raise queue.Empty
            time.sleep(0.001)
    
    def get_imu_data(self, timeout=None):
        """IMU 데이터만 반환 - (timestamp, accel, gyro)"""
        start = time.time()
        while True:
            with self._lock:
                if self.data_buffer:
                    latest_data = self.data_buffer[-1]
                    accel = [latest_data['ax'], latest_data['ay'], latest_data['az']]
                    gyro = [latest_data['gx'], latest_data['gy'], latest_data['gz']]
                    return (latest_data['timestamp'], accel, gyro)
            if timeout and (time.time() - start) > timeout:
                raise queue.Empty
            time.sleep(0.001)
    
    def read(self, timeout=None):
        """FT300Reader와 호환 - (forces, torques) 반환"""
        _, forces, torques = self.get_frame(timeout=timeout)
        return forces, torques
    
    def read_latest(self, timeout=0.0):
        """FT300Reader와 호환 - 최신 데이터 반환"""
        start = time.time()
        while True:
            with self._lock:
                if self._latest is not None:
                    return self._latest
            if timeout and (time.time() - start) > timeout:
                raise queue.Empty
            time.sleep(0.001)
    
    def get_hz(self):
        """현재 수신 주파수 계산"""
        if self.start_time is None or self.packet_count == 0:
            return 0.0
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.packet_count / elapsed
        return 0.0