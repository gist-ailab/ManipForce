
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import socket
import struct
import time
import threading
from collections import deque
import numpy as np

class AidinFTSensorUDP:
    def __init__(self, sensor_ip='172.27.190.4', sensor_port=8890, buffer_size=1000):
        self.sensor_ip = sensor_ip
        self.sensor_port = sensor_port
        self.buffer_size = buffer_size
        
        self.socket = None
        self.connected = False
        self.data_buffer = deque(maxlen=buffer_size)
        self.lock = threading.Lock()
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
        
    def connect(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # 소켓 버퍼 크기 늘리기 (1MB)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
            self.socket.settimeout(5.0)
            self.socket.bind(('', 0))
            self.connected = True
            print(f"FT 센서 연결됨: {self.sensor_ip}:{self.sensor_port}")
            return True
        except Exception as e:
            print(f"연결 실패: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        self.stop_streaming()
        if self.socket:
            self.socket.close()
        self.connected = False
    
    def send_command(self, command):
        if not self.connected:
            return False
        try:
            self.socket.sendto(command, (self.sensor_ip, self.sensor_port))
            time.sleep(0.1)
            return True
        except:
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
        self.socket.settimeout(2.0)
        self.start_time = time.time()
        self.packet_count = 0
        
        while self.running and self.connected:
            try:
                data, addr = self.socket.recvfrom(1024)
                
                if len(data) == self.DATA_SIZE:
                    parsed_data = self.parse_data(data)
                    if parsed_data:
                        with self.lock:
                            self.data_buffer.append(parsed_data)
                        self.packet_count += 1
                
            except socket.timeout:
                continue
            except:
                break
    
    def start_streaming(self):
        if not self.connected:
            return False
        # 혹시 모를 스트리밍 강제 종료 및 안정화
        self.stop_transmit()
        time.sleep(0.2)
        if not self.start_transmit():
            return False
        self.running = True
        self.thread = threading.Thread(target=self._data_receiver_thread, daemon=True)
        self.thread.start()
        return True
    
    def stop_streaming(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.stop_transmit()
    
    def get_latest_data(self):
        with self.lock:
            if self.data_buffer:
                return self.data_buffer[-1].copy()
        return None
    
    def get_force_torque(self):
        data = self.get_latest_data()
        if data:
            return np.array([
                data['fx'], data['fy'], data['fz'],
                data['tx'], data['ty'], data['tz']
            ])
        return None
    
    def get_hz(self):
        """현재 수신 주파수 계산"""
        if self.start_time is None or self.packet_count == 0:
            return 0.0
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            return self.packet_count / elapsed
        return 0.0
    
    def print_current_ft(self):
        """현재 FT 값을 간단히 출력"""
        ft = self.get_force_torque()
        hz = self.get_hz()
        if ft is not None:
            print(f"FT: {ft.round(2)} | {hz:.1f} Hz")
        else:
            print(f"FT: 데이터 없음 | {hz:.1f} Hz")
    
    def clear_buffer(self):
        with self.lock:
            self.data_buffer.clear()
    
    def __enter__(self):
        """Context manager 진입"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.disconnect()


# 사용 예제
if __name__ == "__main__":
    print("=== FT 센서 테스트 ===")
    
    # FT 센서 테스트
    with AidinFTSensorUDP(
        sensor_ip='172.27.190.4', 
        sensor_port=8890 #8890, 50000
    ) as ft_sensor:
        if ft_sensor.connected:
            print("연결 성공! 스트리밍 시작...")
            ft_sensor.bias_mode()
            time.sleep(1)
            
            if ft_sensor.start_streaming():
                try:
                    print("FT 데이터 수신 중... (Ctrl+C로 종료)")
                    print("-" * 60)
                    
                    for i in range(200):
                        time.sleep(0.1)
                        
                        if i % 10 == 0:  # 1초마다 출력
                            print(f"[{i//10:2d}s] ", end="")
                            ft_sensor.print_current_ft()
                    
                    # 최종 Hz 출력
                    final_hz = ft_sensor.get_hz()
                    print(f"\n평균 수신 주파수: {final_hz:.1f} Hz")
                
                except KeyboardInterrupt:
                    print("\n종료됨")
                
                finally:
                    ft_sensor.stop_streaming()
        else:
            print("연결 실패")