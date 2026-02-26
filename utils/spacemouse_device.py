import multiprocessing as mp
import numpy as np
import time
import pyspacemouse
from diffusion_policy.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer


class SpacemouseDevice(mp.Process):
    def __init__(self,
                 shm_manager,
                 get_max_k=30,
                 frequency=200,
                 deadzone=(0, 0, 0, 0, 0, 0),
                 dtype=np.float32,
                 n_buttons=2,
                 ):
        """
        pyspacemouse 패키지를 사용하여 3Dconnexion SpaceMouse 장치의 상태를
        지속적으로 읽어 공유 메모리에 업데이트하는 프로세스.

        pyspacemouse는 이미 정규화된 값(-1.0 ~ 1.0)을 반환합니다.

        deadzone: [0, 1] 범위, 이 값 이하의 축 입력은 0으로 처리됩니다.

        좌표계 (SpaceMouse 기준):
            front
            z
            ^   _
            |  (O) space mouse
            |
            *-----> x right
            y
        """
        super().__init__()

        if np.issubdtype(type(deadzone), np.number):
            deadzone = np.full(6, fill_value=deadzone, dtype=dtype)
        else:
            deadzone = np.array(deadzone, dtype=dtype)
        assert (deadzone >= 0).all()

        # 설정값
        self.frequency = frequency
        self.dtype = dtype
        self.deadzone = deadzone
        self.n_buttons = n_buttons

        # 위치 변환 행렬
        # pyspacemouse 축: x=좌우, y=앞뒤, z=위아래(뽑기)
        # robot 축: x=앞뒤, y=좌우, z=위아래
        self.tx_pos = np.array([
            [0, -1,  0],   # robot_x = -spnav_y (앞뒤)
            [1,  0,  0],   # robot_y =  spnav_x (좌우)
            [0,  0,  1]    # robot_z =  spnav_z (위아래/뽑기)
        ], dtype=dtype)

        # 회전 변환 행렬 (위치와 독립적으로 조정 가능)
        # pyspacemouse 회전 축: roll=X기울기(좌우), pitch=Y기울기(앞뒤), yaw=Z회전(비틀기)
        # state[3:] = [roll, pitch, yaw]
        self.tx_rot = np.array([
            [-1, 0,  0],   # robot_rx = -spnav_pitch (앞뒤 기울기)
            [0,  -1,  0],   # robot_ry =  spnav_roll  (좌우 기울기)
            [0,  0,  -1]    # robot_rz = -spnav_yaw   (비틀기)
        ], dtype=dtype)

        example = {
            # 6DOF (translation x,y,z + rotation roll,pitch,yaw)
            'motion_event': np.zeros((6,), dtype=dtype),
            # 버튼 상태
            'button_state': np.zeros((n_buttons,), dtype=bool),
            'receive_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        # 공유 변수
        self.ready_event = mp.Event()
        self.stop_event = mp.Event()
        self.ring_buffer = ring_buffer

    # ======= 상태 읽기 API ==========

    def get_motion_state(self):
        """
        정규화된 6DOF 상태를 반환합니다. (범위: -1.0 ~ 1.0)
        [tx, ty, tz, rx, ry, rz]
        deadzone 이하의 값은 0으로 처리됩니다.
        """
        state = self.ring_buffer.get()
        state = np.array(state['motion_event'][:6], dtype=self.dtype)
        is_dead = (-self.deadzone < state) & (state < self.deadzone)
        state[is_dead] = 0
        return state

    def get_motion_state_transformed(self):
        """
        오른손 좌표계로 변환된 6DOF 상태를 반환합니다.

        z
        *------> y right
        |   _
        |  (O) space mouse
        v
        x
        back
        """
        state = self.get_motion_state()
        tf_state = np.zeros_like(state)
        tf_state[:3] = self.tx_pos @ state[:3]
        tf_state[3:] = self.tx_rot @ state[3:]
        return tf_state

    def get_button_state(self):
        """버튼 상태 배열을 반환합니다."""
        state = self.ring_buffer.get()
        return state['button_state']

    def is_button_pressed(self, button_id):
        """특정 버튼이 눌려있는지 확인합니다."""
        return self.get_button_state()[button_id]

    # ========== 시작/종료 API ===========

    def start(self, wait=True):
        super().start()
        if wait:
            self.ready_event.wait()

    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= 메인 루프 ==========

    def run(self):
        try:
            with pyspacemouse.open() as device:
                motion_event = np.zeros((6,), dtype=self.dtype)
                button_state = np.zeros((self.n_buttons,), dtype=bool)

                # 초기 상태를 즉시 전송하여 클라이언트가 읽기 시작할 수 있도록 함
                self.ring_buffer.put({
                    'motion_event': motion_event,
                    'button_state': button_state,
                    'receive_timestamp': time.time()
                })
                self.ready_event.set()

                period = 1.0 / self.frequency
                while not self.stop_event.is_set():
                    state = device.read()
                    receive_timestamp = time.time()

                    if state is not None:
                        motion_event[0] = state.x
                        motion_event[1] = state.y
                        motion_event[2] = state.z
                        motion_event[3] = state.roll
                        motion_event[4] = state.pitch
                        motion_event[5] = state.yaw

                        # 버튼 상태 업데이트 (n_buttons 범위 내에서)
                        buttons = state.buttons
                        for i in range(min(self.n_buttons, len(buttons))):
                            button_state[i] = bool(buttons[i])

                        self.ring_buffer.put({
                            'motion_event': motion_event.copy(),
                            'button_state': button_state.copy(),
                            'receive_timestamp': receive_timestamp
                        })

                    time.sleep(period)

        except Exception as e:
            print(f"[SpacemouseDevice] 오류 발생: {e}")
            # 오류 발생 시에도 ready_event를 set하여 데드락 방지
            if not self.ready_event.is_set():
                self.ready_event.set()
            raise
