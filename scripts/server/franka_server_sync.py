"""
Franka 로봇 제어 서버 (Sync 버전)
- move_to_pos 후 도달 완료 시 ACK(현재 pose 포함)를 클라이언트에 push
- FT publish는 메인 스레드에서 100Hz 유지
- 로봇 커맨드 수신/실행/ACK는 별도 스레드
"""

import numpy as np
import time
import argparse
from modules.config import Config
import signal
import sys
import socket
import json
import threading
from scipy.spatial.transform import Rotation as R
from robot_infra.envs.franka_vc_env import FrankaVC
import rospy
from geometry_msgs.msg import WrenchStamped
import select

# Z축 오프셋
T_offset = np.eye(4)
T_offset[2, 3] = 0.05

def transform_pose(orig_pose, T_offset):
    pos = orig_pose[0:3]
    w, x, y, z = orig_pose[3:7]
    quat_xyzw = np.array([x, y, z, w])
    T_orig = np.eye(4)
    T_orig[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    T_orig[:3, 3] = pos
    T_new = T_offset @ T_orig
    new_pos = T_new[:3, 3]
    new_quat = R.from_matrix(T_new[:3, :3]).as_quat()
    new_quat_wxyz = np.array([new_quat[3], new_quat[0], new_quat[1], new_quat[2]])
    return np.hstack([new_pos, new_quat_wxyz])

def signal_handler(sig, frame):
    print('\n프로그램을 종료합니다.')
    if 'control' in globals():
        control.stop()
    if rospy.is_initialized():
        rospy.signal_shutdown('Program ended')
    sys.exit(0)

def _get_observation(control, initial_force_torque):
    curr_ee_pose = control.get_ee_pose()
    curr_ee_trans = curr_ee_pose['pose'][:3]
    curr_ee_quat = curr_ee_pose['pose'][3:]
    curr_force = control.get_ee_force()['force']
    curr_torque = control.get_ee_torque()['torque']
    force_array = -(np.array(curr_force) - initial_force_torque[:3])
    torque_array = -(np.array(curr_torque) - initial_force_torque[3:])
    curr_ee_ft = np.concatenate((force_array, torque_array))
    curr_joint_pos = control.get_joint_pos()
    return curr_ee_pose, curr_ee_trans, curr_ee_quat, curr_ee_ft, curr_joint_pos

def initialize_force_torque_sensors(control):
    initial_force = control.get_ee_force()['force']
    initial_torque = control.get_ee_torque()['torque']
    initial_force_torque = np.concatenate((initial_force, initial_torque))
    return initial_force_torque

def _publish_ft_topic(control, initial_force_torque, force_torque_pub):
    curr_force = control.get_ee_force()['force']
    curr_torque = control.get_ee_torque()['torque']
    force_torque_msg = WrenchStamped()
    force_torque_msg.header.stamp = rospy.Time.now()
    force_torque_msg.header.frame_id = 'ee_frame'
    force_torque_msg.wrench.force.x = curr_force[0]
    force_torque_msg.wrench.force.y = curr_force[1]
    force_torque_msg.wrench.force.z = curr_force[2]
    force_torque_msg.wrench.torque.x = curr_torque[0]
    force_torque_msg.wrench.torque.y = curr_torque[1]
    force_torque_msg.wrench.torque.z = curr_torque[2]
    force_torque_pub.publish(force_torque_msg)

def move_to_initial_pose(control):
    initial_pose = np.array([0.45, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0])
    control.move_to_pos(initial_pose)
    time.sleep(2.0)
    curr_pose = control.get_ee_pose()['pose']
    print("\n=== 초기화 완료 ===")
    print(f"현재 위치: {curr_pose[:3]}")
    print(f"현재 자세: {curr_pose[3:]}")
    print("===================\n")
    return initial_pose


def _command_handler_thread(control, server_socket, reach_threshold, reach_timeout):
    """
    클라이언트로부터 커맨드를 수신하고, 로봇 실행 후 도달 완료 ACK를 전송하는 스레드.

    프로토콜:
    - 클라이언트 → 서버: {"target_pose": [...], "gripper_command": "...", ...}\n
    - 서버 → 클라이언트: {"status": "reached"|"timeout", "pose": [...]}\n
    """
    client_socket = None
    buffer = ""

    while not rospy.is_shutdown():
        # 클라이언트 연결 대기
        if client_socket is None:
            readable, _, _ = select.select([server_socket], [], [], 0.01)
            if readable:
                client_socket, addr = server_socket.accept()
                client_socket.setblocking(False)
                buffer = ""
                print(f"[CMD] {addr}에서 연결됨")
            continue

        # 데이터 수신
        try:
            data = client_socket.recv(4096).decode('utf-8')
            if not data:
                print("[CMD] 클라이언트 연결 종료")
                client_socket.close()
                client_socket = None
                continue
            buffer += data
        except BlockingIOError:
            time.sleep(0.001)
            continue
        except ConnectionResetError:
            print("[CMD] 클라이언트 연결이 끊어졌습니다.")
            client_socket.close()
            client_socket = None
            continue

        # 메시지 파싱 및 실행
        while '\n' in buffer:
            message, buffer = buffer.split('\n', 1)
            if not message.strip():
                continue
            try:
                received_data = json.loads(message)
            except json.JSONDecodeError as e:
                print(f"[CMD] JSON 디코딩 오류: {e}")
                continue

            # 그리퍼 제어
            if 'gripper_command' in received_data:
                gripper_state = received_data['gripper_command']
                if gripper_state == 'close' and control.currgrip == 0:
                    control.close_gripper()
                elif gripper_state == 'open' and control.currgrip == 1:
                    control.open_gripper()

            # target_pose가 없으면 ACK만 보냄 (gripper only 등)
            if 'target_pose' not in received_data:
                try:
                    curr_pose = control.get_ee_pose()['pose']
                    ack = json.dumps({
                        "status": "ok",
                        "pose": np.array(curr_pose).tolist()
                    }, separators=(',', ':'))
                    client_socket.sendall(ack.encode('utf-8') + b'\n')
                except Exception:
                    pass
                continue

            target_pose = np.array(received_data['target_pose'])

            # reach_threshold / timeout override (클라이언트에서 개별 설정 가능)
            thresh = received_data.get('reach_threshold', reach_threshold)
            timeout = received_data.get('reach_timeout_ms', reach_timeout * 1000) / 1000.0

            # 로봇 이동 명령
            control.move_to_pos(target_pose)

            # 도달 대기
            start_t = time.time()
            status = "timeout"
            while time.time() - start_t < timeout:
                curr_pose = control.get_ee_pose()['pose']
                pos_err = np.linalg.norm(np.array(curr_pose[:3]) - target_pose[:3])
                if pos_err < thresh:
                    status = "reached"
                    break
                time.sleep(0.002)  # 500Hz 체크

            # ACK 전송: 도달 상태 + 현재 pose
            curr_pose = control.get_ee_pose()['pose']
            ack = json.dumps({
                "status": status,
                "pose": np.array(curr_pose).tolist()
            }, separators=(',', ':'))
            try:
                client_socket.sendall(ack.encode('utf-8') + b'\n')
            except (socket.error, BrokenPipeError):
                print("[CMD] ACK 전송 실패, 클라이언트 연결 해제")
                client_socket.close()
                client_socket = None


def run_robot_server(control, force_torque_pub, host='0.0.0.0', port=5000,
                     reach_threshold=0.002, reach_timeout=0.3):
    """
    로봇 제어 서버 실행
    - 메인 스레드: FT publish (100Hz)
    - 커맨드 스레드: 클라이언트 수신 → 로봇 이동 → 도달 ACK
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    try:
        server_socket.bind((host, port))
        server_socket.listen(1)
        print(f"[서버] {host}:{port}에서 대기 중 (reach_threshold={reach_threshold}m, timeout={reach_timeout}s)")

        initial_force_torque = initialize_force_torque_sensors(control)

        # 커맨드 핸들러 스레드 시작
        cmd_thread = threading.Thread(
            target=_command_handler_thread,
            args=(control, server_socket, reach_threshold, reach_timeout),
            daemon=True
        )
        cmd_thread.start()

        # 메인 스레드: FT publish 유지
        rospy_rate = rospy.Rate(100)
        while not rospy.is_shutdown():
            _publish_ft_topic(control, initial_force_torque, force_torque_pub)
            rospy_rate.sleep()

    except KeyboardInterrupt:
        print("\n서버를 종료합니다.")
    finally:
        server_socket.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_robot', type=str,
                        default='robot_infra/configs/robot_params.yaml',
                        help='로봇 설정 파일 경로')
    parser.add_argument('--port', type=int, default=4999,
                        help='서버 포트 번호')
    parser.add_argument('--gripper', type=str, default="open", choices=['open', 'close'],
                        help="Control the state of the gripper: open or close")
    parser.add_argument('--reach_threshold', type=float, default=0.002,
                        help='도달 판정 임계값 (m)')
    parser.add_argument('--reach_timeout', type=float, default=0.3,
                        help='도달 대기 타임아웃 (s)')
    args = parser.parse_args()

    start_gripper = 0 if args.gripper == 'close' else 1

    config_robot = Config(args.config_robot).get_config()
    control = FrankaVC(config_robot=config_robot, hz=200, start_gripper=start_gripper)
    control.precision_mode()

    try:
        curr_pose = control.get_ee_pose()['pose']
        print(f"현재 위치: {curr_pose[:3]}")
        print(f"현재 자세: {curr_pose[3:]}")

        rospy.init_node('robot_force_torque_publisher')
        force_torque_pub = rospy.Publisher('ee_force_torque', WrenchStamped, queue_size=5)

        run_robot_server(control, force_torque_pub, port=args.port,
                         reach_threshold=args.reach_threshold,
                         reach_timeout=args.reach_timeout)

    except KeyboardInterrupt:
        print('\n프로그램이 사용자에 의해 중단되었습니다.')
    finally:
        if 'control' in locals():
            control.stop()
        if rospy.is_initialized():
            rospy.signal_shutdown('Program ended')


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    main()
