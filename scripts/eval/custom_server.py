"""
Franka 로봇 제어 서버 (최소 버전)
- Flask 서버로 pose/state 요청 처리
- 최신 명령만 실행 (오래된 명령 skip)
- Task state에 따라 임피던스 자동 조정
"""
from flask import Flask, request, jsonify
import numpy as np
import rospy
import time
import subprocess
import signal
import sys
import threading
from scipy.spatial.transform import Rotation as R
from absl import app, flags

from franka_msgs.msg import ErrorRecoveryActionGoal, FrankaState
from franka_gripper.msg import GraspActionGoal, MoveActionGoal
from serl_franka_controllers.msg import ZeroJacobian
from sensor_msgs.msg import JointState
import geometry_msgs.msg as geom_msg
from dynamic_reconfigure.client import Client as ReconfClient

FLAGS = flags.FLAGS
flags.DEFINE_string("robot_ip", "172.27.190.2", "IP address of the franka robot's controller box")
flags.DEFINE_float("gripper_dist", 0.05, "Gripper open distance")
flags.DEFINE_bool("force_base_frame", False, "Use base frame for force/torque")

JOINT_RESET_TARGET = [-0.07, -0.1, 0.0, -2.5, -0.1, 2.5, -0.6]

# 전역 종료 플래그
shutdown_flag = False

def signal_handler(sig, frame):
    global shutdown_flag
    print('\n[SHUTDOWN] Ctrl+C 감지, 서버 종료 중...')
    shutdown_flag = True
    if rospy.is_initialized():
        rospy.signal_shutdown('User interrupted')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


class FrankaServer:
    """Franka 로봇 제어 서버"""

    def __init__(self):
        self.grippermovepub = rospy.Publisher("/franka_gripper/move/goal", MoveActionGoal, queue_size=1)
        self.grippergrasppub = rospy.Publisher("/franka_gripper/grasp/goal", GraspActionGoal, queue_size=1)
        self.eepub = rospy.Publisher("/cartesian_impedance_controller/equilibrium_pose", geom_msg.PoseStamped, queue_size=10)
        self.resetpub = rospy.Publisher("/franka_control/error_recovery/goal", ErrorRecoveryActionGoal, queue_size=1)
        self.gripper_sub = rospy.Subscriber("/franka_gripper/joint_states", JointState, self._update_gripper)
        self.jacobian_sub = rospy.Subscriber("/cartesian_impedance_controller/franka_jacobian", ZeroJacobian, self._set_jacobian)
        time.sleep(2)
        self.state_sub = rospy.Subscriber("franka_state_controller/franka_states", FrankaState, self._set_currpos)

    def start_impedance(self):
        """임피던스 컨트롤러 시작"""
        self.imp = subprocess.Popen([
            "roslaunch", "serl_franka_controllers", "impedance_gumi.launch",
            "robot_ip:=" + FLAGS.robot_ip, "load_gripper:=true"
        ], stdout=subprocess.PIPE)
        time.sleep(5)

    def stop_impedance(self):
        """임피던스 컨트롤러 중지"""
        self.imp.terminate()
        time.sleep(1)

    def clear(self):
        """에러 복구"""
        msg = ErrorRecoveryActionGoal()
        self.resetpub.publish(msg)

    def move(self, pose: list):
        """로봇 이동: [x, y, z, qx, qy, qz, qw]"""
        assert len(pose) == 7
        msg = geom_msg.PoseStamped()
        msg.header.frame_id = "0"
        msg.header.stamp = rospy.Time.now()
        msg.pose.position = geom_msg.Point(pose[0], pose[1], pose[2])
        msg.pose.orientation = geom_msg.Quaternion(pose[3], pose[4], pose[5], pose[6])
        self.eepub.publish(msg)

    def _set_currpos(self, msg):
        tmatrix = np.array(list(msg.O_T_EE)).reshape(4, 4).T
        r = R.from_matrix(tmatrix[:3, :3])
        pose = np.concatenate([tmatrix[:3, -1], r.as_quat()])
        self.pos = pose
        self.dq = np.array(list(msg.dq)).reshape((7,))
        self.q = np.array(list(msg.q)).reshape((7,))
        force_torque = msg.O_F_ext_hat_K if FLAGS.force_base_frame else msg.K_F_ext_hat_K
        self.force = np.array(list(force_torque)[:3])
        self.torque = np.array(list(force_torque)[3:])
        try:
            self.vel = self.jacobian @ self.dq
        except:
            self.vel = np.zeros(6)

    def _set_jacobian(self, msg):
        self.jacobian = np.array(list(msg.zero_jacobian)).reshape((6, 7), order="F")

    def _update_gripper(self, msg):
        self.gripper_pos = np.sum(msg.position)

    def close(self):
        grasp = GraspActionGoal()
        grasp.goal.width = 0.02
        grasp.goal.speed = 0.15
        grasp.goal.epsilon.inner = 1
        grasp.goal.epsilon.outer = 1
        grasp.goal.force = 0.2
        self.grippergrasppub.publish(grasp)

    def open(self):
        msg = MoveActionGoal()
        msg.goal.width = FLAGS.gripper_dist
        msg.goal.speed = 0.15
        self.grippermovepub.publish(msg)

    def control_gripper(self, width):
        msg = MoveActionGoal()
        msg.goal.width = width
        msg.goal.speed = 0.15
        self.grippermovepub.publish(msg)


###############################################################################

def main(_):
    webapp = Flask(__name__)

    try:
        roscore = subprocess.Popen("roscore")
        time.sleep(1)
    except Exception as e:
        raise Exception("roscore not running", e)

    rospy.init_node("franka_control_api")

    robot_server = FrankaServer()
    robot_server.start_impedance()

    reconf_client = ReconfClient("cartesian_impedance_controllerdynamic_reconfigure_compliance_param_node")

    # 서버 시작 시 안전한 기본값으로 초기화 (precision_mode)
    print("[INIT] Resetting to precision mode (stiffness=2000, clip=7mm/s)")
    reconf_client.update_configuration({"translational_stiffness": 2000})
    reconf_client.update_configuration({"translational_damping": 80})
    reconf_client.update_configuration({"rotational_stiffness": 150})
    reconf_client.update_configuration({"rotational_damping": 7})
    reconf_client.update_configuration({"translational_Ki": 10})
    reconf_client.update_configuration({"rotational_Ki": 5})
    for direction in ['x', 'y', 'z', 'neg_x', 'neg_y', 'neg_z']:
        reconf_client.update_configuration({"translational_clip_" + direction: 0.007})
        reconf_client.update_configuration({"rotational_clip_" + direction: 0.04})
    print("[INIT] Impedance parameters reset complete")

    # 현재 임피던스 모드 추적
    current_impedance_mode = None

    # 최신 target만 유지
    latest_target = {"pos": None, "time": 0}
    target_lock = threading.Lock()

    # Flask Routes
    @webapp.route("/startimp", methods=["POST"])
    def start_impedance():
        robot_server.clear()
        robot_server.start_impedance()
        return "Started impedance"

    @webapp.route("/stopimp", methods=["POST"])
    def stop_impedance():
        robot_server.stop_impedance()
        return "Stopped impedance"

    @webapp.route("/getpos", methods=["POST"])
    def get_pos():
        return jsonify({"pose": np.array(robot_server.pos).tolist()})

    @webapp.route("/getvel", methods=["POST"])
    def get_vel():
        return jsonify({"vel": np.array(robot_server.vel).tolist()})

    @webapp.route("/getforce", methods=["POST"])
    def get_force():
        return jsonify({"force": np.array(robot_server.force).tolist()})

    @webapp.route("/gettorque", methods=["POST"])
    def get_torque():
        return jsonify({"torque": np.array(robot_server.torque).tolist()})

    @webapp.route("/getq", methods=["POST"])
    def get_q():
        return jsonify({"q": np.array(robot_server.q).tolist()})

    @webapp.route("/getdq", methods=["POST"])
    def get_dq():
        return jsonify({"dq": np.array(robot_server.dq).tolist()})

    @webapp.route("/getjacobian", methods=["POST"])
    def get_jacobian():
        return jsonify({"jacobian": np.array(robot_server.jacobian).tolist()})

    @webapp.route("/get_gripper", methods=["POST"])
    def get_gripper():
        return jsonify({"gripper": robot_server.gripper_pos})

    @webapp.route("/open_gripper", methods=["POST"])
    def open():
        robot_server.open()
        return "Opened"

    @webapp.route("/close_gripper", methods=["POST"])
    def close():
        robot_server.close()
        return "Closed"

    @webapp.route("/control_gripper", methods=["POST"])
    def control_gripper():
        data = request.get_json()
        width = data.get('width', 0.04)
        robot_server.control_gripper(width)
        return "Control Gripper"

    @webapp.route("/clearerr", methods=["POST"])
    def clear():
        robot_server.clear()
        return "Clear"

    @webapp.route("/pose", methods=["POST"])
    def pose():
        nonlocal latest_target
        recv_time = time.time()
        data = request.json
        pos = np.array(data["arr"])

        # 자동 에러 복구 (예전 방식)
        robot_server.clear()

        # 최신 target으로 덮어쓰기
        with target_lock:
            latest_target = {"pos": pos, "time": recv_time, "data": data}

        return "Queued"

    @webapp.route("/pose_sync", methods=["POST"])
    def pose_sync():
        """Sync pose: 로봇이 도달할 때까지 블로킹 후 현재 pose 반환"""
        nonlocal current_impedance_mode
        data = request.json
        pos = np.array(data["arr"])
        thresh = data.get("reach_threshold", 0.0005)  # 기본값 0.5mm
        timeout_s = data.get("reach_timeout_ms", 500) / 1000.0

        # 자동 에러 복구 (예전 방식)
        robot_server.clear()

        # 현재 위치와 목표 거리 기반 timeout 자동 조정 (큰 이동 시)
        current_pos = robot_server.pos[:3]
        distance = np.linalg.norm(pos[:3] - current_pos)
        if distance > 0.05:  # 50mm 이상 이동
            # 50mm/s 속도 가정 + 여유 2초
            estimated_time = (distance / 0.030) + 2.0  # 30mm/s + 2s 여유
            timeout_s = max(timeout_s, estimated_time)
            print(f"[POSE_SYNC] 큰 이동 감지 ({distance*1000:.1f}mm), timeout 자동 연장 → {timeout_s:.1f}s")

        print(f"[POSE_SYNC] Received target: {pos[:3]}, thresh={thresh*1000:.1f}mm, timeout={timeout_s*1000:.0f}ms")

        # Task state 처리 (REACHING/CONTACT 모두 precision_mode 사용)
        if "task_state" in data:
            task_state = data["task_state"]
            # 현재는 둘 다 precision_mode로 통일 (예전 방식)

        # 시작 위치 저장
        start_pos = robot_server.pos[:3].copy()

        # 직접 이동 명령
        robot_server.move(pos)

        # 도달 대기: 실제로 움직이기 시작할 때까지 기다림
        start_t = time.time()
        moved = False
        status = "timeout"

        while time.time() - start_t < timeout_s:
            curr = robot_server.pos[:3]

            # 1mm 이상 움직였는지 체크 (실제 움직임 시작 확인)
            if not moved:
                if np.linalg.norm(curr - start_pos) > 0.001:
                    moved = True
                else:
                    time.sleep(0.002)
                    continue

            # 움직임 시작 후 도달 체크
            pos_err = np.linalg.norm(curr - pos[:3])
            if pos_err < thresh:
                status = "reached"
                break
            time.sleep(0.002)

        curr_pose = robot_server.pos
        pos_err_final = np.linalg.norm(curr_pose[:3] - pos[:3])
        err_xyz = (curr_pose[:3] - pos[:3]) * 1000  # mm 단위
        print(f"[POSE_SYNC] {status.upper()}: err_total={pos_err_final*1000:.2f}mm [x={err_xyz[0]:+.2f}, y={err_xyz[1]:+.2f}, z={err_xyz[2]:+.2f}], elapsed={time.time()-start_t:.3f}s")

        return jsonify({
            "status": status,
            "pose": np.array(curr_pose).tolist()
        })

    @webapp.route("/getstate", methods=["POST"])
    def get_state():
        return jsonify({
            "pose": np.array(robot_server.pos).tolist(),
            "vel": np.array(robot_server.vel).tolist(),
            "force": np.array(robot_server.force).tolist(),
            "torque": np.array(robot_server.torque).tolist(),
            "q": np.array(robot_server.q).tolist(),
            "dq": np.array(robot_server.dq).tolist(),
            "jacobian": np.array(robot_server.jacobian).tolist(),
            "gripper": robot_server.gripper_pos,
        })

    @webapp.route("/precision_mode", methods=["POST"])
    def precision_mode():
        reconf_client.update_configuration({"translational_stiffness": 2000})
        reconf_client.update_configuration({"translational_damping": 80})
        reconf_client.update_configuration({"rotational_stiffness": 150})
        reconf_client.update_configuration({"rotational_damping": 7})
        reconf_client.update_configuration({"translational_Ki": 10})
        reconf_client.update_configuration({"rotational_Ki": 5})
        for direction in ['x', 'y', 'z', 'neg_x', 'neg_y', 'neg_z']:
            reconf_client.update_configuration({"translational_clip_" + direction: 0.007})
            reconf_client.update_configuration({"rotational_clip_" + direction: 0.04})
        return 'Precision'

    @webapp.route("/compliance_mode", methods=["POST"])
    def compliance_mode():
        reconf_client.update_configuration({"translational_stiffness": 800})
        reconf_client.update_configuration({"translational_damping": 89})
        reconf_client.update_configuration({"rotational_stiffness": 150})
        reconf_client.update_configuration({"rotational_damping": 7})
        reconf_client.update_configuration({"translational_Ki": 10})
        reconf_client.update_configuration({"rotational_Ki": 0})
        for direction in ['x', 'y', 'z', 'neg_x', 'neg_y', 'neg_z']:
            reconf_client.update_configuration({"translational_clip_" + direction: 0.007})
            reconf_client.update_configuration({"rotational_clip_" + direction: 0.07})
        return 'Compliance'

    # Pose executor thread: 최신 명령만 실행 + 임피던스 자동 조정
    def pose_executor():
        nonlocal latest_target, current_impedance_mode
        last_exec_time = 0

        while not shutdown_flag:
            time.sleep(0.005)  # 200Hz

            with target_lock:
                if latest_target["pos"] is None:
                    continue
                target = latest_target.copy()
                latest_target["pos"] = None  # 소비됨 표시

            pos = target["pos"]
            data = target.get("data", {})

            # Task state에 따른 임피던스 자동 조정
            if "task_state" in data:
                task_state = data["task_state"]

                if task_state == "REACHING" and current_impedance_mode != "REACHING":
                    # Reaching: precision_mode 기본값 사용 (예전 방식)
                    print("[IMPEDANCE] Switching to REACHING mode (precision_mode: stiffness=2000, damping=80, clip=7mm/s)")
                    reconf_client.update_configuration({"translational_stiffness": 2000})
                    reconf_client.update_configuration({"translational_damping": 80})
                    for direction in ['x', 'y', 'z', 'neg_x', 'neg_y', 'neg_z']:
                        reconf_client.update_configuration({"translational_clip_" + direction: 0.007})
                    reconf_client.update_configuration({"translational_Ki": 10})
                    reconf_client.update_configuration({"rotational_stiffness": 150})
                    reconf_client.update_configuration({"rotational_damping": 7})
                    reconf_client.update_configuration({"rotational_Ki": 5})
                    for direction in ['x', 'y', 'z', 'neg_x', 'neg_y', 'neg_z']:
                        reconf_client.update_configuration({"rotational_clip_" + direction: 0.04})
                    current_impedance_mode = "REACHING"

                elif task_state in ["CONTACT", "INSERTED"] and current_impedance_mode != "CONTACT":
                    # Contact: precision_mode 기본값 사용 (예전 방식)
                    print("[IMPEDANCE] Switching to CONTACT mode (precision_mode: stiffness=2000, damping=80, clip=7mm/s)")
                    reconf_client.update_configuration({"translational_stiffness": 2000})
                    reconf_client.update_configuration({"translational_damping": 80})
                    for direction in ['x', 'y', 'z', 'neg_x', 'neg_y', 'neg_z']:
                        reconf_client.update_configuration({"translational_clip_" + direction: 0.007})
                    reconf_client.update_configuration({"translational_Ki": 10})
                    reconf_client.update_configuration({"rotational_stiffness": 150})
                    reconf_client.update_configuration({"rotational_damping": 7})
                    reconf_client.update_configuration({"rotational_Ki": 5})
                    for direction in ['x', 'y', 'z', 'neg_x', 'neg_y', 'neg_z']:
                        reconf_client.update_configuration({"rotational_clip_" + direction: 0.04})
                    current_impedance_mode = "CONTACT"

            exec_time = time.time()
            # print(f"[EXEC] pos={pos[:3]}, dt={exec_time-last_exec_time:.3f}s")  # 로그 스팸 방지
            robot_server.move(pos)
            last_exec_time = exec_time

    executor_thread = threading.Thread(target=pose_executor, daemon=True)
    executor_thread.start()

    print("[SERVER] Flask 서버 시작 (Ctrl+C로 종료)")
    webapp.run(host="0.0.0.0", threaded=True)


if __name__ == "__main__":
    app.run(main)
