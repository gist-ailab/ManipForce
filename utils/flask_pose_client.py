"""Flask-based robot pose client: sends pose commands via REST API."""
import json
import time
import socket
import requests


class FlaskPoseClient:
    """Drop-in replacement for TCP socket — sends pose commands via Flask server.

    sync=True:  /pose_sync (서버가 도달 대기 후 pose 반환, blocking)
    sync=False: /pose      (비동기, fire-and-forget)
    """

    def __init__(self, host, rest_port=5000, timeout=2.0, sync=False):
        self.base_url = f"http://{host}:{rest_port}"
        self.sync = sync
        self.pose_url = f"{self.base_url}/pose_sync" if sync else f"{self.base_url}/pose"
        self.timeout = timeout
        self._session = requests.Session()
        self._last_gripper = None
        self.last_ack = None
        mode_str = "sync" if sync else "async"
        print(f"[FlaskPoseClient] 연결: {self.base_url} ({mode_str} mode)")

    def sendall(self, data_bytes):
        """client_socket.sendall(json_bytes + b'\\n') 호환."""
        message = data_bytes.decode('utf-8').strip()
        if not message:
            return
        try:
            parsed = json.loads(message)
        except json.JSONDecodeError:
            return

        is_reset = parsed.get("reset", False)

        gripper_cmd = parsed.get("gripper_command", "keep")
        if gripper_cmd != "keep" and gripper_cmd != self._last_gripper:
            try:
                if gripper_cmd == "close":
                    self._session.post(f"{self.base_url}/close_gripper", timeout=self.timeout)
                elif gripper_cmd == "open":
                    self._session.post(f"{self.base_url}/open_gripper", timeout=self.timeout)
                self._last_gripper = gripper_cmd
            except requests.RequestException:
                pass

        target_pose = parsed.get("target_pose")
        if target_pose is None:
            return

        payload = {"arr": target_pose}
        if "task_state" in parsed:
            payload["task_state"] = parsed["task_state"]
        if self.sync:
            payload["reach_threshold"] = parsed.get("reach_threshold", 0.002)
            payload["reach_timeout_ms"] = parsed.get("reach_timeout_ms", 500)

        try:
            resp = self._session.post(self.pose_url, json=payload, timeout=self.timeout)
            if self.sync:
                self.last_ack = resp.json()
                if is_reset:
                    print(f"[FlaskPoseClient] RESET 완료! ack={self.last_ack.get('status')}", flush=True)
            else:
                self.last_ack = None
                if is_reset:
                    print(f"[FlaskPoseClient] RESET 완료! resp={resp.text}", flush=True)
        except Exception as e:
            self.last_ack = None
            mode_str = "sync" if self.sync else "async"
            raise socket.error(f"Flask {self.pose_url} 전송 실패 ({mode_str}): {e}")

    def close(self):
        self._session.close()
        print("[FlaskPoseClient] 연결 종료")


def connect_to_server(host='localhost', port=4999, timeout=5, sync=False):
    """Flask 서버로 명령을 보내는 클라이언트 반환."""
    try:
        return FlaskPoseClient(host, rest_port=5000, timeout=timeout, sync=sync)
    except Exception as e:
        print(f"FlaskPoseClient 생성 실패: {e}")
        return None


def send_pose_command(client_socket, pose, gripper='keep', reset=False, **extra):
    """Helper to build and send a pose command JSON."""
    data = {
        'target_pose': pose.tolist() if hasattr(pose, 'tolist') else pose,
        'gripper_command': gripper,
        'reset': reset,
        'timestamp': time.time(),
    }
    data.update(extra)
    client_socket.sendall(json.dumps(data, separators=(',', ':')).encode('utf-8') + b'\n')
