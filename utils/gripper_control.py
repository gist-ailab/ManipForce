"""Gripper HTTP control and smart gripper logic."""
import requests


def close_gripper_http(server_ip, port):
    try:
        return requests.post(f"http://{server_ip}:{port}/close_gripper", timeout=1.0).status_code == 200
    except Exception:
        return False


def open_gripper_http(server_ip, port):
    try:
        return requests.post(f"http://{server_ip}:{port}/open_gripper", timeout=1.0).status_code == 200
    except Exception:
        return False


def restore_precision_mode(server_ip, port=5000):
    """Policy 종료 시 impedance 파라미터를 안전한 precision_mode로 복원."""
    try:
        requests.post(f"http://{server_ip}:{port}/precision_mode", timeout=2.0)
        print("[IMPEDANCE] precision_mode 복원 완료")
    except Exception as e:
        print(f"[IMPEDANCE] precision_mode 복원 실패: {e}")


def smart_gripper_control(predicted_gripper, gripper_history, config_gripper,
                          franka_api, server_ip, current_gripper_target):
    """History-based gripper control logic. 1.0 = open, 0.0 = close.

    Args:
        predicted_gripper: model predicted gripper value
        gripper_history: deque of recent predictions
        config_gripper: config['gripper'] dict
        franka_api: FrankaAPI instance
        server_ip: robot server IP
        current_gripper_target: current gripper target state ('open'/'close')

    Returns:
        (action, new_gripper_target): action is 'keep'/'close'/'open',
            new_gripper_target is the updated target state
    """
    import time

    try:
        actual_gripper = franka_api.get_gripper_sync()
    except Exception:
        actual_gripper = 0.025

    gripper_history.append(predicted_gripper)
    if len(gripper_history) < config_gripper['history_length']:
        return 'keep', current_gripper_target

    hist_len = config_gripper['history_length']
    recent = list(gripper_history)[-hist_len:]
    close_count = sum(1 for p in recent if p < config_gripper['close_threshold'])
    open_count = sum(1 for p in recent if p >= config_gripper['open_threshold'])

    is_closed = (actual_gripper < 0.045)
    is_open = (actual_gripper >= 0.045)
    wants_close = (close_count >= config_gripper['close_count_threshold'])
    wants_open = (open_count >= config_gripper['open_count_threshold'])

    if wants_close:
        if is_closed:
            return 'keep', 'close'
        elif current_gripper_target != 'close':
            if close_gripper_http(server_ip, config_gripper['http_port']):
                time.sleep(0.1)
                return 'close', 'close'
        return 'close', current_gripper_target

    elif wants_open:
        if is_open:
            return 'keep', 'open'
        elif current_gripper_target != 'open':
            if open_gripper_http(server_ip, config_gripper['http_port']):
                time.sleep(0.1)
                return 'open', 'open'
        return 'open', current_gripper_target

    return 'keep', current_gripper_target
