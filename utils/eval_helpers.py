"""Eval helper functions: observation, inference, mode transitions."""

import sys
import termios
import time
import json
import numpy as np
import cv2
import torch

from scipy.spatial.transform import Rotation as R
from diffusion_policy.common.pytorch_util import dict_apply
from utils.real_inference_util import get_real_gumi_obs_dict
from utils.gripper_control import restore_precision_mode
from utils.flask_pose_client import send_pose_command
from utils.eval_logger import save_trial_result
from utils.obs_builder import convert_torque_scale, get_ft_time_range, select_ft_window, build_obs_np
from utils.gui_thread import check_keyboard_input
from utils.hardware_init import compute_initial_pose, reset_to_home
from utils.observe import print_observe_table, save_observe_csv


def prefill_obs_history(state, config, camera, additional_cam, cam_lock=None):
    """Policy 진입 전: 카메라 링 버퍼가 충분히 차도록 대기."""
    ds = int(config['image'].get('down_sample_steps', 1))
    obs_ds = int(config['image'].get('obs_down_sample_steps', ds))
    fps = config.get('action', {}).get('data_hz', 30)
    wait_sec = obs_ds / fps * config['image']['history_length'] + 0.1
    time.sleep(wait_sec)


def _sample_pose_by_time(pose_buffer, target_ts):
    """Pose 링 버퍼에서 target timestamps에 가장 가까운 pose 샘플링."""
    if not pose_buffer:
        return None
    buf = list(pose_buffer)
    buf_ts = np.array([b[0] for b in buf])
    results = []
    for t in target_ts:
        idx = np.argmin(np.abs(buf_ts - t))
        results.append(buf[idx][1])  # 7D pose
    return results


def _compute_pose_wrt_start(poses, start_pose):
    """학습과 동일한 방식으로 pose_wrt_start 계산: relative position + relative quaternion.

    Args:
        poses: list of 7D poses [x,y,z, qx,qy,qz,qw] (XYZW from server)
        start_pose: 7D start pose [x,y,z, qx,qy,qz,qw] (XYZW)
    Returns:
        (N, 7) array of relative poses [dx,dy,dz, rel_qx,rel_qy,rel_qz,rel_qw] (XYZW)
    """
    start_pos = start_pose[:3]
    start_rot = R.from_quat(start_pose[3:])

    result = np.zeros((len(poses), 7), dtype=np.float32)
    for i, pose in enumerate(poses):
        result[i, :3] = pose[:3] - start_pos
        curr_rot = R.from_quat(pose[3:])
        rel_rot = start_rot.inv() * curr_rot
        result[i, 3:] = rel_rot.as_quat()  # xyzw
    return result



def get_obs(state, config, camera, additional_cam, cam_lock=None,
            ft_collector=None, last_executed_action=None):
    """Build observation dict by time-based sampling from camera ring buffers.

    Images: sample hist_len frames at ds/fps intervals from camera ring buffer.
    FT: sample fixed ds/fps window from FT ring buffer.
    pose_wrt_start: sample from EE pose buffer at same timestamps as images.
    All match training's down_sample_steps spacing.
    """
    hist_len = config['image']['history_length']
    ds = int(config['image'].get('down_sample_steps', 1))
    obs_ds = int(config['image'].get('obs_down_sample_steps', ds))
    fps = config.get('action', {}).get('data_hz', 30)
    interval = obs_ds / fps

    # 최신 카메라 프레임의 실제 timestamp를 기준으로 샘플링
    latest = camera.get_latest_rgb()
    now = latest[0] if latest is not None else time.time()

    # Time-based sampling: [now - (hist_len-1)*interval, ..., now]
    target_ts = [now - (hist_len - 1 - i) * interval for i in range(hist_len)]

    frames_cam1 = camera.get_frames_by_time(target_ts)
    frames_cam2 = additional_cam.get_frames_by_time(target_ts)

    if frames_cam1 is None or frames_cam2 is None:
        # Fallback: cameras not ready yet, wait and retry
        time.sleep(0.1)
        frames_cam1 = camera.get_frames_by_time(target_ts)
        frames_cam2 = additional_cam.get_frames_by_time(target_ts)

    img_timestamps = [f[0] for f in frames_cam1]
    cam1_imgs = [f[1] for f in frames_cam1]
    cam2_imgs = [f[1] for f in frames_cam2]

    # Update state for display
    state.prev_frames_cam1.clear()
    state.prev_frames_cam2.clear()
    state.prev_timestamps.clear()
    for i in range(hist_len):
        state.prev_frames_cam1.append(cam1_imgs[i])
        state.prev_frames_cam2.append(cam2_imgs[i])
        state.prev_timestamps.append(img_timestamps[i])

    # FT sensor (sensor mode only: bias 제거 + optional offset)
    ft_buf, ts_buf = ft_collector.window(config['ft_sensor']['buffer_length'])
    t0, t1 = get_ft_time_range(state.prev_timestamps, config)
    state.ft_model_t0 = t0
    state.ft_model_t1 = t1
    ft_sel, ts_sel = select_ft_window(ft_buf, ts_buf, t0, t1, config)

    # FT[5:100] recalibration: 학습과 동일한 정규화 방식 재현
    if not state.ft_recalib_done and state.ft_recalib_start_time is not None:
        new_mask = ts_buf > state.ft_recalib_start_time
        if new_mask.any():
            new_ft = ft_buf[new_mask]
            n_new = len(new_ft)
            if n_new >= 100:
                recalib_bias = new_ft[5:100].mean(axis=0).astype(np.float32)
                old_bias = state.gripping_ft_bias.copy()
                state.gripping_ft_bias = recalib_bias
                state.ft_recalib_done = True
                diff = recalib_bias - old_bias
                print(f"\n[FT RECALIB] FT[5:100] bias 재계산 완료 (N={n_new})")
                print(f"  이전 bias: F=[{old_bias[0]:+.3f},{old_bias[1]:+.3f},{old_bias[2]:+.3f}] "
                      f"T=[{old_bias[3]:+.4f},{old_bias[4]:+.4f},{old_bias[5]:+.4f}]")
                print(f"  새 bias:   F=[{recalib_bias[0]:+.3f},{recalib_bias[1]:+.3f},{recalib_bias[2]:+.3f}] "
                      f"T=[{recalib_bias[3]:+.4f},{recalib_bias[4]:+.4f},{recalib_bias[5]:+.4f}]")
                print(f"  차이:      F=[{diff[0]:+.3f},{diff[1]:+.3f},{diff[2]:+.3f}] "
                      f"T=[{diff[3]:+.4f},{diff[4]:+.4f},{diff[5]:+.4f}]")

    # Subtract gripping bias (학습 zarr와 동일: 에피소드 초기 FT 평균 제거)
    ft_sel = ft_sel - state.gripping_ft_bias

    # 실시간 FT에 training 전체 평균을 더해서 offset 적용
    ft_offset = config['ft_sensor'].get('fixed_values', None)
    if ft_offset is not None:
        ft_sel = ft_sel + np.array(ft_offset, dtype=np.float32)

    state.ft_model_input = ft_sel[-1].copy() if ft_sel.ndim == 2 else ft_sel.copy()
    state.ft_model_input_history.append((ts_sel.copy(), ft_sel.copy()))

    # 물리적 moment arm 차이 보정 (GUMI 12cm vs Panda 18cm)
    ft_sel = convert_torque_scale(ft_sel)
    obs = build_obs_np(cam1_imgs, cam2_imgs,
                       ft_sel, config, ts_sel, last_executed_action)

    # pose_wrt_start: sample EE poses at same timestamps as images
    if state.policy_start_pose is not None and len(state.ee_pose_buffer) > 0:
        sampled_poses = _sample_pose_by_time(state.ee_pose_buffer, img_timestamps)
        if sampled_poses is not None:
            pose_wrt_start = _compute_pose_wrt_start(sampled_poses, state.policy_start_pose)
            obs['pose_wrt_start'] = pose_wrt_start

    f1_rgb = cam1_imgs[-1]
    f2_rgb = cam2_imgs[-1]
    f1_bgr = cv2.cvtColor(f1_rgb, cv2.COLOR_RGB2BGR)
    f2_bgr = cv2.cvtColor(f2_rgb, cv2.COLOR_RGB2BGR)
    return obs, f1_rgb, f2_rgb, f1_bgr, f2_bgr, ts_sel


def reset_policy_attrs(policy):
    """Clear per-episode policy attributes."""
    for attr in ('_init_done', '_policy_step', '_prev_ee_pose', '_stall_count', '_last_target'):
        if hasattr(policy, attr):
            delattr(policy, attr)


def switch_to_teleop(state, policy, franka_api, server_ip, *, save_video=False):
    """Common code for switching back to teleop mode.
    Returns new prev_target_pose from current EE.
    """
    restore_precision_mode(server_ip)
    if save_video:
        state.video_recorder.stop_and_save()
    else:
        state.video_recorder.stop_and_delete()
    state.reset_task_state()
    reset_policy_attrs(policy)
    return np.array(franka_api.get_pose_sync(), dtype=np.float64)


def wait_for_result_key(state, config, client_socket, franka_api, key_queue):
    """Hold pose and wait for user to press s/f then e.
    Returns True if result recorded, False if escaped with t.
    """
    hold_pose = np.array(franka_api.get_pose_sync(), dtype=np.float64)
    decided = False
    while True:
        try:
            client_socket.sendall(json.dumps({
                'target_pose': hold_pose.tolist(),
                'timestamp': time.time(),
                'gripper_command': 'keep', 'reset': False
            }, separators=(',', ':')).encode('utf-8') + b'\n')
        except Exception:
            pass
        ek = check_keyboard_input(state, key_queue)
        if not decided:
            if ek == 's':
                state.task_state = 'SUCCESS'
                save_trial_result(success=True, state=state, config=config)
                decided = True
                print("[RESULT] SUCCESS — 'e'를 눌러 종료")
            elif ek == 'f':
                state.task_state = 'FAIL'
                save_trial_result(success=False, state=state, config=config)
                decided = True
                print("[RESULT] FAIL — 'e'를 눌러 종료")
            elif ek == 't':
                return False
        else:
            if ek == 'e':
                return True
        time.sleep(0.01)


def wait_for_exit_key(state, client_socket, franka_api, key_queue):
    """Hold current pose and wait for 'e' key."""
    hold_pose = np.array(franka_api.get_pose_sync(), dtype=np.float64)
    while True:
        try:
            send_pose_command(client_socket, hold_pose)
        except Exception:
            pass
        if check_keyboard_input(state, key_queue) == 'e':
            break
        time.sleep(0.01)


def handle_result_key(key, state, config, client_socket, franka_api, policy,
                      server_ip, config_path, ft_collector, hw_refs, key_queue):
    """Handle s/f key: save result, hold, wait for e, reset, switch to teleop.
    Returns (new_initial_pose, new_prev_target_pose).
    """
    state.task_state = 'SUCCESS' if key == 's' else 'FAIL'
    save_trial_result(success=(key == 's'), state=state, config=config)
    print(f"[RESULT] {state.task_state} — 'e'를 눌러 종료")
    wait_for_exit_key(state, client_socket, franka_api, key_queue)
    new_initial = compute_initial_pose(config_path, config)
    try:
        reset_to_home(client_socket, franka_api, new_initial, config, state,
                      hw_refs, ft_collector=ft_collector)
    except Exception:
        pass
    new_prev = switch_to_teleop(state, policy, franka_api, server_ip, save_video=True)
    return new_initial, new_prev


def run_inference(policy, obs_np, cfg, device, obs_pose_rep, state, config):
    """Run model forward pass and return (action_np, result_dict).
    Updates state.model_time_accum / model_call_count.
    """
    obs_dict_np = get_real_gumi_obs_dict(
        env_obs=obs_np, shape_meta=cfg.task.shape_meta,
        obs_pose_repr=obs_pose_rep)
    obs_dict = dict_apply(obs_dict_np,
                          lambda x: torch.from_numpy(x).unsqueeze(0).to(device))

    t_model_start = time.time()
    result = policy.predict_action(obs_dict)
    action = result['action_pred'][0].detach().to('cpu').numpy()

    t_model_end = time.time()
    state.model_time_accum += (t_model_end - t_model_start)
    state.model_call_count += 1

    if 'delta_p' in result:
        delta_p = result['delta_p'][0].detach().to('cpu').numpy()[0]
        action[-1, :3] += delta_p * 0.01

    return action, result


def detect_robot_movement(state, current_ee_pose):
    """Check if robot has started moving; if so, reset timer and step counter."""
    if not state.robot_moving and state.robot_start_ee_pose is not None:
        ee_dist = np.linalg.norm(current_ee_pose[:3] - state.robot_start_ee_pose[:3])
        if ee_dist > state.robot_move_threshold:
            state.robot_moving = True
            state.policy_start_time = time.time()
            state.total_inference_steps = 0
            print(f"\n[MOVE] 로봇 이동 감지! dist={ee_dist*1000:.1f}mm → 타이머/스텝 시작")


def warmup_policy(policy, state, config, camera, additional_cam, cam_lock,
                  ft_collector, last_executed_action, franka_api, client_socket,
                  gripper_state, cfg, obs_pose_rep, device):
    """Policy 첫 iteration: 현재 pose 고정 + warmup inference."""
    policy._init_done = True
    policy._policy_step = 0
    current_ee = franka_api.get_pose_sync()
    prev_target_pose = np.array(current_ee, dtype=np.float64)
    state.policy_start_pose = prev_target_pose.copy()
    try:
        send_pose_command(client_socket, current_ee, gripper=gripper_state)
    except Exception:
        pass
    policy.reset()

    # Measure gripping FT bias (학습 zarr의 compensate_gripping_force_per_episode와 동일)
    print("[FT BIAS] 안정화 대기 중 (2s)...")
    time.sleep(2.0)
    ft_buf, _ = ft_collector.window(config['ft_sensor']['buffer_length'])
    if len(ft_buf) >= 5:
        state.gripping_ft_bias = ft_buf[-30:].mean(axis=0).astype(np.float32)
    else:
        state.gripping_ft_bias = np.zeros(6, dtype=np.float32)
    print(f"[FT BIAS] gripping bias: F=[{state.gripping_ft_bias[0]:+.2f},{state.gripping_ft_bias[1]:+.2f},{state.gripping_ft_bias[2]:+.2f}]N  "
          f"T=[{state.gripping_ft_bias[3]:+.3f},{state.gripping_ft_bias[4]:+.3f},{state.gripping_ft_bias[5]:+.3f}]")

    # FT recalibration 시작 시점 기록 (학습의 FT[5:100] mean 방식 재현)
    ft_buf_now, ts_buf_now = ft_collector.window(1)
    if len(ts_buf_now) > 0:
        state.ft_recalib_start_time = ts_buf_now[-1]
    state.ft_recalib_done = False
    state.ft_recalib_buffer = []

    print("Warming up policy inference")
    obs_np, *_ = get_obs(state, config, camera, additional_cam, cam_lock,
                         ft_collector, last_executed_action)
    with torch.no_grad():
        obs_dict_np = get_real_gumi_obs_dict(
            env_obs=obs_np, shape_meta=cfg.task.shape_meta,
            obs_pose_repr=obs_pose_rep)
        obs_dict = dict_apply(obs_dict_np,
                              lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
        policy.predict_action(obs_dict)
    return prev_target_pose


def toggle_observe_mode(state, franka_api, config):
    """Observe 모드 ON/OFF 토글. state.observe_mode를 반전시킨다."""
    state.observe_mode = not state.observe_mode
    if state.observe_mode:
        state.observe_actions = []
        state.observe_freeze_pose = np.array(franka_api.get_pose_sync(), dtype=np.float64)
        ee_q = state.observe_freeze_pose[3:]
        ee_euler = Rotation.from_quat(ee_q).as_euler('xyz', degrees=True)
        print(f"\n{'='*60}")
        print(f"[OBSERVE] 관찰 모드 ON — 로봇 고정, 인퍼런스 계속")
        print(f"  EE pos:  x={state.observe_freeze_pose[0]:.4f} y={state.observe_freeze_pose[1]:.4f} z={state.observe_freeze_pose[2]:.4f}")
        print(f"  EE quat(xyzw): [{ee_q[0]:.4f}, {ee_q[1]:.4f}, {ee_q[2]:.4f}, {ee_q[3]:.4f}]")
        print(f"  EE euler(xyz°): roll={ee_euler[0]:+.1f}  pitch={ee_euler[1]:+.1f}  yaw={ee_euler[2]:+.1f}")
        ee_rot = Rotation.from_quat(ee_q)
        bx = ee_rot.apply([1, 0, 0])
        by = ee_rot.apply([0, 1, 0])
        bz = ee_rot.apply([0, 0, 1])
        print(f"  Body X → World: [{bx[0]:+.3f}, {bx[1]:+.3f}, {bx[2]:+.3f}]")
        print(f"  Body Y → World: [{by[0]:+.3f}, {by[1]:+.3f}, {by[2]:+.3f}]")
        print(f"  Body Z → World: [{bz[0]:+.3f}, {bz[1]:+.3f}, {bz[2]:+.3f}]")
        print(f"  'o' 키로 해제")
        print(f"{'='*60}")
    else:
        if state.observe_actions:
            print_observe_table(state.observe_actions, show_last=30)
            save_observe_csv(state.observe_actions, state.observe_freeze_pose, config)
        if state.observe_fig is not None:
            import matplotlib.pyplot as _plt
            _plt.close(state.observe_fig)
            state.observe_fig = None
        print(f"\n[OBSERVE] 관찰 모드 OFF — 로봇 제어 재개 ({len(state.observe_actions)}회 수집)")


def check_ft_contact(state, ft_collector, config=None):
    """FT contact detection. force threshold 기반으로 접촉 감지."""
    ft_buf_check, _ = ft_collector.window(1)
    if len(ft_buf_check) > 0:
        state.ft_latest = ft_buf_check[-1]

    # baseline 대비 force 변화량
    force_delta = state.ft_latest[:3] - state.gripping_ft_bias[:3]
    ft_force_norm = float(np.linalg.norm(force_delta))

    thresholds = np.array([3.0, 3.0, 3.0])

    contact_detected = any(abs(force_delta[i]) > thresholds[i] for i in range(3))

    if contact_detected and not state.ft_contact_active:
        state.ft_contact_active = True
        state.ft_time_start = time.time()
        if state.task_state == 'REACHING':
            state.task_state = 'CONTACT'
            state.contact_step = state.total_inference_steps
            state.contact_time = time.time() - state.policy_start_time if state.policy_start_time else 0.0
            exceeded = [f"F{ax}={force_delta[i]:+.2f}(>{thresholds[i]:.1f})"
                        for i, ax in enumerate(['x','y','z']) if abs(force_delta[i]) > thresholds[i]]
            print(f"\n[CONTACT] 접촉 감지! {' '.join(exceeded)} |dF|={ft_force_norm:.2f}N | "
                  f"Reaching: {state.contact_time:.2f}s / {state.contact_step} steps")


def check_quat_jump(current_ee_pose, trajectory):
    """Debug: quaternion 점프 감지 경고 출력."""
    if len(trajectory) <= 1:
        return
    q_ee = current_ee_pose[3:]
    q_t0 = trajectory[0][3:]
    q_tn = trajectory[-1][3:]
    dot_ee_t0 = np.dot(q_ee, q_t0)
    ang_ee_t0 = np.degrees(2 * np.arccos(np.clip(abs(dot_ee_t0), 0, 1)))
    ang_t0_tn = np.degrees(2 * np.arccos(np.clip(abs(np.dot(q_t0, q_tn)), 0, 1)))
    if ang_ee_t0 > 10:
        print(f"\n[QUAT WARN] EE→traj[0] gap={ang_ee_t0:.1f}° "
              f"ee_q={q_ee} traj0_q={q_t0}")
    if ang_t0_tn > 30:
        print(f"\n[QUAT WARN] traj[0]→traj[-1] span={ang_t0_tn:.1f}° "
              f"traj0_q={q_t0} trajN_q={q_tn}")


def cleanup_session(gui_thread, state, ft_collector, ft_reader, imu_source,
                    franka_api, client_socket, server_ip, old_tty_settings,
                    camera=None, additional_cam=None):
    """finally 블록: GUI 중지, 센서 정리, 카메라 해제, 소켓 종료, tty 복원."""
    if gui_thread is not None:
        gui_thread.stop()
    state.video_recorder.stop_and_save()
    # imu_source는 카메라 객체이므로 별도 stop 안 함 (카메라 close에서 처리)
    for cleanup_fn in [ft_collector.stop, ft_reader.stop]:
        try:
            cleanup_fn()
        except Exception:
            pass
    # 카메라 해제 (Ctrl+C 시 device busy 방지)
    for cam in (camera, additional_cam):
        if cam is not None:
            try:
                cam.close()
            except Exception:
                pass
    try:
        curr_pose = franka_api.get_pose_sync()
        if curr_pose is not None:
            send_pose_command(client_socket, curr_pose)
            time.sleep(0.1)
    except Exception as e:
        print(f"[종료] 정지 명령 전송 실패: {e}")
    restore_precision_mode(server_ip)
    client_socket.close()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass
    if old_tty_settings is not None:
        try:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty_settings)
        except Exception:
            pass
    print("\n서버 연결을 종료합니다.")
