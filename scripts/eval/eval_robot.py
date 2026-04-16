import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import atexit
import signal
import time
import socket
import termios
import tty
import threading
import queue
from multiprocessing.managers import SharedMemoryManager

import click
import numpy as np
import scipy.spatial.transform as st
import yaml
import torch

from utils.spacemouse_device import SpacemouseDevice as Spacemouse
from utils.precise_sleep import precise_wait
from utils.flask_pose_client import connect_to_server, send_pose_command

from utils.eval_state import EvalState
from utils.model_loader import load_model
from utils.hardware_init import init_hardware, compute_initial_pose, reset_to_home, calibrate_startup
from utils.gui_thread import GuiThread, check_keyboard_input
from utils.eval_helpers import (
    prefill_obs_history, get_obs, reset_policy_attrs, switch_to_teleop,
    handle_result_key, wait_for_result_key,
    run_inference, detect_robot_movement, warmup_policy,
    toggle_observe_mode, check_ft_contact, check_quat_jump, cleanup_session
)
from utils.action_builder import build_action_queue, build_interpolated_trajectory
from utils.eval_logger import log_debug_tracking
from utils.observe import observe_body_to_world, print_observe_table, update_observe_3d


def load_config(config_path='inference_config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


@click.command()
@click.option('--config_path', '-c', default='inference_config.yaml', type=str, help="설정 파일 경로")
@click.option('--model_checkpoint_path', '-mcp', default='', type=str, help="모델 체크포인트 경로")
def main(config_path, model_checkpoint_path):
    config = load_config(config_path)
    S = EvalState(config)

    cam_lock = threading.Lock()
    key_queue = queue.Queue()

    if S.headless:
        print("[INFO] Headless mode: cv2 창 비활성화, 키 입력은 터미널에서")

    server_ip = config['robot']['server_ip']
    server_port = config['robot']['server_port']
    frequency = config['control']['frequency']
    dt = 1 / frequency

    franka_api, ft_reader, ft_collector, imu_source, gravity_compensator, camera, additional_cam = init_hardware(config)

    # Ctrl+C / kill 시에도 카메라 해제 보장
    def _emergency_cam_cleanup(*_args):
        for cam in (camera, additional_cam):
            if cam is not None:
                try:
                    cam.close()
                except Exception:
                    pass
    atexit.register(_emergency_cam_cleanup)
    signal.signal(signal.SIGTERM, lambda *a: (_emergency_cam_cleanup(), sys.exit(0)))

    hw_refs = {
        'gravity_comp': gravity_compensator,
        'imu_source': imu_source,
        'ft_reader': ft_reader,
    }

    try:
        current_pose = franka_api.get_pose_sync()
    except Exception:
        pass
    print(f"\n=== 초기 로봇 자세 (5-4-3) ===")
    print(f"Initial pose: {list(current_pose)}")

    initial_pose = compute_initial_pose(config_path, config)
    sync_mode = config.get('control', {}).get('sync_mode', False)
    client_socket = connect_to_server(server_ip, server_port, sync=sync_mode)
    if client_socket is None:
        return

    with SharedMemoryManager() as shm_manager:

        with Spacemouse(shm_manager=shm_manager) as sm:
            ckpt_path = model_checkpoint_path or config.get('paths', {}).get('model_checkpoint') or config.get('model_checkpoint_path', '')
            if not ckpt_path:
                raise ValueError("모델 체크포인트 경로가 없습니다.")

            policy, cfg, rotation_repr, device = load_model(ckpt_path, config, S)
            obs_pose_rep = cfg.task.pose_repr.obs_pose_repr

            S.build_output_dir(config_path, config)
            print(f"[OUTPUT] 저장 폴더: {os.path.abspath(S.output_dir)}")

            _gui_thread = None
            _old_tty_settings = None
            try:
                target_pose = initial_pose.copy()
                current_pos = initial_pose[:3].copy()
                current_rot = st.Rotation.from_quat(initial_pose[3:])

                calibrate_startup(client_socket, initial_pose, config, S, hw_refs, ft_collector)

                gripper_state = 'keep'
                last_btn0_state = False

                if not S.headless:
                    _gui_thread = GuiThread(
                        camera=camera, additional_cam=additional_cam,
                        cam_lock=cam_lock, key_queue=key_queue,
                        state=S, config=config, ft_collector=ft_collector)
                    _gui_thread.start()
                    print("[GUI] 실시간 GUI thread 시작")

                print("""
                === 제어 모드 ===
                t: Teleop 모드
                p: Policy 모드
                q: 프로그램 종료
                r: 로봇 위치 리셋
                """)
                mode = S.current_mode = 'teleop'
                print("\nteleop 모드로 시작합니다...")

                if S.headless:
                    _old_tty_settings = termios.tcgetattr(sys.stdin)
                    tty.setcbreak(sys.stdin.fileno())
                    print("[INFO] 키 입력은 터미널에서 (q/s/f/t/p/r)")
                else:
                    print("[INFO] 키 입력은 OpenCV 창에 포커스를 맞춘 후 사용하세요")

                iter_idx = 0
                last_executed_action = np.zeros(8, dtype=np.float32)
                last_executed_action[7] = 1.0
                prev_target_pose = initial_pose.copy()
                _pending_key = None

                # ─── Main Loop ───
                while True:
                    cycle_start = time.monotonic()
                    key = _pending_key or check_keyboard_input(S, key_queue)
                    _pending_key = None

                    # ═══ Key Handling (shared) ═══
                    if key:
                        if key == 'q':
                            S.video_recorder.stop_and_save()
                            try:
                                reset_to_home(client_socket, franka_api, initial_pose,
                                              config, S, hw_refs)
                            except Exception:
                                pass
                            print("\n프로그램 종료")
                            break

                        elif key == 't':
                            prev_target_pose = switch_to_teleop(S, policy, franka_api, server_ip, save_video=False)
                            current_pos = prev_target_pose[:3].copy()
                            current_rot = st.Rotation.from_quat(prev_target_pose[3:])
                            mode = S.current_mode = 'teleop'
                            print("\n텔레오퍼레이션 모드로 전환 (타이머 리셋)")
                            continue

                        elif key in ('s', 'f'):
                            initial_pose, prev_target_pose = handle_result_key(
                                key, S, config, client_socket, franka_api, policy,
                                server_ip, config_path, ft_collector, hw_refs, key_queue)
                            mode = S.current_mode = 'teleop'
                            continue

                        elif key == 'p':
                            mode = S.current_mode = 'policy'
                            S.policy_start_time = None
                            S.task_state = 'REACHING'
                            S.total_inference_steps = 0
                            S.contact_step = S.contact_time = None
                            S.insert_step = S.insert_time = None
                            S.robot_moving = False
                            S.robot_start_ee_pose = np.array(franka_api.get_pose_sync(), dtype=np.float64)

                            S.video_recorder.start(S.output_dir, prefix=S.gui_mode_label.replace(' ', '_'))
                            print("\nPolicy 모드로 전환")
                            _cur_ee = franka_api.get_pose_sync()
                            prev_target_pose = np.array(_cur_ee, dtype=np.float64)
                            S.policy_start_pose = prev_target_pose.copy()
                            current_pos = np.array(_cur_ee[:3], dtype=np.float64)
                            current_rot = st.Rotation.from_quat(_cur_ee[3:])
                            prefill_obs_history(S, config, camera, additional_cam, cam_lock)
                            policy.reset()
                            reset_policy_attrs(policy)
                            continue

                        elif key == 'r':
                            initial_pose = compute_initial_pose(config_path, config)
                            try:
                                reset_to_home(client_socket, franka_api, initial_pose,
                                              config, S, hw_refs,
                                              ft_collector=ft_collector, gripper=gripper_state)
                            except (socket.error, BrokenPipeError) as e:
                                print(f"\n리셋 명령 전송 실패: {e}")
                            current_pos = initial_pose[:3].copy()
                            current_rot = st.Rotation.from_quat(initial_pose[3:])
                            target_pose[:3] = current_pos
                            target_pose[3:] = current_rot.as_quat()
                            if mode == 'policy':
                                prev_target_pose = initial_pose.copy()
                                reset_policy_attrs(policy)
                                prefill_obs_history(S, config, camera, additional_cam, cam_lock)
                            continue

                        else:
                            _pending_key = key

                    # ═══ Teleop Mode ═══
                    if mode == 'teleop':
                        sm_state = sm.get_motion_state_transformed()
                        btn0 = sm.is_button_pressed(0)
                        if btn0 and not last_btn0_state:
                            gripper_state = 'close' if gripper_state == 'open' else 'open'
                        last_btn0_state = btn0

                        pos_scale = config['robot']['max_pos_speed'] * dt
                        rot_scale = config['robot']['max_rot_speed'] * dt
                        current_pos += sm_state[:3] * pos_scale
                        drot_xyz = sm_state[3:] * rot_scale
                        if np.any(drot_xyz != 0):
                            current_rot = st.Rotation.from_euler('xyz', drot_xyz) * current_rot
                        target_pose[:3] = current_pos
                        target_pose[3:] = current_rot.as_quat()

                        # Update state from camera ring buffers for display
                        latest1 = camera.get_latest_rgb()
                        latest2 = additional_cam.get_latest_rgb()
                        if latest1 is not None and latest2 is not None:
                            S.prev_frames_cam1.append(latest1[1])
                            S.prev_frames_cam2.append(latest2[1])
                            S.prev_timestamps.append(latest1[0])

                        # EE pose buffer (for pose_wrt_start when switching to policy)
                        S.ee_pose_buffer.append((time.time(), target_pose.copy()))

                        ft_buf, _ = ft_collector.window(1)
                        if len(ft_buf) > 0:
                            S.ft_latest = ft_buf[-1]

                        try:
                            send_pose_command(client_socket, target_pose, gripper=gripper_state)
                        except (socket.error, BrokenPipeError) as e:
                            print(f"서버 연결 오류: {e}. 재연결 시도...")
                            client_socket.close()
                            client_socket = connect_to_server(server_ip, server_port, sync=sync_mode)
                            if client_socket is None:
                                time.sleep(0.1)
                                continue

                    # ═══ Policy Mode ═══
                    if mode == 'policy':
                        # Warmup
                        if not hasattr(policy, '_init_done'):
                            prev_target_pose = warmup_policy(
                                policy, S, config, camera, additional_cam, cam_lock,
                                ft_collector, last_executed_action, franka_api,
                                client_socket, gripper_state, cfg, obs_pose_rep, device)
                            continue

                        # Observation
                        obs_np, img1, img2, img1_view, img2_view, ft_ts = get_obs(
                            S, config, camera, additional_cam, cam_lock,
                            ft_collector, last_executed_action)

                        # Key during policy
                        kp = _pending_key or check_keyboard_input(S, key_queue)
                        _pending_key = None
                        if kp:
                            if kp == 'q':
                                S.video_recorder.stop_and_save()
                                try:
                                    reset_to_home(client_socket, franka_api, initial_pose,
                                                  config, S, hw_refs)
                                except Exception:
                                    pass
                                print("\n프로그램 종료")
                                break
                            elif kp == 't':
                                prev_target_pose = switch_to_teleop(S, policy, franka_api, server_ip, save_video=False)
                                current_pos = prev_target_pose[:3].copy()
                                current_rot = st.Rotation.from_quat(prev_target_pose[3:])
                                mode = S.current_mode = 'teleop'
                                print("\n텔레오퍼레이션 모드로 전환")
                                continue
                            elif kp in ('s', 'f'):
                                initial_pose, prev_target_pose = handle_result_key(
                                    kp, S, config, client_socket, franka_api, policy,
                                    server_ip, config_path, ft_collector, hw_refs, key_queue)
                                mode = S.current_mode = 'teleop'
                                continue
                            elif kp == 'o':
                                toggle_observe_mode(S, franka_api, config)
                                continue
                            elif kp == 'r':
                                initial_pose = compute_initial_pose(config_path, config)
                                try:
                                    reset_to_home(client_socket, franka_api, initial_pose,
                                                  config, S, hw_refs, ft_collector=ft_collector)
                                except (socket.error, BrokenPipeError) as e:
                                    print(f"\n리셋 명령 전송 실패: {e}")
                                prev_target_pose = initial_pose.copy()
                                reset_policy_attrs(policy)
                                prefill_obs_history(S, config, camera, additional_cam, cam_lock)
                                continue

                        current_ee_pose = np.array(franka_api.get_pose_sync(), dtype=np.float64)
                        S.ee_pose_buffer.append((time.time(), current_ee_pose.copy()))

                        # Gear insertion detection
                        z_thresh = config.get('robot', {}).get('gear_insertion_z_threshold', None)
                        if z_thresh is not None and not S.gear_inserted and current_ee_pose[2] < z_thresh:
                            S.gear_inserted = True
                            S.task_state = 'INSERTED'
                            S.insert_step = S.total_inference_steps - (S.contact_step or 0)
                            S.insert_time = (time.time() - S.ft_time_start) if S.ft_time_start else 0.0
                            print(f"\n[GEAR] 삽입 감지! EE z={current_ee_pose[2]:.4f} < {z_thresh}")
                            print("[GEAR] 로봇 정지. s(success) / f(fail) 을 눌러주세요.")
                            decided = wait_for_result_key(S, config, client_socket, franka_api, key_queue)
                            initial_pose = compute_initial_pose(config_path, config)
                            try:
                                reset_to_home(client_socket, franka_api, initial_pose,
                                              config, S, hw_refs, ft_collector=ft_collector)
                            except Exception:
                                pass
                            prev_target_pose = switch_to_teleop(S, policy, franka_api, server_ip, save_video=decided)
                            mode = S.current_mode = 'teleop'
                            print("\n텔레오퍼레이션 모드로 전환")
                            continue

                        check_ft_contact(S, ft_collector, config)

                        # ── Inference ──
                        action_mode = config['action'].get('action_mode', 'delta')
                        gripper_state = 'keep'

                        with torch.no_grad():
                            t_start = time.time()
                            action, result = run_inference(
                                policy, obs_np, cfg, device, obs_pose_rep, S, config)

                        # ── Build trajectory ──
                        action_queue = build_action_queue(action, rotation_repr, config)
                        data_hz = config['action'].get('data_hz', 7.5)
                        control_hz = config['control'].get('frequency', 200.0)
                        trajectory = build_interpolated_trajectory(
                            action_queue, prev_target_pose, current_ee_pose,
                            data_hz, control_hz, config)

                        if not hasattr(policy, '_policy_step'):
                            policy._policy_step = 0
                        policy._policy_step += 1
                        S.total_inference_steps += 1

                        S.last_raw_action = action_queue[-1].copy()
                        S.last_current_ee = current_ee_pose.copy()
                        S.last_target_pose = trajectory[-1].copy() if trajectory else None

                        detect_robot_movement(S, current_ee_pose)

                        t_end = time.time()
                        infer_ms = (t_end - t_start) * 1000
                        n_traj = len(trajectory)
                        traj_sec = n_traj / control_hz
                        print(f"\r[INTERP] inference {infer_ms:.0f}ms → "
                              f"trajectory {n_traj} pts ({traj_sec:.2f}s @ {control_hz:.0f}Hz)",
                              end='', flush=True)

                        check_quat_jump(current_ee_pose, trajectory)

                        # ── Trajectory 전송 ──
                        if S.observe_mode:
                            chunk_body_pos = np.array([a[:3] for a in action_queue])
                            chunk_body_rot = np.array([a[3:7] for a in action_queue])
                            chunk_world_pos = observe_body_to_world(chunk_body_pos, current_ee_pose, config)
                            S.observe_actions.append((chunk_body_pos, chunk_world_pos, chunk_body_rot))
                            print_observe_table(S.observe_actions)
                            update_observe_3d(S.observe_actions, S)
                            try:
                                send_pose_command(client_socket, S.observe_freeze_pose,
                                                  gripper=gripper_state, task_state=S.task_state)
                            except (socket.error, BrokenPipeError):
                                pass
                        else:
                            traj_dt = 1.0 / control_hz
                            traj_start = time.monotonic()
                            for ti, traj_pose in enumerate(trajectory):
                                target_time = traj_start + ti * traj_dt
                                now = time.monotonic()
                                if target_time > now:
                                    precise_wait(target_time)
                                try:
                                    send_pose_command(client_socket, traj_pose, gripper=gripper_state,
                                                      task_state=S.task_state)
                                except (socket.error, BrokenPipeError):
                                    try:
                                        client_socket.close()
                                        client_socket = connect_to_server(server_ip, server_port, sync=sync_mode)
                                    except Exception:
                                        pass
                                    break
                                if ti % 50 == 0:
                                    _k = check_keyboard_input(S, key_queue)
                                    if _k in ('q', 't', 's', 'f'):
                                        _pending_key = _k
                                        break

                            prev_target_pose = trajectory[-1].copy()
                            last_executed_action = action_queue[-1].copy()
                            policy._last_action = last_executed_action.copy()
                        if not S.observe_mode:
                            now = time.time()
                            # model inference time only (t_start set before run_inference)
                            model_ms = S.model_time_accum / max(1, S.model_call_count) * 1000
                            S.gui_inference_hz = 1000.0 / model_ms if model_ms > 0 else 0
                            # full cycle: obs→inference→trajectory→next obs
                            S.gui_cycle_ms = (now - t_start) * 1000

                        if config.get('debug', {}).get('save_debug_csv', False):
                            log_debug_tracking(
                                policy._policy_step, action_mode,
                                current_ee_pose, trajectory[0],
                                np.zeros(3), S, config,
                                raw_action=action_queue[0])
                            log_debug_tracking(
                                policy._policy_step, action_mode,
                                current_ee_pose, trajectory[-1],
                                trajectory[-1][:3] - trajectory[0][:3], S, config,
                                raw_action=action_queue[-1])

                    # Cycle timing
                    cycle_end = cycle_start + dt
                    if cycle_end - time.monotonic() > 0:
                        precise_wait(cycle_end)
                    iter_idx += 1

            finally:
                cleanup_session(_gui_thread, S, ft_collector, ft_reader, imu_source,
                                franka_api, client_socket, server_ip, _old_tty_settings,
                                camera=camera, additional_cam=additional_cam)

if __name__ == "__main__":
    main()
