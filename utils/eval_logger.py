"""Eval logging: FT trace, diffusion intermediates, trial results, debug CSVs."""
import os
import csv
import time
from datetime import datetime
import numpy as np
import torch


def save_ft_data(ft_collector, filename="ft_trace.csv"):
    ft_collector.stop()
    with ft_collector._lock:
        ft_list = list(ft_collector.buf)
        ts_list = list(ft_collector.ts_buf)
    if not ft_list:
        return
    arr_ts = np.array(ts_list, dtype=np.float64).reshape(-1, 1)
    arr_ft = np.vstack(ft_list).astype(np.float32)
    np.savetxt(filename, np.hstack([arr_ts, arr_ft]),
               delimiter=",", header="timestamp,fx,fy,fz,tx,ty,tz", comments="")


def save_diffusion_intermediates(state):
    if not state.diffusion_intermediates:
        return
    ts_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = os.path.join(state.output_dir, f"diffusion_intermediates_{ts_str}.pt")
    torch.save(state.diffusion_intermediates, fname)
    print(f"[LOG] Diffusion intermediates 저장: {fname} ({len(state.diffusion_intermediates)} inference steps)")
    state.diffusion_intermediates = []


def save_trial_result(success, state, config):
    fname = os.path.join(state.output_dir, "result.csv")
    write_header = not os.path.exists(fname)
    now = time.time()
    total_time = (now - state.policy_start_time) if state.policy_start_time else 0.0
    action_hz = state.total_inference_steps / total_time if total_time > 0 else 0.0
    model_hz = state.model_call_count / state.model_time_accum if state.model_time_accum > 0 else 0.0

    reach_time = state.contact_time if state.contact_time is not None else total_time
    reach_steps = state.contact_step if state.contact_step is not None else state.total_inference_steps

    if state.insert_time is not None:
        cont_time, cont_steps = state.insert_time, state.insert_step or 0
    elif state.ft_time_start is not None:
        cont_time = now - state.ft_time_start
        cont_steps = state.total_inference_steps - (state.contact_step or 0)
    else:
        cont_time, cont_steps = 0.0, 0

    row = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'mode_label': state.gui_mode_label,
        'result': 'SUCCESS' if success else 'FAIL',
        'action_hz': f"{action_hz:.2f}",
        'model_hz': f"{model_hz:.2f}",
        'total_time_s': f"{total_time:.3f}",
        'total_inference_steps': state.total_inference_steps,
        'reaching_time_s': f"{reach_time:.3f}",
        'reaching_steps': reach_steps,
        'contact_time_s': f"{cont_time:.3f}",
        'contact_steps': cont_steps,
    }
    with open(fname, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)

    label = 'SUCCESS' if success else 'FAIL'
    print(f"\n[RESULT] {label} | total={total_time:.1f}s ({state.total_inference_steps} steps) | "
          f"reach={reach_time:.1f}s ({reach_steps}) | contact={cont_time:.1f}s ({cont_steps}) | "
          f"Action={action_hz:.1f}Hz Model={model_hz:.1f}Hz")
    print(f"[RESULT] 결과 저장: {fname}")


def log_debug_tracking(policy_step, action_mode, current_ee_pose, this_target_pose,
                       transformed_action, state, config, raw_action=None):
    """Tracking CSV 한 줄 기록."""
    if not config.get('debug', {}).get('save_debug_csv', False):
        return
    path = os.path.join(state.output_dir, 'debug_tracking.csv')
    if policy_step == 1:
        with open(path, 'w') as f:
            f.write('step,time,action_mode,'
                    'ee_x,ee_y,ee_z,ee_qw,ee_qx,ee_qy,ee_qz,'
                    'tgt_x,tgt_y,tgt_z,tgt_q0,tgt_q1,tgt_q2,tgt_q3,'
                    'tf_pos_x,tf_pos_y,tf_pos_z,'
                    'raw_dx,raw_dy,raw_dz,raw_q0,raw_q1,raw_q2,raw_q3,'
                    'ft_fx,ft_fy,ft_fz,ft_tx,ft_ty,ft_tz,'
                    'ca_active,ca_fb_x,ca_fb_y,ca_fb_z,ca_off_x,ca_off_y,ca_off_z\n')
    try:
        raw_vals = [0.0] * 7
        if raw_action is not None:
            raw_vals = list(raw_action[:min(7, len(raw_action))])
            raw_vals += [0.0] * (7 - len(raw_vals))
        ft_vals = list(state.ft_model_input[:6])
        ca_vals = [1 if state.ca_active else 0,
                   *state.ca_force_body.tolist(),
                   *state.ca_offset_mm.tolist()]
        vals = [policy_step, f"{time.time():.3f}", action_mode,
                *[f"{v:.6f}" for v in current_ee_pose],
                *[f"{v:.6f}" for v in this_target_pose],
                *[f"{v:.6f}" for v in transformed_action[:3]],
                *[f"{v:.6f}" for v in raw_vals],
                *[f"{v:.6f}" for v in ft_vals],
                *[f"{v:.6f}" for v in ca_vals]]
        with open(path, 'a') as f:
            f.write(','.join(str(v) for v in vals) + '\n')
    except Exception:
        pass


def log_debug_rel_to_start(policy_step, selected_action, action_8d, current_ee_pose,
                           transformed_action, cur_rot, action_rot, target_euler,
                           this_target_pose, state, config):
    """rel_to_start 상세 디버그 CSV."""
    if not config.get('debug', {}).get('save_debug_csv', False):
        return
    path = os.path.join(state.output_dir, 'debug_rel_to_start.csv')
    if policy_step == 1:
        with open(path, 'w') as f:
            f.write('step,time,'
                    'raw_ax0,raw_ax1,raw_ax2,raw_ax3,raw_ax4,raw_ax5,raw_ax6,'
                    'a8d_pos0,a8d_pos1,a8d_pos2,a8d_qx,a8d_qy,a8d_qz,a8d_qw,'
                    'ee_x,ee_y,ee_z,ee_qw,ee_qx,ee_qy,ee_qz,'
                    'tf_pos0,tf_pos1,tf_pos2,tf_qx,tf_qy,tf_qz,tf_qw,'
                    'cur_euler_x,cur_euler_y,cur_euler_z,'
                    'act_euler_x,act_euler_y,act_euler_z,'
                    'tgt_euler_x,tgt_euler_y,tgt_euler_z,'
                    'final_x,final_y,final_z,final_qw,final_qx,final_qy,final_qz\n')
    try:
        vals = [policy_step, f"{time.time():.3f}",
                *[f"{v:.6f}" for v in selected_action],
                *[f"{v:.6f}" for v in action_8d[:7]],
                *[f"{v:.6f}" for v in current_ee_pose],
                *[f"{v:.6f}" for v in transformed_action],
                *[f"{v:.3f}" for v in cur_rot.as_euler('xyz', degrees=True)],
                *[f"{v:.3f}" for v in action_rot.as_euler('xyz', degrees=True)],
                *[f"{v:.3f}" for v in target_euler],
                *[f"{v:.6f}" for v in this_target_pose]]
        with open(path, 'a') as f:
            f.write(','.join(str(v) for v in vals) + '\n')
    except Exception:
        pass
