"""Action processing: coordinate transform, action queue, interpolated trajectory, target pose."""
import time
import numpy as np
from scipy.spatial.transform import Rotation, Slerp as ScipySlerp

from utils.real_inference_util import convert_action_10d_to_8d, convert_action_7d_to_8d
from utils.pose_util import quat_xyzw_to_wxyz


def transform_action_to_world(pose, current_ee_pose, ft_data, config, state):
    """Coordinate transform: model output (body-frame delta) → world-frame delta.

    Legacy transform: R_mat_pos/R_mat_rot + per-axis scale + rot.inv().apply().
    ArUco body frame → Franka world frame.
    """
    current_quat = np.roll(current_ee_pose[3:], -1)  # [w,x,y,z] -> [x,y,z,w]
    current_rot = Rotation.from_quat(current_quat)

    R_mat_pos = np.array(config['coordinate_transform']['R_mat_pos'])
    R_mat_rot = np.array(config['coordinate_transform']['R_mat_rot'])

    rel_pos = R_mat_pos @ pose[:3]

    rotvec = Rotation.from_quat(np.array(pose[3:7])).as_rotvec()
    R_new = R_mat_rot @ Rotation.from_rotvec(rotvec).as_matrix() @ R_mat_rot.T
    rel_quat = Rotation.from_matrix(R_new).as_quat()

    rel_pos = current_rot.inv().apply(rel_pos)

    # Contact assist: FT 변화량(baseline 대비) 축별 offset 추가
    ca = config.get('contact_assist', {})
    if ca.get('enabled', False) and ft_data is not None:
        ft_latest = ft_data[-1] if ft_data.ndim == 2 else ft_data
        force_body = ft_latest[:3] - state.ca_ft_baseline[:3]

        threshold_cfg = ca.get('force_threshold', 3.0)
        if isinstance(threshold_cfg, (list, tuple)):
            thresholds = np.array(threshold_cfg, dtype=np.float64)
        else:
            thresholds = np.array([threshold_cfg] * 3, dtype=np.float64)

        force_norm = np.linalg.norm(force_body)
        state.ca_force_body = force_body.copy()

        hysteresis = ca.get('hysteresis', 0.5)
        any_exceeded = any(abs(force_body[i]) > thresholds[i] for i in range(3))
        any_below = all(abs(force_body[i]) < thresholds[i] * hysteresis for i in range(3))

        if not hasattr(state, '_ca_contact_on'):
            state._ca_contact_on = False
        if state._ca_contact_on:
            if any_below:
                state._ca_contact_on = False
        else:
            if any_exceeded:
                state._ca_contact_on = True

        if state._ca_contact_on:
            ca_mode = ca.get('mode', 'fixed_direction')

            if ca_mode == 'force_direction':
                gain = ca.get('gain', 0.0002)
                max_offset = ca.get('max_offset', 0.005)
                if force_norm > 0.01:
                    force_dir = force_body / force_norm
                    offset_mag = min(force_norm * gain, max_offset)
                    offset_world = current_rot.inv().apply(force_dir * offset_mag)
                else:
                    offset_world = np.zeros(3)
            else:
                insert_dir = np.array(ca.get('insert_direction', [1, 0, 0]), dtype=np.float64)
                insert_dir = insert_dir / np.linalg.norm(insert_dir)
                insert_offset = ca.get('insert_offset', 0.003)
                offset_world = insert_dir * insert_offset

            rel_pos += offset_world
            state.ca_active = True
            state.ca_force_norm = force_norm
            state.ca_offset_mm = offset_world * 1000
            state.ca_force_dir_world = offset_world / np.linalg.norm(offset_world) if np.linalg.norm(offset_world) > 1e-6 else np.zeros(3)
            state.ca_last_applied_time = time.time()
            state.ca_apply_count += 1
        else:
            state.ca_active = False
            state.ca_force_norm = force_norm
            state.ca_offset_mm = np.zeros(3)
    else:
        state.ca_active = False

    return np.concatenate([rel_pos, rel_quat])


def build_action_queue(action, rotation_repr, config):
    """Convert raw action chunk (N, D) into a list of individual action_8d."""
    chunk_steps = config['action'].get('chunk_steps', 1)
    chunk_steps = min(chunk_steps, action.shape[0])

    queue = []
    for ai in range(chunk_steps):
        if rotation_repr == 'axis_angle':
            a8d = convert_action_7d_to_8d(action[ai])
        else:
            a8d = convert_action_10d_to_8d(action[ai])
        queue.append(a8d)
    return queue


def build_interpolated_trajectory(action_queue, base_pose, current_ee_pose, ft_now,
                                  data_hz, control_hz, config, state):
    """Build a dense trajectory by interpolating between action waypoints.

    Returns list of target poses (each shape (7,): [x,y,z, qx,qy,qz,qw]).
    """
    action_mode = config['action'].get('action_mode', 'delta')
    step_duration = 1.0 / data_hz
    control_dt = 1.0 / control_hz
    steps_per_action = max(1, int(round(step_duration / control_dt)))

    delta_base = config['action'].get('delta_base', 'prev_target')
    if action_mode == 'delta' and delta_base == 'current_ee':
        start_pose = current_ee_pose.copy()
    else:
        start_pose = base_pose.copy()

    if action_mode != 'delta':
        ee_quat_xyzw = np.array([current_ee_pose[4], current_ee_pose[5],
                                  current_ee_pose[6], current_ee_pose[3]])
        identity_wp = current_ee_pose.copy()
        identity_wp[3:] = ee_quat_xyzw
        waypoints = [identity_wp.copy()]
    else:
        waypoints = [start_pose.copy()]
    current_wp = waypoints[0].copy()

    for action_8d in action_queue:
        transformed = transform_action_to_world(action_8d, current_ee_pose, ft_now, config, state)
        if action_mode == 'delta':
            wp = current_wp.copy()
            wp[:3] += transformed[:3]
            base_rot = Rotation.from_quat(wp[3:])
            delta_rot = Rotation.from_quat(transformed[3:])
            wp[3:] = (base_rot * delta_rot).as_quat()
        else:
            wp = current_ee_pose.copy()
            wp[:3] += transformed[:3]
            cur_rot = Rotation.from_quat(ee_quat_xyzw)
            action_rot = Rotation.from_quat(transformed[3:7])
            wp[3:] = (cur_rot * action_rot).as_quat()
        current_wp = wp.copy()
        waypoints.append(wp)

    # Quaternion sign consistency (double-cover fix)
    for i in range(1, len(waypoints)):
        if np.dot(waypoints[i-1][3:], waypoints[i][3:]) < 0:
            waypoints[i][3:] *= -1

    # Interpolate between waypoints at control_hz
    trajectory = []
    for i in range(len(waypoints) - 1):
        wp_start = waypoints[i]
        wp_end = waypoints[i + 1]
        rot_start = Rotation.from_quat(wp_start[3:])
        rot_end = Rotation.from_quat(wp_end[3:])
        slerp = ScipySlerp([0.0, 1.0], Rotation.concatenate([rot_start, rot_end]))

        for j in range(steps_per_action):
            alpha = j / steps_per_action
            pose = np.zeros(7, dtype=np.float64)
            pose[:3] = wp_start[:3] + alpha * (wp_end[:3] - wp_start[:3])
            pose[3:] = slerp(alpha).as_quat()
            trajectory.append(pose)

    trajectory.append(waypoints[-1].copy())

    if action_mode != 'delta':
        for pose in trajectory:
            pose[3:] = quat_xyzw_to_wxyz(pose[3:])

    return trajectory


def compute_target_pose(action_8d, action_mode, current_ee_pose, prev_target_pose,
                        ft_now, policy, config, state):
    """Compute target pose from action. Returns (this_target_pose, transformed_action, debug_info)."""
    transformed_action = transform_action_to_world(action_8d, current_ee_pose, ft_now, config, state)

    debug_info = {}

    if action_mode == 'rel_to_start':
        this_target = np.zeros(7, dtype=np.float64)
        this_target[:3] = current_ee_pose[:3] + transformed_action[:3]

        ee_quat_xyzw = np.array([current_ee_pose[4], current_ee_pose[5],
                                  current_ee_pose[6], current_ee_pose[3]])
        cur_rot = Rotation.from_quat(ee_quat_xyzw)
        action_rot = Rotation.from_quat(transformed_action[3:7])
        target_rot = cur_rot * action_rot
        this_target[3:] = target_rot.as_quat()

        # Stall detection
        if not hasattr(policy, '_prev_ee_pose'):
            policy._prev_ee_pose = current_ee_pose.copy()
            policy._stall_count = 0

        ee_moved = np.linalg.norm(current_ee_pose[:3] - policy._prev_ee_pose[:3])
        if ee_moved < 0.0002:
            policy._stall_count += 1
            if policy._stall_count > 5 and hasattr(policy, '_last_target'):
                prev_gap = np.linalg.norm(policy._last_target[:3] - current_ee_pose[:3])
                curr_gap = np.linalg.norm(this_target[:3] - current_ee_pose[:3])
                if prev_gap > curr_gap:
                    this_target[:3] = policy._last_target[:3]
        else:
            policy._stall_count = 0

        policy._prev_ee_pose = current_ee_pose.copy()
        policy._last_target = this_target.copy()
        debug_info['cur_rot'] = cur_rot
        debug_info['action_rot'] = action_rot
    else:
        delta_base = config['action'].get('delta_base', 'prev_target')
        if delta_base == 'current_ee':
            base_pose = current_ee_pose.copy()
        else:
            base_pose = prev_target_pose.copy()
        this_target = base_pose.copy()
        this_target[:3] += transformed_action[:3]
        base_rot = Rotation.from_quat(base_pose[3:])
        delta_rot = Rotation.from_quat(transformed_action[3:])
        this_target[3:] = (base_rot * delta_rot).as_quat()

    # Euler angle 안전 클램프 (pitch 제한)
    target_euler = Rotation.from_quat(this_target[3:]).as_euler('xyz', degrees=True)
    if target_euler[1] < -35:
        target_euler[1] = -35
    new_quat = Rotation.from_euler('xyz', target_euler, degrees=True).as_quat()
    this_target[3:] = new_quat / np.linalg.norm(new_quat)

    debug_info['target_euler'] = target_euler

    if action_mode == 'rel_to_start':
        this_target[3:] = quat_xyzw_to_wxyz(this_target[3:])

    return this_target, transformed_action, debug_info
