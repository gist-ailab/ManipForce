"""Eval GUI display: FT graph, action panel, model input thumbnails, prediction overlay."""
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image


def resize_with_padding(img, target_size=224):
    h, w = img.shape[:2]
    aspect = w / h
    if aspect > 1:
        new_w, new_h = target_size, int(target_size / aspect)
    else:
        new_h, new_w = target_size, int(target_size * aspect)

    pil_img = Image.fromarray(img)
    resized = pil_img.resize((new_w, new_h))
    new_img = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    new_img.paste(resized, ((target_size - new_w) // 2, (target_size - new_h) // 2))
    return np.array(new_img)


def draw_ft_graph(ft_history, ft_timestamps, width=640, graph_h=180,
                  display_seconds=5.0, contact_threshold=5.0,
                  model_ft_history=None):
    """Draw real-time FT force/torque graph. Top half = forces, bottom half = torques."""
    force_h = graph_h // 2
    torque_h = graph_h - force_h
    canvas = np.zeros((graph_h, width, 3), dtype=np.uint8)

    if len(ft_history) < 2:
        cv2.putText(canvas, "FT: collecting...", (10, graph_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        return canvas

    ft_arr = np.array(ft_history, dtype=np.float32)
    ts_arr = np.array(ft_timestamps, dtype=np.float64)

    t_now = ts_arr[-1]
    t_start = t_now - display_seconds
    mask = ts_arr >= t_start
    ft_arr, ts_arr = ft_arr[mask], ts_arr[mask]
    if len(ft_arr) < 2:
        return canvas

    margin_l, margin_r = 50, 10
    plot_w = width - margin_l - margin_r

    def time_to_x(t):
        return int(margin_l + (t - t_start) / display_seconds * plot_w)

    def _polyline(values, val_to_y):
        step = max(1, len(values) // 400)
        pts = []
        for i in range(0, len(values), step):
            pts.append([time_to_x(ts_arr[i]), val_to_y(values[i])])
        return np.array(pts, dtype=np.int32)

    # Force graph (top half)
    f_range = max(float(np.abs(ft_arr[:, :3]).max()), 3.0) * 1.2

    def val_to_y_f(v):
        return int((1 - (v + f_range) / (2 * f_range)) * (force_h - 2))

    y0_f = val_to_y_f(0)
    cv2.line(canvas, (margin_l, y0_f), (width - margin_r, y0_f), (60, 60, 60), 1)
    for thr in (contact_threshold, -contact_threshold):
        yt = val_to_y_f(thr)
        cv2.line(canvas, (margin_l, yt), (width - margin_r, yt), (0, 0, 100), 1)

    force_colors = [(0, 0, 255), (0, 200, 0), (255, 100, 0)]
    for ch, col in enumerate(force_colors):
        pts = _polyline(ft_arr[:, ch], val_to_y_f)
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts], False, col, 1, cv2.LINE_AA)

    f_norm = np.linalg.norm(ft_arr[:, :3], axis=1)
    pts_norm = _polyline(f_norm, val_to_y_f)
    if len(pts_norm) >= 2:
        cv2.polylines(canvas, [pts_norm], False, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.putText(canvas, f"+/-{f_range:.0f}N", (2, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
    for i, (lbl, col) in enumerate(zip(['Fx', 'Fy', 'Fz', '|F|'],
                                        force_colors + [(200, 200, 200)])):
        cv2.putText(canvas, lbl, (width - 95 + i * 22, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1)
    fn = float(np.linalg.norm(ft_arr[-1, :3]))
    cv2.putText(canvas, f"|F|={fn:.1f}N", (2, force_h - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # Divider
    cv2.line(canvas, (0, force_h), (width, force_h), (80, 80, 80), 1)

    # Torque graph (bottom half)
    t_range = max(float(np.abs(ft_arr[:, 3:]).max()), 0.5) * 1.2

    def val_to_y_t(v):
        return int(force_h + (1 - (v + t_range) / (2 * t_range)) * (torque_h - 2))

    y0_t = val_to_y_t(0)
    cv2.line(canvas, (margin_l, y0_t), (width - margin_r, y0_t), (60, 60, 60), 1)

    torque_colors = [(255, 255, 0), (255, 0, 255), (0, 255, 255)]
    for ch, col in enumerate(torque_colors):
        pts = _polyline(ft_arr[:, ch + 3], val_to_y_t)
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts], False, col, 1, cv2.LINE_AA)

    cv2.putText(canvas, f"+/-{t_range:.1f}Nm", (2, force_h + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
    for i, (lbl, col) in enumerate(zip(['Tx', 'Ty', 'Tz'], torque_colors)):
        cv2.putText(canvas, lbl, (width - 70 + i * 22, force_h + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1)

    # Overlay model input FT windows as shaded regions
    # Each entry: (ts_array shape (N,), ft_array shape (N,6))
    # Current = green, previous = pink
    if model_ft_history and len(model_ft_history) >= 1:
        entries_to_draw = []
        # Previous (second-to-last) in pink
        if len(model_ft_history) >= 2:
            entries_to_draw.append((model_ft_history[-2], (180, 100, 200), "prev"))
        # Current (last) in green
        entries_to_draw.append((model_ft_history[-1], (0, 200, 100), "curr"))

        for (ts_win, ft_win), color, label in entries_to_draw:
            if ft_win.ndim == 1:
                continue
            x0 = max(margin_l, time_to_x(float(ts_win[0])))
            x1 = min(width - margin_r, time_to_x(float(ts_win[-1])))
            if x1 <= x0:
                continue
            overlay = canvas.copy()
            cv2.rectangle(overlay, (x0, 0), (x1, graph_h), color, -1)
            cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)
            cv2.rectangle(canvas, (x0, 0), (x1, graph_h), color, 1)
            lbl_text = "model in" if label == "curr" else "prev in"
            lbl_y = 12 if label == "curr" else 24
            cv2.putText(canvas, lbl_text, (max(x0 + 2, margin_l + 2), lbl_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    cv2.rectangle(canvas, (margin_l, 0), (width - margin_r, graph_h - 1), (80, 80, 80), 1)
    return canvas


def draw_action_panel(state, config, width, panel_h=70):
    """Draw model output action in multiple coordinate representations."""
    panel = np.zeros((panel_h, width, 3), dtype=np.uint8)

    if state.last_raw_action is None or state.last_current_ee is None:
        cv2.putText(panel, "Action: waiting for inference...", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        return panel

    raw = state.last_raw_action
    tf = state.last_transformed_action
    tgt = state.last_target_pose
    ee = state.last_current_ee
    y = 14

    pos_str = f"dx={raw[0]:+.4f} dy={raw[1]:+.4f} dz={raw[2]:+.4f}"
    cv2.putText(panel, f"Model: {pos_str}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 100), 1)
    y += 16

    if tf is not None:
        tf_pos = f"dx={tf[0]:+.4f} dy={tf[1]:+.4f} dz={tf[2]:+.4f}"
        tf_mm = f"({np.linalg.norm(tf[:3])*1000:.1f}mm)"
        cv2.putText(panel, f"Robot delta: {tf_pos} {tf_mm}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 200, 180), 1)
    y += 16

    if tgt is not None:
        tgt_str = f"x={tgt[0]:.4f} y={tgt[1]:.4f} z={tgt[2]:.4f}"
        cv2.putText(panel, f"Target: {tgt_str}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 180, 255), 1)
        ee_str = f"x={ee[0]:.4f} y={ee[1]:.4f} z={ee[2]:.4f}"
        cv2.putText(panel, f"EE: {ee_str}", (width // 2 + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    y += 16

    if tgt is not None and len(tgt) >= 7:
        try:
            action_mode = config['action'].get('action_mode', 'delta')
            if action_mode == 'rel_to_start':
                tgt_quat_xyzw = np.array([tgt[4], tgt[5], tgt[6], tgt[3]])
            else:
                tgt_quat_xyzw = tgt[3:7]
            tgt_euler = Rotation.from_quat(tgt_quat_xyzw).as_euler('xyz', degrees=True)
            cv2.putText(panel, f"Euler: r={tgt_euler[0]:+.1f} p={tgt_euler[1]:+.1f} y={tgt_euler[2]:+.1f} deg",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 150, 255), 1)
        except Exception:
            pass

    return panel


def make_model_input_panel(state, config, panel_h):
    """Create a panel showing all model input images (all history frames) at full panel height.
    Returns a BGR image of shape (panel_h, panel_w, 3).
    """
    target_size = config['image']['target_resolution'][0]
    hist_len = config['image']['history_length']
    thumb_size = panel_h // hist_len  # stack vertically per camera

    thumbs = []
    for cam_idx, (cam_label, frames) in enumerate([("C1", state.prev_frames_cam1),
                                                     ("C2", state.prev_frames_cam2)]):
        if not frames or len(frames) == 0:
            continue
        col_thumbs = []
        for t_idx in range(len(frames)):
            img_rgb = frames[t_idx]
            processed = resize_with_padding(img_rgb, target_size=target_size)
            thumb = cv2.resize(processed, (thumb_size, thumb_size))
            thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)
            is_latest = (t_idx == len(frames) - 1)
            border_color = (0, 255, 0) if is_latest else (0, 120, 0)
            cv2.rectangle(thumb_bgr, (0, 0), (thumb_size - 1, thumb_size - 1), border_color, 1)
            label = f"{cam_label} t-{len(frames)-1-t_idx}"
            cv2.putText(thumb_bgr, label, (2, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, border_color, 1)
            col_thumbs.append(thumb_bgr)
        # Stack vertically: oldest on top, newest on bottom
        col_img = np.vstack(col_thumbs)
        # Pad to panel_h if needed
        if col_img.shape[0] < panel_h:
            col_img = cv2.copyMakeBorder(col_img, 0, panel_h - col_img.shape[0], 0, 0,
                                          cv2.BORDER_CONSTANT, value=[0, 0, 0])
        thumbs.append(col_img)

    if not thumbs:
        panel = np.zeros((panel_h, panel_h * 2 + 5, 3), dtype=np.uint8)
        cv2.putText(panel, "No model input", (10, panel_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        return panel

    parts = thumbs
    return np.hstack(parts)


def draw_prediction_overlay(vis_img, state, config):
    """카메라 뷰 위에 모델 예측 방향을 미니 위젯으로 시각화."""
    if state.last_raw_action is None or state.last_current_ee is None:
        return

    raw = state.last_raw_action
    ee = state.last_current_ee

    R_mat_pos = np.array(config['coordinate_transform']['R_mat_pos'])
    R_mat_rot = np.array(config['coordinate_transform']['R_mat_rot'])
    current_quat = np.roll(ee[3:], -1)
    current_rot = Rotation.from_quat(current_quat)

    world_pos = current_rot.inv().apply(R_mat_pos @ raw[:3])
    body_rotvec = Rotation.from_quat(raw[3:7]).as_rotvec()
    R_new = R_mat_rot @ Rotation.from_rotvec(body_rotvec).as_matrix() @ R_mat_rot.T
    world_rotvec = Rotation.from_matrix(R_new).as_rotvec()
    world_rv_deg = np.degrees(world_rotvec)

    h, w = vis_img.shape[:2]
    box_size = 120
    margin = 10
    cx, cy = margin + box_size // 2, margin + box_size // 2
    arrow_scale = 30000

    # Position widget
    overlay = vis_img.copy()
    cv2.rectangle(overlay, (margin, margin), (margin + box_size, margin + box_size), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, vis_img, 0.4, 0, vis_img)

    cv2.putText(vis_img, "Pos(world)", (margin + 2, margin + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    # 축 가이드: +X → 위, +Y → 왼쪽 (로봇 정면 기준)
    cv2.arrowedLine(vis_img, (cx, cy), (cx, cy - 30), (80, 80, 80), 1, tipLength=0.3)
    cv2.putText(vis_img, "+X", (cx + 3, cy - 33), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (80, 80, 80), 1)
    cv2.arrowedLine(vis_img, (cx, cy), (cx - 30, cy), (80, 80, 80), 1, tipLength=0.3)
    cv2.putText(vis_img, "+Y", (cx - 48, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (80, 80, 80), 1)

    # screen: sx = -world_y (왼쪽이 +Y), sy = -world_x (위쪽이 +X)
    ax = int(-world_pos[1] * arrow_scale)
    ay = int(-world_pos[0] * arrow_scale)
    arrow_len = np.sqrt(ax**2 + ay**2)
    if arrow_len > 2:
        max_len = box_size // 2 - 15
        if arrow_len > max_len:
            ax = int(ax * max_len / arrow_len)
            ay = int(ay * max_len / arrow_len)
        cv2.arrowedLine(vis_img, (cx, cy), (cx + ax, cy + ay), (0, 255, 0), 2, tipLength=0.25)

    z_mm = world_pos[2] * 1000
    z_color = (0, 200, 255) if z_mm > 0.1 else (255, 100, 0) if z_mm < -0.1 else (180, 180, 180)
    cv2.putText(vis_img, f"Z:{z_mm:+.1f}mm", (margin + 2, margin + box_size - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, z_color, 1)
    pos_mm = np.linalg.norm(world_pos) * 1000
    cv2.putText(vis_img, f"|d|={pos_mm:.1f}mm", (margin + box_size - 60, margin + box_size - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 200, 200), 1)

    # Delta rotation widget: 모델 예측 회전을 축별 막대로 표시
    rx = margin + box_size + 10

    overlay2 = vis_img.copy()
    cv2.rectangle(overlay2, (rx, margin), (rx + box_size, margin + box_size), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.6, vis_img, 0.4, 0, vis_img)

    cv2.putText(vis_img, "dRot(world)", (rx + 2, margin + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # 축별 막대: Rx(roll), Ry(pitch), Rz(yaw)
    bar_w = 20
    bar_max_h = (box_size - 40) // 2  # 중심 기준 위아래 최대 길이
    deg_max = 5.0  # 5도에서 최대 막대 길이
    bar_y_center = margin + 20 + bar_max_h  # 막대 중심선
    labels = ['Rx', 'Ry', 'Rz']
    colors = [(0, 150, 255), (255, 150, 0), (0, 255, 255)]

    # 중심선
    cv2.line(vis_img, (rx + 5, bar_y_center), (rx + box_size - 5, bar_y_center), (60, 60, 60), 1)

    for i, (lbl, col) in enumerate(zip(labels, colors)):
        bx = rx + 10 + i * (bar_w + 15)
        deg = world_rv_deg[i]
        bar_h = int(np.clip(deg / deg_max, -1.0, 1.0) * bar_max_h)

        if abs(bar_h) > 1:
            if bar_h > 0:
                cv2.rectangle(vis_img, (bx, bar_y_center - bar_h), (bx + bar_w, bar_y_center), col, -1)
            else:
                cv2.rectangle(vis_img, (bx, bar_y_center), (bx + bar_w, bar_y_center - bar_h), col, -1)

        cv2.putText(vis_img, lbl, (bx, margin + box_size - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, col, 1)
        cv2.putText(vis_img, f"{deg:+.1f}", (bx - 2, margin + box_size - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (180, 180, 180), 1)


def make_display_image(left_bgr, right_bgr, mode, state, config,
                       ft_data=None, ft_collector=None):
    """Compose full display image: cameras + model input + FT graph + action + info."""
    h1, w1 = left_bgr.shape[:2]
    h2, w2 = right_bgr.shape[:2]
    target_height = max(h1, h2)

    def _vpad(img, h, th):
        if h < th:
            pt = (th - h) // 2
            return cv2.copyMakeBorder(img, pt, th - h - pt, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        return img

    vis_img = np.hstack([_vpad(left_bgr, h1, target_height), _vpad(right_bgr, h2, target_height)])
    scale = 0.6
    h, w = vis_img.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    vis_img = cv2.resize(vis_img, (new_w, new_h))

    # Model input panel (same height as camera row, placed to the right)
    model_panel = make_model_input_panel(state, config, panel_h=new_h)
    mp_h, mp_w = model_panel.shape[:2]
    # Pad model panel height to match camera row if needed
    if mp_h < new_h:
        model_panel = cv2.copyMakeBorder(model_panel, 0, new_h - mp_h, 0, 0,
                                         cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif mp_h > new_h:
        model_panel = model_panel[:new_h]
    vis_img = np.hstack([vis_img, model_panel])
    total_w = vis_img.shape[1]

    if ft_collector is not None:
        ft_hist, ft_ts = ft_collector.graph_data()
        ca_thr = config.get('contact_assist', {}).get('force_threshold', 3.0)
        if isinstance(ca_thr, (list, tuple)):
            ca_thr = float(max(ca_thr))
        ft_graph = draw_ft_graph(ft_hist, ft_ts,
                                 width=total_w, graph_h=180, contact_threshold=ca_thr,
                                 model_ft_history=state.ft_model_input_history)
    else:
        ft_graph = np.zeros((180, total_w, 3), dtype=np.uint8)
        cv2.putText(ft_graph, "FT graph: no collector", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

    # Info panel
    state_colors = {'IDLE': (128, 128, 128), 'REACHING': (0, 255, 255), 'CONTACT': (0, 140, 255),
                    'INSERTED': (0, 255, 255), 'SUCCESS': (0, 255, 0), 'FAIL': (0, 0, 255)}
    info_h = 80
    info_panel = np.zeros((info_h, total_w, 3), dtype=np.uint8)

    mode_text = f"{mode.upper()} | {state.gui_mode_label}"
    if state.gui_inference_hz > 0:
        mode_text += f"  model:{state.gui_inference_hz:.1f}Hz"
    if state.gui_cycle_ms > 0:
        mode_text += f"  cycle:{state.gui_cycle_ms:.0f}ms"
    hz_color = (0, 255, 0) if state.gui_inference_hz >= 20 else (0, 165, 255) if state.gui_inference_hz > 0 else (255, 255, 255)
    cv2.putText(info_panel, mode_text, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, hz_color, 1)
    cv2.putText(info_panel, state.task_state, (total_w - 130, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_colors.get(state.task_state, (128, 128, 128)), 2)

    cv2.putText(info_panel, f"{state.gui_timesteps_str}", (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1)
    if state.policy_start_time is not None:
        r_time = state.contact_time if state.contact_time is not None else (time.time() - state.policy_start_time)
        r_steps = state.contact_step if state.contact_step is not None else state.total_inference_steps
        cv2.putText(info_panel, f"Reach: {r_time:.1f}s/{r_steps}st", (280, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)
    if state.ft_time_start is not None:
        c_time = state.insert_time if state.insert_time is not None else (time.time() - state.ft_time_start)
        c_steps = state.insert_step if state.insert_step is not None else (state.total_inference_steps - (state.contact_step or 0))
        cv2.putText(info_panel, f"Contact: {c_time:.1f}s/{c_steps}st", (460, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 80, 255), 1)

    if ft_data is not None:
        ft = np.asarray(ft_data, dtype=np.float32)
        if ft.ndim == 2:
            ft = ft[-1]
        ft_d = ft - state.gripping_ft_bias
        fn = float(np.linalg.norm(ft_d[:3]))
        cv2.putText(info_panel, f"dF:[{ft_d[0]:+.1f},{ft_d[1]:+.1f},{ft_d[2]:+.1f}] |dF|={fn:.1f}N  "
                    f"dT:[{ft_d[3]:+.2f},{ft_d[4]:+.2f},{ft_d[5]:+.2f}]",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 220, 255), 1)

    cv2.putText(info_panel, "t(stop) p(start) s(ok) f(fail) e(exit) r(reset) q(quit)",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

    draw_prediction_overlay(vis_img, state, config)

    return np.vstack([vis_img, ft_graph, info_panel])
