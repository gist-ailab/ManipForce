#!/usr/bin/env python3
"""
학습 데이터 에피소드 시각화
==========================
zarr 데이터셋에서 에피소드 하나를 선택하여 시각화.
- 카메라 이미지 (cam1, cam2)
- action: model coord (dx,dy,dz) + quaternion
- state: pose_wrt_start (시작 대비 상대 위치)
- FT 그래프 (force/torque 6축 시계열)
- 누적 trajectory (xy, xz 평면)

사용법:
  python scripts/visualize/visualize_episode.py --zarr /path/to/data.zarr
  python scripts/visualize/visualize_episode.py --zarr /path/to/data.zarr --episode 5
  python scripts/visualize/visualize_episode.py --zarr /path/to/data.zarr --episode 5 --save video.mp4

조작:
  ←/→ or a/d : 프레임 이동
  Space      : 재생/일시정지
  +/-        : 재생 속도 조절
  q/ESC      : 종료
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import argparse
import numpy as np
import zarr
import cv2
from scipy.spatial.transform import Rotation


def load_episode(zarr_path, ep_idx):
    """Load one episode from zarr dataset."""
    z = zarr.open(zarr_path, 'r')

    img_ends = z['meta/episode_ends'][:]
    img_starts = np.concatenate([[0], img_ends[:-1]])
    ft_ends = z['meta/episode_ft_ends'][:]
    ft_starts = np.concatenate([[0], ft_ends[:-1]])

    n_episodes = len(img_ends)
    if ep_idx < 0 or ep_idx >= n_episodes:
        print(f"Episode {ep_idx} out of range [0, {n_episodes - 1}]")
        sys.exit(1)

    si, ei = int(img_starts[ep_idx]), int(img_ends[ep_idx])
    fsi, fei = int(ft_starts[ep_idx]), int(ft_ends[ep_idx])

    ep = {
        'cam1': z['data/handeye_cam_1'][si:ei],
        'action': z['data/action'][si:ei],
        'state': z['data/state'][si:ei],
        'img_ts': z['data/img_timestamps'][si:ei],
        'ft_data': z['data/ft_data'][fsi:fei],
        'ft_ts': z['data/ft_timestamps'][fsi:fei],
        'n_episodes': n_episodes,
    }

    if 'handeye_cam_2' in z['data']:
        ep['cam2'] = z['data/handeye_cam_2'][si:ei]

    if 'pose_wrt_start' in z['data']:
        ep['pose_wrt_start'] = z['data/pose_wrt_start'][si:ei]

    return ep


def draw_ft_graph(ft_data, ft_ts, img_ts, frame_idx, width=640, height=200):
    """Draw FT time-series graph with current frame marker."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    if len(ft_data) == 0:
        return canvas

    labels = ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']
    colors = [
        (80, 80, 255),    # fx - red
        (80, 255, 80),    # fy - green
        (255, 180, 80),   # fz - blue
        (180, 80, 255),   # tx - purple
        (80, 255, 255),   # ty - yellow
        (255, 255, 80),   # tz - cyan
    ]

    margin_l, margin_r, margin_t, margin_b = 50, 10, 15, 25
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b

    # Force and torque separate ranges
    f_max = max(np.abs(ft_data[:, :3]).max(), 1.0)
    t_max = max(np.abs(ft_data[:, 3:]).max(), 0.1)

    # Draw zero line
    y_zero_f = margin_t + plot_h // 4
    y_zero_t = margin_t + 3 * plot_h // 4
    cv2.line(canvas, (margin_l, y_zero_f), (width - margin_r, y_zero_f), (40, 40, 40), 1)
    cv2.line(canvas, (margin_l, y_zero_t), (width - margin_r, y_zero_t), (40, 40, 40), 1)

    # Separator
    y_sep = margin_t + plot_h // 2
    cv2.line(canvas, (margin_l, y_sep), (width - margin_r, y_sep), (60, 60, 60), 1)

    # Y-axis labels
    cv2.putText(canvas, f"+{f_max:.1f}N", (2, margin_t + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (120, 120, 120), 1)
    cv2.putText(canvas, f"-{f_max:.1f}N", (2, y_sep - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (120, 120, 120), 1)
    cv2.putText(canvas, f"+{t_max:.2f}", (2, y_sep + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (120, 120, 120), 1)
    cv2.putText(canvas, f"-{t_max:.2f}", (2, height - margin_b - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (120, 120, 120), 1)

    n_ft = len(ft_data)
    half_h = plot_h // 2

    # Draw force channels (top half)
    for ch in range(3):
        pts = []
        for i in range(n_ft):
            x = margin_l + int(i / max(n_ft - 1, 1) * plot_w)
            val = ft_data[i, ch]
            y = y_zero_f - int(val / f_max * (half_h // 2 - 5))
            y = np.clip(y, margin_t, y_sep - 1)
            pts.append((x, y))
        if len(pts) > 1:
            cv2.polylines(canvas, [np.array(pts)], False, colors[ch], 1, cv2.LINE_AA)

    # Draw torque channels (bottom half)
    for ch in range(3):
        pts = []
        for i in range(n_ft):
            x = margin_l + int(i / max(n_ft - 1, 1) * plot_w)
            val = ft_data[i, ch + 3]
            y = y_zero_t - int(val / t_max * (half_h // 2 - 5))
            y = np.clip(y, y_sep + 1, height - margin_b)
            pts.append((x, y))
        if len(pts) > 1:
            cv2.polylines(canvas, [np.array(pts)], False, colors[ch + 3], 1, cv2.LINE_AA)

    # Current frame marker (vertical line)
    if frame_idx < len(img_ts):
        cur_t = img_ts[frame_idx]
        ft_t0, ft_t1 = ft_ts[0], ft_ts[-1]
        if ft_t1 > ft_t0:
            ratio = (cur_t - ft_t0) / (ft_t1 - ft_t0)
            x_marker = margin_l + int(np.clip(ratio, 0, 1) * plot_w)
            cv2.line(canvas, (x_marker, margin_t), (x_marker, height - margin_b),
                     (0, 255, 255), 1)

    # Legend
    for i, (label, color) in enumerate(zip(labels, colors)):
        lx = margin_l + i * 55
        cv2.putText(canvas, label, (lx, height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    return canvas


def draw_trajectory(states, frame_idx, width=200, height=200):
    """Draw XY and XZ trajectory plots."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    if states is None or len(states) == 0:
        return canvas

    pos = states[:, :3]  # (N, 3)
    half_w = width // 2
    margin = 15

    for plot_idx, (ax1, ax2, title) in enumerate([(0, 1, 'XY'), (0, 2, 'XZ')]):
        ox = plot_idx * half_w
        plot_area = half_w - 2 * margin

        d1, d2 = pos[:, ax1], pos[:, ax2]
        all_vals = np.concatenate([d1, d2])
        vrange = max(np.abs(all_vals).max(), 0.001) * 1.2

        def to_px(v1, v2):
            px = ox + margin + int((v1 / vrange + 1) / 2 * plot_area)
            py = margin + int((-v2 / vrange + 1) / 2 * (height - 2 * margin))
            return (np.clip(px, ox, ox + half_w - 1), np.clip(py, 0, height - 1))

        # Cross at origin
        cx, cy = to_px(0, 0)
        cv2.line(canvas, (cx - 4, cy), (cx + 4, cy), (50, 50, 50), 1)
        cv2.line(canvas, (cx, cy - 4), (cx, cy + 4), (50, 50, 50), 1)

        # Trajectory line
        pts = [to_px(d1[i], d2[i]) for i in range(len(d1))]
        if len(pts) > 1:
            # Past trajectory (dim)
            if frame_idx > 0:
                cv2.polylines(canvas, [np.array(pts[:frame_idx + 1])],
                              False, (60, 120, 60), 1, cv2.LINE_AA)
            # Future trajectory (dimmer)
            if frame_idx < len(pts) - 1:
                cv2.polylines(canvas, [np.array(pts[frame_idx:])],
                              False, (40, 40, 60), 1, cv2.LINE_AA)

        # Current position
        if frame_idx < len(pts):
            cv2.circle(canvas, pts[frame_idx], 4, (0, 255, 0), -1)

        # Start marker
        cv2.circle(canvas, pts[0], 3, (255, 100, 100), -1)

        # Title
        cv2.putText(canvas, title, (ox + 5, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
        cv2.putText(canvas, f"{vrange*1000:.0f}mm", (ox + half_w - 45, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (100, 100, 100), 1)

    return canvas


def draw_action_panel(action, state, width=640, panel_h=55):
    """Draw action info similar to eval GUI."""
    panel = np.zeros((panel_h, width, 3), dtype=np.uint8)

    if action is None:
        return panel

    y = 14

    # Row 1: position delta
    pos_str = f"dx={action[0]:+.5f}  dy={action[1]:+.5f}  dz={action[2]:+.5f}  |d|={np.linalg.norm(action[:3])*1000:.2f}mm"
    cv2.putText(panel, f"Action: {pos_str}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 100), 1)
    y += 16

    # Row 2: quaternion as euler
    if len(action) >= 7:
        try:
            q = action[3:7]
            euler = Rotation.from_quat(q).as_euler('xyz', degrees=True)
            cv2.putText(panel, f"Rot: r={euler[0]:+.2f}  p={euler[1]:+.2f}  y={euler[2]:+.2f} deg  "
                        f"q=[{q[0]:+.4f},{q[1]:+.4f},{q[2]:+.4f},{q[3]:+.4f}]",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 200, 180), 1)
        except Exception:
            pass
    y += 16

    # Row 3: gripper + state position
    grip_str = f"grip={action[7]:.2f}" if len(action) > 7 else ""
    state_str = ""
    if state is not None:
        state_str = f"  |  State: x={state[0]:+.4f} y={state[1]:+.4f} z={state[2]:+.4f}"
    cv2.putText(panel, f"{grip_str}{state_str}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 150, 255), 1)

    return panel


def draw_info_bar(ep_idx, n_episodes, frame_idx, n_frames, playing, speed, width=640, height=25):
    """Draw bottom info bar."""
    bar = np.zeros((height, width, 3), dtype=np.uint8)
    status = "PLAY" if playing else "PAUSE"
    text = (f"Episode {ep_idx}/{n_episodes - 1}  |  Frame {frame_idx}/{n_frames - 1}  |  "
            f"{status} x{speed:.1f}  |  [a/d] frame  [space] play  [+/-] speed  [q] quit")
    cv2.putText(bar, text, (10, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)
    return bar


def main():
    parser = argparse.ArgumentParser(description="학습 데이터 에피소드 시각화")
    parser.add_argument('--zarr', required=True, help="zarr 데이터셋 경로")
    parser.add_argument('--episode', '-e', type=int, default=0, help="에피소드 인덱스")
    parser.add_argument('--save', default=None, help="비디오 저장 경로 (e.g. output.mp4)")
    parser.add_argument('--cam-scale', type=float, default=0.4, help="카메라 이미지 스케일")
    args = parser.parse_args()

    print(f"Loading episode {args.episode} from {args.zarr} ...")
    ep = load_episode(args.zarr, args.episode)
    n_frames = len(ep['cam1'])
    print(f"  Frames: {n_frames}, FT samples: {len(ep['ft_data'])}")

    frame_idx = 0
    playing = False
    speed = 1.0
    video_writer = None

    cv2.namedWindow("Episode Viewer", cv2.WINDOW_NORMAL)

    while True:
        # --- Camera images ---
        cam1 = ep['cam1'][frame_idx]  # RGB
        cam1_bgr = cv2.cvtColor(cam1, cv2.COLOR_RGB2BGR)
        h1, w1 = cam1_bgr.shape[:2]
        cam1_small = cv2.resize(cam1_bgr, (int(w1 * args.cam_scale), int(h1 * args.cam_scale)))

        if 'cam2' in ep:
            cam2 = ep['cam2'][frame_idx]
            cam2_bgr = cv2.cvtColor(cam2, cv2.COLOR_RGB2BGR)
            h2, w2 = cam2_bgr.shape[:2]
            scale2 = cam1_small.shape[0] / h2
            cam2_small = cv2.resize(cam2_bgr, (int(w2 * scale2), cam1_small.shape[0]))
            cam_row = np.hstack([cam1_small, cam2_small])
        else:
            cam_row = cam1_small

        vis_width = cam_row.shape[1]

        # --- Action panel ---
        action = ep['action'][frame_idx]
        state = ep['pose_wrt_start'][frame_idx] if 'pose_wrt_start' in ep else ep['state'][frame_idx]
        action_panel = draw_action_panel(action, state, width=vis_width)

        # --- FT graph ---
        ft_graph = draw_ft_graph(ep['ft_data'], ep['ft_ts'], ep['img_ts'],
                                 frame_idx, width=vis_width, height=180)

        # --- Trajectory ---
        traj_states = ep.get('pose_wrt_start', ep['state'])
        traj_panel = draw_trajectory(traj_states, frame_idx,
                                     width=vis_width, height=160)

        # --- Info bar ---
        info = draw_info_bar(args.episode, ep['n_episodes'], frame_idx, n_frames,
                             playing, speed, width=vis_width)

        # --- Compose ---
        vis = np.vstack([cam_row, action_panel, ft_graph, traj_panel, info])

        cv2.imshow("Episode Viewer", vis)

        # Video writer
        if args.save and video_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(args.save, fourcc, 30, (vis.shape[1], vis.shape[0]))
        if video_writer:
            video_writer.write(vis)

        # --- Input ---
        wait_ms = max(1, int(1000 / 30 / speed)) if playing else 0
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord('q') or key == 27:  # q or ESC
            break
        elif key == ord(' '):
            playing = not playing
        elif key == ord('d') or key == 83:  # d or →
            frame_idx = min(frame_idx + 1, n_frames - 1)
        elif key == ord('a') or key == 81:  # a or ←
            frame_idx = max(frame_idx - 1, 0)
        elif key == ord('+') or key == ord('='):
            speed = min(speed * 1.5, 10.0)
        elif key == ord('-') or key == ord('_'):
            speed = max(speed / 1.5, 0.1)
        elif playing:
            frame_idx += 1
            if frame_idx >= n_frames:
                frame_idx = 0
                playing = False

    if video_writer:
        video_writer.release()
        print(f"Video saved: {args.save}")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
