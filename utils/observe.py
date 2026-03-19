"""Observe mode: table display, CSV export, 3D visualization."""
import os
import numpy as np
from scipy.spatial.transform import Rotation


def observe_body_to_world(chunk_body, current_ee_pose, config):
    """Model body output → world delta (same transform as action_builder)."""
    R_mat_pos = np.array(config['coordinate_transform']['R_mat_pos'])
    current_quat = np.roll(current_ee_pose[3:], -1)  # wxyz→xyzw
    current_rot = Rotation.from_quat(current_quat)
    world = np.array([current_rot.inv().apply(R_mat_pos @ b) for b in chunk_body])
    return world


def print_observe_table(observe_actions, show_last=15):
    """Print recent N inferences as a table in observe mode."""
    n = len(observe_actions)
    print(f"\033[2J\033[H", end='')

    print(f"{'='*90}")
    print(f"  [OBSERVE MODE] 로봇 고정 | 인퍼런스 {n}회 | 'o' 키로 해제")
    print(f"{'='*90}")

    body_sums = [bp.sum(axis=0) for bp, wp, br in observe_actions]
    world_sums = [wp.sum(axis=0) for bp, wp, br in observe_actions]

    start = max(0, n - show_last)
    disp_b = body_sums[start:]
    disp_w = world_sums[start:]

    world_dir_names = ['+X(우)', '-X(좌)', '+Y(앞)', '-Y(뒤)', '+Z(위)', '-Z(아래)']

    print(f"  {'#':>4}  ┃{'BODY (그리퍼 기준)':^30}┃{'WORLD (로봇 베이스 기준)':^34}┃")
    print(f"  {'':>4}  ┃{'X':>8} {'Y':>8} {'Z':>8} {'|d|':>7}┃{'X':>8} {'Y':>8} {'Z':>8} {'|d|':>7}  방향    ┃")
    print(f"  {'─'*4}──╋{'─'*31}╋{'─'*34}╋")
    for i, (bs, ws) in enumerate(zip(disp_b, disp_w)):
        idx = start + i + 1
        bmag = np.linalg.norm(bs) * 1000
        wmag = np.linalg.norm(ws) * 1000
        abs_w = np.abs(ws)
        dom = np.argmax(abs_w)
        wdir = world_dir_names[dom * 2 + (0 if ws[dom] > 0 else 1)]
        print(f"  {idx:4d}  ┃{bs[0]*1000:+8.2f} {bs[1]*1000:+8.2f} {bs[2]*1000:+8.2f} {bmag:7.2f}┃"
              f"{ws[0]*1000:+8.2f} {ws[1]*1000:+8.2f} {ws[2]*1000:+8.2f} {wmag:7.2f}  {wdir:<8s}┃")

    if start > 0:
        print(f"  ... ({start}개 생략)")

    if n >= 2:
        all_b = np.array(body_sums)
        all_w = np.array(world_sums)
        mb = all_b.mean(axis=0); sb = all_b.std(axis=0)
        mw = all_w.mean(axis=0); sw = all_w.std(axis=0)
        print(f"{'─'*90}")
        print(f"  평균 ┃{mb[0]*1000:+8.2f} {mb[1]*1000:+8.2f} {mb[2]*1000:+8.2f} {np.linalg.norm(mb)*1000:7.2f}┃"
              f"{mw[0]*1000:+8.2f} {mw[1]*1000:+8.2f} {mw[2]*1000:+8.2f} {np.linalg.norm(mw)*1000:7.2f}          ┃")
        print(f"  std  ┃{sb[0]*1000:8.2f} {sb[1]*1000:8.2f} {sb[2]*1000:8.2f}        ┃"
              f"{sw[0]*1000:8.2f} {sw[1]*1000:8.2f} {sw[2]*1000:8.2f}                  ┃")

        signs = np.sign(all_w)
        cons = []
        for ax in range(3):
            nz = signs[:, ax] != 0
            if nz.sum() > 0:
                dom_sign = np.sign(mw[ax])
                pct = (signs[nz, ax] == dom_sign).sum() / nz.sum() * 100
                cons.append(f"{'XYZ'[ax]}:{pct:.0f}%")
            else:
                cons.append(f"{'XYZ'[ax]}:--")
        dom_ax = np.argmax(np.abs(mw))
        dom_name = world_dir_names[dom_ax * 2 + (0 if mw[dom_ax] > 0 else 1)]
        print(f"  일관성(world): {' | '.join(cons)}")
        print(f"  → 로봇이 가려는 방향: {dom_name}")

    print(f"{'='*90}", flush=True)


def save_observe_csv(observe_actions, freeze_pose, config):
    """Save observe mode results to CSV."""
    import datetime
    import pandas as pd

    R_mat_rot = np.array(config['coordinate_transform']['R_mat_rot'])

    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('eval_results', 'observe')
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f'observe_{ts}.csv')

    rows = []
    for inf_idx, (body_pos, world_pos, body_rot) in enumerate(observe_actions):
        for step_idx in range(len(body_pos)):
            body_q = body_rot[step_idx]
            body_rotvec = Rotation.from_quat(body_q).as_rotvec()

            R_scaled = Rotation.from_quat(body_q).as_matrix()
            R_new = R_mat_rot @ R_scaled @ R_mat_rot.T
            world_rotvec = Rotation.from_matrix(R_new).as_rotvec()

            rows.append({
                'inference': inf_idx + 1,
                'chunk_step': step_idx + 1,
                'chunk_size': len(body_pos),
                'body_dx': body_pos[step_idx, 0],
                'body_dy': body_pos[step_idx, 1],
                'body_dz': body_pos[step_idx, 2],
                'body_rx': body_rotvec[0],
                'body_ry': body_rotvec[1],
                'body_rz': body_rotvec[2],
                'world_dx': world_pos[step_idx, 0],
                'world_dy': world_pos[step_idx, 1],
                'world_dz': world_pos[step_idx, 2],
                'world_rx': world_rotvec[0],
                'world_ry': world_rotvec[1],
                'world_rz': world_rotvec[2],
                'ee_x': freeze_pose[0],
                'ee_y': freeze_pose[1],
                'ee_z': freeze_pose[2],
                'ee_qw': freeze_pose[3],
                'ee_qx': freeze_pose[4],
                'ee_qy': freeze_pose[5],
                'ee_qz': freeze_pose[6],
            })

    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  [SAVED] {csv_path} ({len(rows)} rows, {len(observe_actions)} inferences)")


def update_observe_3d(observe_actions, state):
    """Visualize action chunks in 3D matplotlib (body + world frames)."""
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    if state.observe_fig is None or not plt.fignum_exists(state.observe_fig.number):
        plt.ion()
        state.observe_fig = plt.figure('Observe 3D', figsize=(14, 7))
        state._observe_view = {}

    fig = state.observe_fig
    for i, ax_old in enumerate(fig.axes):
        if hasattr(ax_old, 'elev'):
            state._observe_view[i] = (ax_old.elev, ax_old.azim)
    fig.clf()
    ax_body = fig.add_subplot(121, projection='3d')
    ax_world = fig.add_subplot(122, projection='3d')

    n = len(observe_actions)
    cmap = plt.cm.viridis
    show_last = min(n, 20)

    for ax, frame_idx, title in [(ax_body, 0, 'Body Frame (gripper)'),
                                  (ax_world, 1, 'World Frame (robot base)')]:
        for i in range(max(0, n - show_last), n):
            chunk = observe_actions[i][frame_idx]
            waypoints = np.zeros((len(chunk) + 1, 3))
            waypoints[1:] = np.cumsum(chunk, axis=0)
            waypoints *= 1000

            age = (i - max(0, n - show_last)) / max(show_last - 1, 1)
            color = cmap(age)
            alpha = 0.3 + 0.7 * age

            ax.plot(waypoints[:, 0], waypoints[:, 1], waypoints[:, 2],
                    '-o', color=color, alpha=alpha, markersize=3, linewidth=1.5)
            ax.text(waypoints[-1, 0], waypoints[-1, 1], waypoints[-1, 2],
                    f'{i+1}', fontsize=7, color=color, alpha=alpha)

        ax.scatter([0], [0], [0], c='red', s=80, marker='*', zorder=10)

        if n >= 2:
            chunks = [observe_actions[i][frame_idx] for i in range(n)]
            max_steps = max(len(c) for c in chunks)
            padded = []
            for c in chunks:
                if len(c) < max_steps:
                    padded.append(np.vstack([c, np.zeros((max_steps - len(c), 3))]))
                else:
                    padded.append(c)
            mean_c = np.mean(padded, axis=0)
            mean_wp = np.zeros((max_steps + 1, 3))
            mean_wp[1:] = np.cumsum(mean_c, axis=0) * 1000
            ax.plot(mean_wp[:, 0], mean_wp[:, 1], mean_wp[:, 2],
                    '-s', color='red', linewidth=3, markersize=5, label='mean', zorder=9)

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f'{title} — {n} inferences')
        ax.legend(fontsize=8)

        all_wp = []
        for i in range(n):
            c = observe_actions[i][frame_idx]
            all_wp.append(np.cumsum(c, axis=0) * 1000)
        all_wp = np.vstack(all_wp)
        if len(all_wp) > 0:
            max_range = max(np.abs(all_wp).max(), 0.5) * 1.3
            ax.set_xlim(-max_range, max_range)
            ax.set_ylim(-max_range, max_range)
            ax.set_zlim(-max_range, max_range)

    for i, ax in enumerate(fig.axes):
        if i in state._observe_view:
            ax.view_init(elev=state._observe_view[i][0], azim=state._observe_view[i][1])

    fig.tight_layout()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
