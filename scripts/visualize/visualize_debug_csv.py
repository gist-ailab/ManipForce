#!/usr/bin/env python3
"""
debug_tracking.csv 시각화
=========================
eval 실행 중 기록된 CSV를 분석하여:
- 모델 raw output (model coord) 방향
- 변환 후 robot coord delta 방향
- 실제 EE 궤적
- target 궤적
을 2D/3D 그래프로 시각화.

사용법:
  python scripts/visualize/visualize_debug_csv.py eval_results/gear_insertion_2/baseline_16_step_delta_8step_async/debug_tracking.csv
  python scripts/visualize/visualize_debug_csv.py eval_results/gear_insertion_2/*/debug_tracking.csv --compare
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial.transform import Rotation


def load_csv(path):
    df = pd.read_csv(path)
    print(f"Loaded {path}: {len(df)} rows, columns: {list(df.columns)}")
    return df


def has_raw_action(df):
    return 'raw_dx' in df.columns


def plot_single(df, title="", save_path=None):
    has_raw = has_raw_action(df)

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(title or "Debug Tracking Analysis", fontsize=14)
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    steps = df['step'].values
    ee_xyz = df[['ee_x', 'ee_y', 'ee_z']].values
    tgt_xyz = df[['tgt_x', 'tgt_y', 'tgt_z']].values
    tf_xyz = df[['tf_pos_x', 'tf_pos_y', 'tf_pos_z']].values

    # === 1. EE Trajectory (XY, XZ, YZ) ===
    planes = [('x', 'y', 0, 1), ('x', 'z', 0, 2), ('y', 'z', 1, 2)]
    for i, (ax1, ax2, i1, i2) in enumerate(planes):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(ee_xyz[:, i1], ee_xyz[:, i2], 'b-', alpha=0.5, linewidth=0.8, label='EE')
        ax.plot(tgt_xyz[:, i1], tgt_xyz[:, i2], 'r-', alpha=0.5, linewidth=0.8, label='Target')
        ax.plot(ee_xyz[0, i1], ee_xyz[0, i2], 'go', markersize=8, label='Start')
        ax.plot(ee_xyz[-1, i1], ee_xyz[-1, i2], 'rs', markersize=8, label='End')
        ax.set_xlabel(f'robot_{ax1} (m)')
        ax.set_ylabel(f'robot_{ax2} (m)')
        ax.set_title(f'Trajectory {ax1.upper()}{ax2.upper()}')
        ax.legend(fontsize=7)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    # === 2. Robot delta (tf_pos) over time ===
    ax_tf = fig.add_subplot(gs[1, 0])
    ax_tf.plot(steps, tf_xyz[:, 0], 'r-', alpha=0.7, label='tf_dx (robot)')
    ax_tf.plot(steps, tf_xyz[:, 1], 'g-', alpha=0.7, label='tf_dy (robot)')
    ax_tf.plot(steps, tf_xyz[:, 2], 'b-', alpha=0.7, label='tf_dz (robot)')
    ax_tf.set_xlabel('Step')
    ax_tf.set_ylabel('Delta (m)')
    ax_tf.set_title('Robot Coord Delta (per step)')
    ax_tf.legend(fontsize=7)
    ax_tf.grid(True, alpha=0.3)

    # === 3. Raw model output over time ===
    ax_raw = fig.add_subplot(gs[1, 1])
    if has_raw:
        raw_xyz = df[['raw_dx', 'raw_dy', 'raw_dz']].values
        ax_raw.plot(steps, raw_xyz[:, 0], 'r-', alpha=0.7, label='raw_dx (model)')
        ax_raw.plot(steps, raw_xyz[:, 1], 'g-', alpha=0.7, label='raw_dy (model)')
        ax_raw.plot(steps, raw_xyz[:, 2], 'b-', alpha=0.7, label='raw_dz (model)')
        ax_raw.set_title('Model Coord Output (per step)')
    else:
        ax_raw.text(0.5, 0.5, 'raw_action not recorded\n(re-run eval to get)',
                    ha='center', va='center', transform=ax_raw.transAxes, fontsize=10)
        ax_raw.set_title('Model Coord Output (N/A)')
    ax_raw.set_xlabel('Step')
    ax_raw.set_ylabel('Delta (m)')
    ax_raw.legend(fontsize=7)
    ax_raw.grid(True, alpha=0.3)

    # === 4. Cumulative displacement ===
    ax_cum = fig.add_subplot(gs[1, 2])
    ee_disp = ee_xyz - ee_xyz[0]
    ax_cum.plot(steps, ee_disp[:, 0] * 1000, 'r-', alpha=0.7, label='x')
    ax_cum.plot(steps, ee_disp[:, 1] * 1000, 'g-', alpha=0.7, label='y')
    ax_cum.plot(steps, ee_disp[:, 2] * 1000, 'b-', alpha=0.7, label='z')
    ax_cum.set_xlabel('Step')
    ax_cum.set_ylabel('Displacement (mm)')
    ax_cum.set_title('EE Cumulative Displacement (robot coord)')
    ax_cum.legend(fontsize=7)
    ax_cum.grid(True, alpha=0.3)

    # === 5. EE position over time ===
    ax_pos = fig.add_subplot(gs[2, 0])
    ax_pos.plot(steps, ee_xyz[:, 0], 'r-', alpha=0.7, label='ee_x')
    ax_pos.plot(steps, ee_xyz[:, 1], 'g-', alpha=0.7, label='ee_y')
    ax_pos.plot(steps, ee_xyz[:, 2], 'b-', alpha=0.7, label='ee_z')
    ax_pos.set_xlabel('Step')
    ax_pos.set_ylabel('Position (m)')
    ax_pos.set_title('EE Absolute Position')
    ax_pos.legend(fontsize=7)
    ax_pos.grid(True, alpha=0.3)

    # === 6. Delta magnitude ===
    ax_mag = fig.add_subplot(gs[2, 1])
    tf_mag = np.linalg.norm(tf_xyz, axis=1) * 1000
    ax_mag.plot(steps, tf_mag, 'k-', alpha=0.7, label='|delta| (robot)')
    if has_raw:
        raw_mag = np.linalg.norm(raw_xyz, axis=1) * 1000
        ax_mag.plot(steps, raw_mag, 'orange', alpha=0.7, label='|delta| (model)')
    ax_mag.set_xlabel('Step')
    ax_mag.set_ylabel('mm')
    ax_mag.set_title('Action Magnitude')
    ax_mag.legend(fontsize=7)
    ax_mag.grid(True, alpha=0.3)

    # === 7. Orientation (euler) ===
    ax_euler = fig.add_subplot(gs[2, 2])
    try:
        ee_quats = df[['ee_qx', 'ee_qy', 'ee_qz', 'ee_qw']].values  # xyzw for scipy
        eulers = np.array([Rotation.from_quat([q[0], q[1], q[2], q[3]]).as_euler('xyz', degrees=True)
                          for q in ee_quats])
        ax_euler.plot(steps, eulers[:, 0], 'r-', alpha=0.7, label='roll')
        ax_euler.plot(steps, eulers[:, 1], 'g-', alpha=0.7, label='pitch')
        ax_euler.plot(steps, eulers[:, 2], 'b-', alpha=0.7, label='yaw')
        ax_euler.set_ylabel('Degrees')
    except Exception:
        ax_euler.text(0.5, 0.5, 'Euler conversion failed', ha='center', va='center',
                     transform=ax_euler.transAxes)
    ax_euler.set_xlabel('Step')
    ax_euler.set_title('EE Orientation (Euler)')
    ax_euler.legend(fontsize=7)
    ax_euler.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def print_summary(df):
    ee_xyz = df[['ee_x', 'ee_y', 'ee_z']].values
    tgt_xyz = df[['tgt_x', 'tgt_y', 'tgt_z']].values
    tf_xyz = df[['tf_pos_x', 'tf_pos_y', 'tf_pos_z']].values

    total_disp = ee_xyz[-1] - ee_xyz[0]
    total_dist = np.linalg.norm(total_disp)

    print(f"\n{'='*60}")
    print(f"총 스텝: {len(df)}")
    print(f"EE 시작: [{ee_xyz[0,0]:.4f}, {ee_xyz[0,1]:.4f}, {ee_xyz[0,2]:.4f}]")
    print(f"EE 끝:   [{ee_xyz[-1,0]:.4f}, {ee_xyz[-1,1]:.4f}, {ee_xyz[-1,2]:.4f}]")
    print(f"총 이동 (robot): dx={total_disp[0]*1000:+.1f}mm  dy={total_disp[1]*1000:+.1f}mm  dz={total_disp[2]*1000:+.1f}mm  |d|={total_dist*1000:.1f}mm")

    print(f"\nRobot delta 평균: dx={tf_xyz[:,0].mean()*1000:+.3f}mm  dy={tf_xyz[:,1].mean()*1000:+.3f}mm  dz={tf_xyz[:,2].mean()*1000:+.3f}mm")

    if has_raw_action(df):
        raw_xyz = df[['raw_dx', 'raw_dy', 'raw_dz']].values
        print(f"Model raw 평균:   dx={raw_xyz[:,0].mean()*1000:+.3f}mm  dy={raw_xyz[:,1].mean()*1000:+.3f}mm  dz={raw_xyz[:,2].mean()*1000:+.3f}mm")

        # Mapping analysis
        print(f"\n--- Model→Robot 축 매핑 분석 ---")
        raw_sum = raw_xyz.sum(axis=0)
        tf_sum = tf_xyz.sum(axis=0)
        print(f"Model 누적: [{raw_sum[0]*1000:+.1f}, {raw_sum[1]*1000:+.1f}, {raw_sum[2]*1000:+.1f}] mm")
        print(f"Robot 누적: [{tf_sum[0]*1000:+.1f}, {tf_sum[1]*1000:+.1f}, {tf_sum[2]*1000:+.1f}] mm")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="debug_tracking.csv 시각화")
    parser.add_argument('csv_files', nargs='+', help="CSV 파일 경로(들)")
    parser.add_argument('--save', default=None, help="그래프 저장 경로")
    args = parser.parse_args()

    for csv_path in args.csv_files:
        if not os.path.exists(csv_path):
            print(f"File not found: {csv_path}")
            continue

        df = load_csv(csv_path)
        print_summary(df)

        title = os.path.dirname(csv_path).replace('eval_results/', '')
        save_path = args.save
        if save_path is None and len(args.csv_files) == 1:
            save_path = csv_path.replace('.csv', '_plot.png')

        plot_single(df, title=title, save_path=save_path)


if __name__ == '__main__':
    main()
