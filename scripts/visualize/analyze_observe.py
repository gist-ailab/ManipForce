#!/usr/bin/env python3
"""
Observe CSV 분석 — 모델 방향 예측 검증
=======================================
여러 위치에서 수집한 observe CSV를 분석하여:
- EE 위치 vs 물체(target) 위치 offset
- 모델 예측 방향이 offset 반대(물체 쪽)인지 확인
- 좌표계 일관성 체크

사용법:
  # 물체 위치를 지정하고 분석
  python scripts/visualize/analyze_observe.py eval_results/observe/*.csv --target 0.45 0.0 0.2

  # 첫 번째 CSV의 EE 위치를 target으로 사용 (기본)
  python scripts/visualize/analyze_observe.py eval_results/observe/*.csv
"""

import argparse
import glob
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    """3D 화살표"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return min(zs)


def load_observe_csvs(csv_paths):
    """여러 CSV 로드, 각 CSV를 하나의 세션으로"""
    sessions = []
    for path in sorted(csv_paths):
        if not os.path.exists(path):
            print(f"[WARN] {path} not found, skipping")
            continue
        df = pd.read_csv(path)
        if len(df) == 0:
            continue

        # EE pose (세션 내 동일)
        ee_pos = df[['ee_x', 'ee_y', 'ee_z']].iloc[0].values
        ee_quat = df[['ee_qx', 'ee_qy', 'ee_qz', 'ee_qw']].iloc[0].values

        # 각 inference의 chunk sum (body & world)
        body_sums = []
        world_sums = []
        for inf_id, grp in df.groupby('inference'):
            body_sums.append(grp[['body_dx', 'body_dy', 'body_dz']].sum().values)
            world_sums.append(grp[['world_dx', 'world_dy', 'world_dz']].sum().values)

        sessions.append({
            'path': path,
            'name': os.path.basename(path).replace('.csv', ''),
            'ee_pos': ee_pos,
            'ee_quat': ee_quat,
            'body_sums': np.array(body_sums),
            'world_sums': np.array(world_sums),
            'n_inferences': len(body_sums),
            'df': df,
        })

    return sessions


def analyze_sessions(sessions, target_pos):
    """각 세션에서 EE→target offset과 모델 예측 방향을 비교"""
    print(f"\n{'='*90}")
    print(f"  Target (물체) 위치: x={target_pos[0]:.4f}  y={target_pos[1]:.4f}  z={target_pos[2]:.4f}")
    print(f"  세션 수: {len(sessions)}")
    print(f"{'='*90}\n")

    results = []

    for s in sessions:
        ee = s['ee_pos']
        offset = ee - target_pos  # EE가 target 대비 어디에 있는지
        offset_dir = offset / (np.linalg.norm(offset) + 1e-9)  # 단위벡터

        # 모델 예측 평균 (world frame)
        mean_world = s['world_sums'].mean(axis=0)
        mean_world_dir = mean_world / (np.linalg.norm(mean_world) + 1e-9)

        # 모델 예측 평균 (body frame)
        mean_body = s['body_sums'].mean(axis=0)

        # 방향 일치도: offset과 prediction이 반대여야 함
        # cos(angle) = -1이면 완벽히 반대 (정상)
        cos_sim = np.dot(offset_dir, mean_world_dir)
        # 반대 방향이면 cos < 0, 같은 방향이면 cos > 0
        is_correct = cos_sim < -0.3  # 대략 반대 방향

        # XY만 분석 (Z는 고정한다고 했으므로)
        offset_xy = offset[:2]
        pred_xy = mean_world[:2]
        if np.linalg.norm(offset_xy) > 1e-6 and np.linalg.norm(pred_xy) > 1e-6:
            cos_xy = np.dot(offset_xy / np.linalg.norm(offset_xy),
                           pred_xy / np.linalg.norm(pred_xy))
            xy_correct = cos_xy < -0.3
        else:
            cos_xy = 0
            xy_correct = None

        result = {
            'name': s['name'],
            'ee_pos': ee,
            'offset': offset,
            'offset_mm': offset * 1000,
            'mean_body': mean_body,
            'mean_world': mean_world,
            'cos_3d': cos_sim,
            'cos_xy': cos_xy,
            'correct_3d': is_correct,
            'correct_xy': xy_correct,
            'n_inferences': s['n_inferences'],
            'std_world': s['world_sums'].std(axis=0),
        }
        results.append(result)

        # 출력
        status_3d = "OK" if is_correct else "WRONG"
        status_xy = "OK" if xy_correct else ("WRONG" if xy_correct is not None else "N/A")

        print(f"  [{s['name']}] ({s['n_inferences']}회)")
        print(f"    EE 위치:      x={ee[0]:.4f}  y={ee[1]:.4f}  z={ee[2]:.4f}")
        print(f"    Target 대비:  dx={offset[0]*1000:+.1f}mm  dy={offset[1]*1000:+.1f}mm  dz={offset[2]*1000:+.1f}mm")

        # offset 방향 해석
        axis_names = ['X', 'Y', 'Z']
        dom_ax = np.argmax(np.abs(offset[:2]))  # XY 중 큰 축
        offset_desc = f"+{axis_names[dom_ax]}" if offset[dom_ax] > 0 else f"-{axis_names[dom_ax]}"
        expected_desc = f"-{axis_names[dom_ax]}" if offset[dom_ax] > 0 else f"+{axis_names[dom_ax]}"
        print(f"    → 물체 기준 {offset_desc} 방향에 위치 → 모델은 {expected_desc} 예측해야 정상")

        print(f"    모델 예측(body):  dx={mean_body[0]*1000:+.3f}mm  dy={mean_body[1]*1000:+.3f}mm  dz={mean_body[2]*1000:+.3f}mm")
        print(f"    모델 예측(world): dx={mean_world[0]*1000:+.3f}mm  dy={mean_world[1]*1000:+.3f}mm  dz={mean_world[2]*1000:+.3f}mm")
        print(f"    예측 std(world):  dx={result['std_world'][0]*1000:.3f}mm  dy={result['std_world'][1]*1000:.3f}mm  dz={result['std_world'][2]*1000:.3f}mm")

        # 실제 예측된 주된 방향
        pred_dom = np.argmax(np.abs(mean_world[:2]))
        pred_desc = f"+{axis_names[pred_dom]}" if mean_world[pred_dom] > 0 else f"-{axis_names[pred_dom]}"
        print(f"    모델 주된 방향: {pred_desc}")

        print(f"    방향 판정(XY): {status_xy} (cos={cos_xy:+.3f})")
        print(f"    방향 판정(3D): {status_3d} (cos={cos_sim:+.3f})")
        print()

    # 전체 요약
    print(f"{'='*90}")
    print(f"  ■ 전체 요약")
    n_ok_xy = sum(1 for r in results if r['correct_xy'] == True)
    n_wrong_xy = sum(1 for r in results if r['correct_xy'] == False)
    n_total = len(results)
    print(f"    XY 방향 정답률: {n_ok_xy}/{n_total} ({n_ok_xy/max(n_total,1)*100:.0f}%)")
    if n_wrong_xy > 0:
        print(f"    틀린 세션:")
        for r in results:
            if r['correct_xy'] == False:
                print(f"      - {r['name']}: offset={r['offset_mm'][:2]} → pred={r['mean_world'][:2]*1000}")

    # 좌표계 일관성 체크: body→world 방향 패턴
    print(f"\n  ■ 좌표계 체크")
    print(f"    body vs world 방향이 일관적인지:")
    for r in results:
        b = r['mean_body'][:2]
        w = r['mean_world'][:2]
        b_dom = np.argmax(np.abs(b))
        w_dom = np.argmax(np.abs(w))
        b_sign = '+' if b[b_dom] > 0 else '-'
        w_sign = '+' if w[w_dom] > 0 else '-'
        print(f"    {r['name']}: body 주축={b_sign}{'XY'[b_dom]}  world 주축={w_sign}{'XY'[w_dom]}")

    print(f"{'='*90}")
    return results


def plot_results(results, target_pos, save_path=None):
    """2D 조감도(XY)에 EE 위치, offset, 모델 예측 방향을 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, frame, title in [(axes[0], 'world', 'World Frame (robot base)'),
                              (axes[1], 'body', 'Body Frame (gripper)')]:
        # Target 위치
        ax.plot(target_pos[0] * 1000, target_pos[1] * 1000,
                'r*', markersize=20, zorder=10, label='Target (물체)')

        for i, r in enumerate(results):
            ee = r['ee_pos'] * 1000  # mm
            if frame == 'world':
                pred = r['mean_world'][:2] * 1000 * 50  # 시각화 스케일
            else:
                pred = r['mean_body'][:2] * 1000 * 50

            color = 'green' if r['correct_xy'] else 'red'

            # EE 위치
            ax.plot(ee[0], ee[1], 'o', color=color, markersize=10, zorder=5)
            ax.annotate(f'#{i+1}', (ee[0], ee[1]), textcoords="offset points",
                       xytext=(5, 5), fontsize=8)

            # 모델 예측 방향 화살표
            ax.annotate('', xy=(ee[0] + pred[0], ee[1] + pred[1]),
                       xytext=(ee[0], ee[1]),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2))

            # offset 방향 (점선, 회색)
            off = r['offset'][:2] * 1000
            off_norm = off / (np.linalg.norm(off) + 1e-9) * np.linalg.norm(pred)
            ax.annotate('', xy=(ee[0] - off_norm[0], ee[1] - off_norm[1]),
                       xytext=(ee[0], ee[1]),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1, ls='--'))

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    # 범례 설명
    fig.text(0.5, 0.02,
             '실선 화살표: 모델 예측 방향 (green=정답, red=오답) | 점선 화살표: 정답 방향 (물체 쪽)',
             ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n[SAVED] {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Observe CSV 분석")
    parser.add_argument('csv_files', nargs='+', help="observe CSV 파일들")
    parser.add_argument('--target', nargs=3, type=float, default=None,
                       help="물체(target) 위치 x y z (미터). 미지정시 첫 CSV 위치 사용")
    parser.add_argument('--save', default=None, help="그래프 저장 경로")
    args = parser.parse_args()

    # glob 확장
    csv_paths = []
    for p in args.csv_files:
        expanded = glob.glob(p)
        csv_paths.extend(expanded if expanded else [p])

    sessions = load_observe_csvs(csv_paths)
    if not sessions:
        print("No valid CSV files found")
        return

    # Target 위치
    if args.target:
        target_pos = np.array(args.target)
    else:
        # 첫 세션의 EE 위치를 target으로 (가장 물체에 가까운 위치에서 첫 측정했다고 가정)
        target_pos = sessions[0]['ee_pos'].copy()
        print(f"[INFO] --target 미지정. 첫 CSV의 EE 위치를 target으로 사용: {target_pos}")

    results = analyze_sessions(sessions, target_pos)

    save_path = args.save or 'eval_results/observe/observe_analysis.png'
    plot_results(results, target_pos, save_path=save_path)


if __name__ == '__main__':
    main()
