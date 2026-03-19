import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import csv

os.environ.setdefault('QT_QPA_PLATFORM_PLUGIN_PATH', '/home/minhwan/anaconda3/envs/umi/plugins')

# DDIM alphas_cumprod 복원 (squaredcos_cap_v2 schedule, num_train_timesteps=100)
def get_alphas_cumprod(num_train_timesteps=100):
    """diffusers squaredcos_cap_v2 beta schedule 재현"""
    def alpha_bar(t):
        return np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
    betas = []
    for i in range(num_train_timesteps):
        t1 = i / num_train_timesteps
        t2 = (i + 1) / num_train_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
    betas = np.array(betas)
    alphas = 1.0 - betas
    return np.cumprod(alphas)

ALPHAS_CUMPROD = get_alphas_cumprod()


def compute_velocity(pred_epsilon, pred_x0, timestep):
    """v = alpha_t * epsilon - sigma_t * x_0"""
    alpha_prod_t = ALPHAS_CUMPROD[timestep]
    alpha_t = np.sqrt(alpha_prod_t)
    sigma_t = np.sqrt(1 - alpha_prod_t)
    return alpha_t * pred_epsilon - sigma_t * pred_x0


def compute_metrics(data_slice, mode='pred_x0'):
    """inference step 리스트에서 cos_sim, l2_diff, norm 계산
    mode: 'pred_x0', 'pred_epsilon', 'velocity'
    """
    n_denoise = len(data_slice[0]['intermediates'])
    all_cos, all_diff, all_norm = [], [], []

    for entry in data_slice:
        if mode == 'pred_x0':
            preds = np.array([s['pred_x0_unnorm'][0, 0, :].numpy() for s in entry['intermediates']])
        elif mode == 'pred_epsilon':
            preds = np.array([s['pred_epsilon'][0, 0, :].numpy() for s in entry['intermediates']])
        elif mode == 'velocity':
            preds = np.array([
                compute_velocity(
                    s['pred_epsilon'][0, 0, :].numpy(),
                    s['pred_x0'][0, 0, :].numpy(),
                    s['timestep']
                ) for s in entry['intermediates']
            ])

        all_norm.append(np.linalg.norm(preds, axis=1))
        cos, diff = [], []
        for i in range(len(preds) - 1):
            a, b = preds[i], preds[i + 1]
            cos.append(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
            diff.append(np.linalg.norm(b - a))
        all_cos.append(cos)
        all_diff.append(diff)

    return np.array(all_cos), np.array(all_diff), np.array(all_norm)


def plot_phase(axes_row, cos_sims, diffs, norms, timesteps, phase_name, color):
    """1x3 row에 cos_sim, l2_diff, norm 그래프"""
    n_denoise = norms.shape[1]
    x = np.arange(n_denoise - 1)
    n = len(cos_sims)

    # Cosine Similarity
    ax = axes_row[0]
    mean_c = cos_sims.mean(axis=0)
    std_c = cos_sims.std(axis=0)
    ax.plot(x, mean_c, 'o-', color=color, linewidth=2, markersize=5, label=f'{phase_name} (n={n})')
    ax.fill_between(x, mean_c - std_c, np.minimum(mean_c + std_c, 1.0), alpha=0.2, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i+1}→{i+2}' for i in x], rotation=45, fontsize=7)
    ax.set_ylabel('Cosine Similarity')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # L2 Difference
    ax = axes_row[1]
    mean_d = diffs.mean(axis=0)
    std_d = diffs.std(axis=0)
    ax.plot(x, mean_d, 's-', color=color, linewidth=2, markersize=5, label=f'{phase_name} (n={n})')
    ax.fill_between(x, np.maximum(mean_d - std_d, 0), mean_d + std_d, alpha=0.2, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i+1}→{i+2}' for i in x], rotation=45, fontsize=7)
    ax.set_ylabel('L2 Distance')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Norm
    ax = axes_row[2]
    xn = np.arange(n_denoise)
    mean_n = norms.mean(axis=0)
    std_n = norms.std(axis=0)
    ax.plot(xn, mean_n, 'o-', color=color, linewidth=2, markersize=5, label=f'{phase_name} (n={n})')
    ax.fill_between(xn, mean_n - std_n, mean_n + std_n, alpha=0.2, color=color)
    ax.set_xticks(xn)
    ax.set_xticklabels([f'{i+1}\n(t={timesteps[i]})' for i in range(n_denoise)], fontsize=6)
    ax.set_ylabel('L2 Norm')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def find_reaching_steps(pt_path):
    """result.csv에서 이 pt 파일에 해당하는 reaching_steps 찾기"""
    basename = os.path.basename(pt_path)
    ts = basename.replace('diffusion_intermediates_', '').replace('.pt', '')

    result_dir = os.path.dirname(pt_path)
    csv_path = os.path.join(result_dir, 'result.csv')
    if not os.path.exists(csv_path):
        csv_path = 'result.csv'
    if not os.path.exists(csv_path):
        return None

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['timestamp'] == ts:
                return int(row['reaching_steps'])

    print(f"[WARN] timestamp {ts} not found in result.csv, trying closest match...")
    return None


def make_figure(data, reaching_data, contact_data, has_phases, timesteps, mode, mode_label):
    """하나의 mode(pred_x0/pred_epsilon/velocity)에 대해 3x3 또는 1x3 figure 생성"""
    if has_phases:
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))

        cos_all, diff_all, norm_all = compute_metrics(data, mode=mode)
        plot_phase(axes[0], cos_all, diff_all, norm_all, timesteps, 'All', '#333333')
        axes[0, 0].set_title(f'Cosine Similarity (All)')
        axes[0, 1].set_title(f'L2 Difference (All)')
        axes[0, 2].set_title(f'Norm of {mode_label} (All)')

        cos_r, diff_r, norm_r = compute_metrics(reaching_data, mode=mode)
        plot_phase(axes[1], cos_r, diff_r, norm_r, timesteps, 'Reaching', '#4C72B0')
        axes[1, 0].set_title(f'Cosine Similarity (Reaching)')
        axes[1, 1].set_title(f'L2 Difference (Reaching)')
        axes[1, 2].set_title(f'Norm of {mode_label} (Reaching)')

        cos_c, diff_c, norm_c = compute_metrics(contact_data, mode=mode)
        plot_phase(axes[2], cos_c, diff_c, norm_c, timesteps, 'Contact', '#DD8452')
        axes[2, 0].set_title(f'Cosine Similarity (Contact)')
        axes[2, 1].set_title(f'L2 Difference (Contact)')
        axes[2, 2].set_title(f'Norm of {mode_label} (Contact)')

    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        cos_all, diff_all, norm_all = compute_metrics(data, mode=mode)
        plot_phase([axes[0], axes[1], axes[2]], cos_all, diff_all, norm_all, timesteps, 'All', '#333333')
        axes[0].set_title('Cosine Similarity')
        axes[1].set_title('L2 Difference')
        axes[2].set_title(f'Norm of {mode_label}')

    return fig


def plot_convergence(data, reaching_data, contact_data, has_phases, timesteps, pt_path):
    """각 denoising step의 pred_x0 vs 최종 16-step action: position(mm), rotation(deg) 분리"""
    from scipy.spatial.transform import Rotation as R

    def _compute_errors(data_slice):
        all_pos_err = []
        all_rot_err = []
        for entry in data_slice:
            preds = np.array([s['pred_x0_unnorm'][0, 0, :].numpy() for s in entry['intermediates']])
            final = preds[-1]
            pos_errs = []
            rot_errs = []
            for k in range(len(preds)):
                # position error (mm)
                pos_errs.append(np.linalg.norm(preds[k, :3] - final[:3]) * 1000)
                # rotation error (deg) - quaternion 차이
                if preds.shape[1] >= 7:
                    try:
                        r_k = R.from_quat(preds[k, 3:7])
                        r_final = R.from_quat(final[3:7])
                        rot_diff = (r_k.inv() * r_final).magnitude()
                        rot_errs.append(np.degrees(rot_diff))
                    except Exception:
                        rot_errs.append(0.0)
                else:
                    rot_errs.append(0.0)
            all_pos_err.append(pos_errs)
            all_rot_err.append(rot_errs)
        return np.array(all_pos_err), np.array(all_rot_err)

    n_denoise = len(timesteps)
    x = np.arange(n_denoise)
    x_labels = [f't={timesteps[k]}' for k in range(n_denoise)]

    # 2 rows: position / rotation, columns: All / Reaching / Contact (or just All)
    n_cols = 3 if has_phases else 1
    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 9))
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    if has_phases:
        groups = [
            (data, 'All', '#333333', 0),
            (reaching_data, 'Reaching', '#4C72B0', 1),
            (contact_data, 'Contact', '#DD8452', 2),
        ]
    else:
        groups = [(data, 'All', '#333333', 0)]

    for data_slice, label, color, col in groups:
        pos_err, rot_err = _compute_errors(data_slice)
        n = len(pos_err)

        # Position (mm)
        ax = axes[0, col]
        mean_p = pos_err.mean(axis=0)
        std_p = pos_err.std(axis=0)
        ax.bar(x, mean_p, color=color, alpha=0.7, width=0.6)
        ax.errorbar(x, mean_p, yerr=std_p, fmt='none', color='black', capsize=3, linewidth=1)
        for k in range(n_denoise):
            ax.annotate(f'{mean_p[k]:.2f}', (k, mean_p[k] + std_p[k]),
                        textcoords='offset points', xytext=(0, 4), ha='center', fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=7, rotation=45)
        ax.set_ylabel('Position Error (mm)')
        ax.set_title(f'{label} (n={n})')
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Rotation (deg)
        ax = axes[1, col]
        mean_r = rot_err.mean(axis=0)
        std_r = rot_err.std(axis=0)
        ax.bar(x, mean_r, color=color, alpha=0.7, width=0.6)
        ax.errorbar(x, mean_r, yerr=std_r, fmt='none', color='black', capsize=3, linewidth=1)
        for k in range(n_denoise):
            ax.annotate(f'{mean_r[k]:.2f}', (k, mean_r[k] + std_r[k]),
                        textcoords='offset points', xytext=(0, 4), ha='center', fontsize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=7, rotation=45)
        ax.set_ylabel('Rotation Error (deg)')
        ax.grid(True, alpha=0.3, axis='y')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(f'Action Error at Each Denoising Step vs Final 16-step Action\n{os.path.basename(pt_path)}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = pt_path.replace('.pt', '_convergence.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out_path}")


def plot_action_trajectory(data, timesteps, pt_path, example_indices=None):
    """특정 inference step에서 denoising step별 pred_x0를 action dimension별로 overlay"""
    n_infer = len(data)
    if example_indices is None:
        # reaching 초반, 중간, contact 초반 3개
        example_indices = [0, n_infer // 4, n_infer // 2]
    example_indices = [i for i in example_indices if i < n_infer]

    dim_labels = ['dx', 'dy', 'dz', 'drx', 'dry', 'drz', 'grip', 'd7', 'd8', 'd9']
    n_dims = data[0]['intermediates'][0]['pred_x0_unnorm'].shape[-1]
    dim_labels = dim_labels[:n_dims]
    n_denoise = len(timesteps)
    cmap = plt.cm.viridis(np.linspace(0, 1, n_denoise))

    fig, axes = plt.subplots(len(example_indices), min(n_dims, 6), figsize=(20, 4 * len(example_indices)))
    if len(example_indices) == 1:
        axes = axes[np.newaxis, :]
    show_dims = min(n_dims, 6)

    for row, idx in enumerate(example_indices):
        entry = data[idx]
        preds = np.array([s['pred_x0_unnorm'][0, 0, :].numpy() for s in entry['intermediates']])

        for col in range(show_dims):
            ax = axes[row, col]
            for k in range(n_denoise):
                alpha = 0.3 if k < n_denoise - 1 else 1.0
                lw = 1 if k < n_denoise - 1 else 2.5
                ax.axhline(y=preds[k, col], color=cmap[k], alpha=alpha, linewidth=lw,
                           label=f't={timesteps[k]}' if col == 0 else '')
            ax.set_title(f'{dim_labels[col]}' if row == 0 else '', fontsize=10)
            ax.set_ylabel(f'Infer step {idx}' if col == 0 else '', fontsize=9)
            ax.tick_params(left=False, labelleft=True, bottom=False, labelbottom=False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.2)

    # colorbar legend
    axes[0, 0].legend(fontsize=5, loc='upper left', ncol=2)

    fig.suptitle(f'Action pred_x0 per denoising step (each horizontal line = one step)\n{os.path.basename(pt_path)}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = pt_path.replace('.pt', '_action_trajectory.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out_path}")


def plot_schedule_effect(data, timesteps, pt_path):
    """Cosine schedule의 비선형성이 만드는 착시를 보여주는 비교 figure.
    Left: x축 = denoising step (기존) → 초반 급격한 변화처럼 보임
    Right: x축 = α_bar (linear scale) → 실제로는 균일한 변화
    """
    from scipy.spatial.transform import Rotation as R

    n_denoise = len(timesteps)
    alpha_bars = [ALPHAS_CUMPROD[t] for t in timesteps]

    # 각 step의 pred_x0 vs 최종 action 간 cos_sim, L2 diff 계산
    all_cos_to_final = []
    all_l2_to_final = []
    all_pos_err = []

    for entry in data:
        preds = np.array([s['pred_x0_unnorm'][0, 0, :].numpy() for s in entry['intermediates']])
        final = preds[-1]

        cos_vals = []
        l2_vals = []
        pos_vals = []
        for k in range(n_denoise):
            a = preds[k]
            cos_vals.append(np.dot(a, final) / (np.linalg.norm(a) * np.linalg.norm(final) + 1e-8))
            l2_vals.append(np.linalg.norm(a - final))
            pos_vals.append(np.linalg.norm(a[:3] - final[:3]) * 1000)  # mm
        all_cos_to_final.append(cos_vals)
        all_l2_to_final.append(l2_vals)
        all_pos_err.append(pos_vals)

    all_cos_to_final = np.array(all_cos_to_final)
    all_l2_to_final = np.array(all_l2_to_final)
    all_pos_err = np.array(all_pos_err)

    mean_cos = all_cos_to_final.mean(axis=0)
    std_cos = all_cos_to_final.std(axis=0)
    mean_l2 = all_l2_to_final.mean(axis=0)
    std_l2 = all_l2_to_final.std(axis=0)
    mean_pos = all_pos_err.mean(axis=0)
    std_pos = all_pos_err.std(axis=0)

    # step간 cos_sim (consecutive)
    all_cos_consec = []
    for entry in data:
        if 'pred_x0' in entry['intermediates'][0]:
            preds = np.array([s['pred_x0_unnorm'][0, 0, :].numpy() for s in entry['intermediates']])
        cos_c = []
        for i in range(len(preds) - 1):
            a, b = preds[i], preds[i + 1]
            cos_c.append(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
        all_cos_consec.append(cos_c)
    all_cos_consec = np.array(all_cos_consec)
    mean_cos_c = all_cos_consec.mean(axis=0)
    std_cos_c = all_cos_consec.std(axis=0)

    # === Figure: 2x2 비교 ===
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # -- Top-left: Cosine Sim (consecutive) x축=step --
    ax = axes[0, 0]
    x_step = np.arange(n_denoise - 1)
    ax.plot(x_step, mean_cos_c, 'o-', color='#4C72B0', linewidth=2, markersize=6)
    ax.fill_between(x_step, mean_cos_c - std_cos_c, np.minimum(mean_cos_c + std_cos_c, 1.0),
                    alpha=0.2, color='#4C72B0')
    ax.set_xticks(x_step)
    ax.set_xticklabels([f'{timesteps[i]}→{timesteps[i+1]}' for i in x_step], rotation=45, fontsize=7)
    ax.set_xlabel('Denoising Step (timestep)')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('x-axis: Denoising Step (Cosine Schedule)')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 초반 "급격한 변화" 영역 강조
    ax.axvspan(-0.5, 3.5, color='red', alpha=0.05)
    ax.annotate('Appears to change\ndramatically here', xy=(1.5, mean_cos_c[1]),
                fontsize=9, color='red', fontweight='bold', ha='center',
                xytext=(3, mean_cos_c[1] - 0.001),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    # -- Top-right: Cosine Sim (consecutive) x축=α_bar --
    ax = axes[0, 1]
    # midpoint alpha_bar for consecutive pairs
    mid_alphas = [(alpha_bars[i] + alpha_bars[i+1]) / 2 for i in range(n_denoise - 1)]
    ax.plot(mid_alphas, mean_cos_c, 'o-', color='#2ca02c', linewidth=2, markersize=6)
    ax.fill_between(mid_alphas, mean_cos_c - std_cos_c, np.minimum(mean_cos_c + std_cos_c, 1.0),
                    alpha=0.2, color='#2ca02c')
    for i in range(len(mid_alphas)):
        ax.annotate(f't={timesteps[i]}', (mid_alphas[i], mean_cos_c[i]),
                    textcoords='offset points', xytext=(0, 8), fontsize=6, ha='center')
    ax.set_xlabel(r'$\bar{\alpha}_t$ (signal ratio)')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title(r'x-axis: $\bar{\alpha}_t$ (Linear Signal Scale)')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.annotate('Actually uniform\nchange across signal levels', xy=(0.5, mean_cos_c[len(mid_alphas)//2]),
                fontsize=9, color='green', fontweight='bold', ha='center')

    # -- Bottom-left: Position Error vs Final, x축=step --
    ax = axes[1, 0]
    ax.bar(np.arange(n_denoise), mean_pos, color='#4C72B0', alpha=0.7, width=0.6)
    ax.errorbar(np.arange(n_denoise), mean_pos, yerr=std_pos, fmt='none', color='black', capsize=3)
    ax.set_xticks(np.arange(n_denoise))
    ax.set_xticklabels([f't={t}' for t in timesteps], fontsize=7, rotation=45)
    ax.set_xlabel('Denoising Step (timestep)')
    ax.set_ylabel('Position Error vs Final (mm)')
    ax.set_title('Position Error: Step-based x-axis')
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axvspan(-0.5, 3.5, color='red', alpha=0.05)

    # -- Bottom-right: Position Error vs Final, x축=α_bar --
    ax = axes[1, 1]
    ax.scatter(alpha_bars, mean_pos, color='#2ca02c', s=60, zorder=5)
    ax.errorbar(alpha_bars, mean_pos, yerr=std_pos, fmt='none', color='black', capsize=3, zorder=4)
    for i, t in enumerate(timesteps):
        ax.annotate(f't={t}', (alpha_bars[i], mean_pos[i]),
                    textcoords='offset points', xytext=(5, 5), fontsize=6)
    ax.set_xlabel(r'$\bar{\alpha}_t$ (signal ratio, 0=noise, 1=clean)')
    ax.set_ylabel('Position Error vs Final (mm)')
    ax.set_title(r'Position Error: $\bar{\alpha}_t$ x-axis')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.annotate('Smooth, monotonic decrease\n(no dramatic early change)', xy=(0.5, mean_pos[len(alpha_bars)//2]),
                fontsize=9, color='green', fontweight='bold', ha='center')

    fig.suptitle('The "Dramatic Early Change" is a Cosine Schedule Artifact\n'
                 'Left: nonlinear timestep spacing exaggerates early steps  |  '
                 'Right: linear signal scale reveals uniform convergence',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path = pt_path.replace('.pt', '_schedule_effect.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out_path}")

    # === 추가: Δα_bar 정규화 figure ===
    # "각 step이 하는 일의 양"을 Δα_bar로 나누면 균일해진다
    delta_alphas = [alpha_bars[i+1] - alpha_bars[i] for i in range(n_denoise - 1)]

    # L2 change between consecutive pred_x0
    all_l2_consec = []
    for entry in data:
        preds = np.array([s['pred_x0_unnorm'][0, 0, :].numpy() for s in entry['intermediates']])
        l2_c = [np.linalg.norm(preds[i+1] - preds[i]) for i in range(len(preds) - 1)]
        all_l2_consec.append(l2_c)
    all_l2_consec = np.array(all_l2_consec)
    mean_l2_c = all_l2_consec.mean(axis=0)
    std_l2_c = all_l2_consec.std(axis=0)

    # normalized by Δα_bar
    mean_l2_norm = mean_l2_c / (np.abs(delta_alphas) + 1e-8)
    std_l2_norm = std_l2_c / (np.abs(delta_alphas) + 1e-8)

    # 1 - cos_sim
    mean_dissim = 1 - mean_cos_c
    mean_dissim_norm = mean_dissim / (np.abs(delta_alphas) + 1e-8)
    std_dissim_norm = std_cos_c / (np.abs(delta_alphas) + 1e-8)

    fig2, axes2 = plt.subplots(2, 3, figsize=(20, 10))
    step_labels = [f'{timesteps[i]}→{timesteps[i+1]}' for i in range(n_denoise - 1)]
    x_step = np.arange(n_denoise - 1)

    # Row 0: Raw values
    ax = axes2[0, 0]
    ax.bar(x_step, mean_l2_c, color='#4C72B0', alpha=0.7, width=0.6)
    ax.errorbar(x_step, mean_l2_c, yerr=std_l2_c, fmt='none', color='black', capsize=2)
    ax.set_xticks(x_step)
    ax.set_xticklabels(step_labels, rotation=45, fontsize=6)
    ax.set_ylabel('L2 Change')
    ax.set_title('L2 Change per Step (raw)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes2[0, 1]
    ax.bar(x_step, mean_dissim, color='#DD8452', alpha=0.7, width=0.6)
    ax.errorbar(x_step, mean_dissim, yerr=std_cos_c, fmt='none', color='black', capsize=2)
    ax.set_xticks(x_step)
    ax.set_xticklabels(step_labels, rotation=45, fontsize=6)
    ax.set_ylabel('1 - Cosine Similarity')
    ax.set_title('Dissimilarity per Step (raw)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes2[0, 2]
    ax.bar(x_step, np.abs(delta_alphas), color='#2ca02c', alpha=0.7, width=0.6)
    ax.set_xticks(x_step)
    ax.set_xticklabels(step_labels, rotation=45, fontsize=6)
    ax.set_ylabel(r'$|\Delta\bar{\alpha}|$')
    ax.set_title(r'Signal Change ($\Delta\bar{\alpha}_t$) per Step')
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.annotate('Early steps cover\nmuch larger signal range', xy=(1, np.abs(delta_alphas[1])),
                fontsize=8, color='#2ca02c', fontweight='bold',
                xytext=(5, np.abs(delta_alphas[1]) * 0.7),
                arrowprops=dict(arrowstyle='->', color='#2ca02c'))

    # Row 1: Normalized by Δα_bar
    ax = axes2[1, 0]
    ax.bar(x_step, mean_l2_norm, color='#4C72B0', alpha=0.7, width=0.6)
    ax.errorbar(x_step, mean_l2_norm, yerr=std_l2_norm, fmt='none', color='black', capsize=2)
    ax.set_xticks(x_step)
    ax.set_xticklabels(step_labels, rotation=45, fontsize=6)
    ax.set_ylabel(r'L2 / $|\Delta\bar{\alpha}|$')
    ax.set_title(r'L2 Change / $\Delta\bar{\alpha}$ (normalized)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax = axes2[1, 1]
    ax.bar(x_step, mean_dissim_norm, color='#DD8452', alpha=0.7, width=0.6)
    ax.errorbar(x_step, mean_dissim_norm, yerr=std_dissim_norm, fmt='none', color='black', capsize=2)
    ax.set_xticks(x_step)
    ax.set_xticklabels(step_labels, rotation=45, fontsize=6)
    ax.set_ylabel(r'(1-cos) / $|\Delta\bar{\alpha}|$')
    ax.set_title(r'Dissimilarity / $\Delta\bar{\alpha}$ (normalized)')
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # scatter: L2 change vs Δα_bar → linear relationship
    ax = axes2[1, 2]
    ax.scatter(np.abs(delta_alphas), mean_l2_c, color='#4C72B0', s=60, zorder=5)
    ax.errorbar(np.abs(delta_alphas), mean_l2_c, yerr=std_l2_c, fmt='none', color='black', capsize=3, zorder=4)
    for i in range(len(delta_alphas)):
        ax.annotate(f't={timesteps[i]}', (np.abs(delta_alphas[i]), mean_l2_c[i]),
                    textcoords='offset points', xytext=(5, 3), fontsize=6)
    coeffs = np.polyfit(np.abs(delta_alphas), mean_l2_c, 1)
    fit_x = np.linspace(0, max(np.abs(delta_alphas)) * 1.1, 50)
    r_val = np.corrcoef(np.abs(delta_alphas), mean_l2_c)[0, 1]
    ax.plot(fit_x, np.polyval(coeffs, fit_x), '--', color='red', linewidth=2, alpha=0.7,
            label=f'Linear fit (R={r_val:.3f})')
    ax.set_xlabel(r'$|\Delta\bar{\alpha}|$ (signal change)')
    ax.set_ylabel('L2 Change')
    ax.set_title(r'L2 Change $\propto$ $\Delta\bar{\alpha}$ (linear)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig2.suptitle('Denoising Change per Step is Proportional to Signal Change\n'
                  'Top: raw values show large early changes  |  '
                  r'Bottom: normalized by $\Delta\bar{\alpha}$ → uniform work per signal unit',
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path2 = pt_path.replace('.pt', '_schedule_normalized.png')
    plt.savefig(out_path2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out_path2}")


def analyze(pt_path):
    data = torch.load(pt_path, map_location='cpu')
    n_infer = len(data)
    n_denoise = len(data[0]['intermediates'])
    timesteps = [e['timestep'] for e in data[0]['intermediates']]
    print(f"Loaded: {pt_path}")
    print(f"Inference steps: {n_infer}, Denoising steps: {n_denoise}, timesteps: {timesteps}")

    reaching_steps = find_reaching_steps(pt_path)
    if reaching_steps and reaching_steps > 0 and reaching_steps < n_infer:
        print(f"Reaching steps: {reaching_steps}, Contact steps: {n_infer - reaching_steps}")
        reaching_data = data[:reaching_steps]
        contact_data = data[reaching_steps:]
        has_phases = True
    else:
        print(f"Phase split not available (reaching_steps={reaching_steps}), showing all")
        reaching_data, contact_data = None, None
        has_phases = False

    modes = [
        ('pred_x0', 'pred_x0'),
        ('pred_epsilon', 'pred_ε'),
        ('velocity', 'velocity'),
    ]

    # Convergence & action trajectory plots
    plot_convergence(data, reaching_data, contact_data, has_phases, timesteps, pt_path)
    plot_action_trajectory(data, timesteps, pt_path)
    plot_schedule_effect(data, timesteps, pt_path)

    for mode, mode_label in modes:
        print(f"\n── {mode_label} analysis ──")
        fig = make_figure(data, reaching_data, contact_data, has_phases, timesteps, mode, mode_label)

        fig.suptitle(f'Diffusion Denoising Analysis [{mode_label}] — {os.path.basename(pt_path)}\n'
                     f'({n_infer} inference steps × {n_denoise} denoising steps)',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        out_path = pt_path.replace('.pt', f'_{mode}_analysis.png')
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"Saved: {out_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Analyze diffusion denoising intermediates')
    parser.add_argument('pt_file', nargs='?', default=None, help='Path to .pt file')
    parser.add_argument('--dir', '-d', default='eval_results', help='Directory to search for .pt files (default: eval_results)')
    args = parser.parse_args()

    if args.pt_file:
        pt_path = args.pt_file
    else:
        import glob
        pts = sorted(glob.glob(os.path.join(args.dir, 'diffusion_intermediates_*.pt')))
        if not pts:
            print(f"No .pt files found in {args.dir}/")
            sys.exit(1)
        pt_path = pts[-1]
        print(f"Using latest: {pt_path}")

    analyze(pt_path)
