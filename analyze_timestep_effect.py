"""
Same noise, different timestep t → pred_x0 비교 실험
같은 x_T에서 custom_timesteps: [t] 로 1-step DDIM을 각 t에 대해 돌리고,
16-step 결과와 비교하여 scheduler의 t 컨디셔닝 효과를 시각화
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import yaml
import json
import csv
import cv2
import dill
import hydra
from pathlib import Path
from omegaconf import DictConfig, ListConfig, OmegaConf

os.environ.setdefault('QT_QPA_PLATFORM_PLUGIN_PATH', '/home/minhwan/anaconda3/envs/umi/plugins')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusion_policy.common.cv2_util import get_image_transform


def load_policy(ckpt_path, device='cuda'):
    """eval_robot.py의 load_model과 동일하게 모델 로드"""
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cfg.policy.obs_encoder.pretrained = False

    def patch_config(config_obj):
        if isinstance(config_obj, (dict, DictConfig)):
            for k in config_obj.keys():
                v = config_obj[k]
                if k == '_target_' and isinstance(v, str):
                    new_v = None
                    if 'train_diffusion_unet_image_workspace' in v:
                        new_v = 'diffusion_policy.workspace.train_manipforce_workspace.TrainManipForceWorkspace'
                    elif 'timm_obs_encoder' in v or 'TimmObsEncoder' in v:
                        new_v = 'diffusion_policy.model.vision.fmt_obs_encoder.FMTObsEncoder'
                    elif 'diffusion_transformer_timm_policy' in v:
                        new_v = 'diffusion_policy.policy.diffusion_transformer_timm_policy.DiffusionTransformerTimmPolicy'
                    if new_v and new_v != v:
                        config_obj[k] = new_v
                else:
                    patch_config(v)
        elif isinstance(config_obj, (list, ListConfig)):
            for item in config_obj:
                patch_config(item)

    patch_config(cfg)

    OmegaConf.set_struct(cfg, False)
    to_remove = []
    if hasattr(cfg, 'shape_meta') and 'obs' in cfg.shape_meta:
        for obs_key in list(cfg.shape_meta.obs.keys()):
            obs_cfg = cfg.shape_meta.obs[obs_key]
            if hasattr(obs_cfg, 'type') and obs_cfg.type == 'low_dim':
                has_weights = any(f'obs_encoder.low_dim_proj.{obs_key}' in k
                                 for k in payload['state_dicts'].get('model', {}).keys())
                if not has_weights:
                    to_remove.append(obs_key)
    for obs_key in to_remove:
        for getter in [
            lambda: cfg.shape_meta.obs,
            lambda: cfg.task.shape_meta.obs,
            lambda: cfg.policy.shape_meta.obs,
            lambda: cfg.policy.obs_encoder.shape_meta.obs,
        ]:
            try:
                obs_meta = getter()
                if obs_key in obs_meta:
                    del obs_meta[obs_key]
            except Exception:
                pass

    try:
        cls = hydra.utils.get_class(cfg._target_)
    except (ImportError, AttributeError):
        from diffusion_policy.workspace.train_manipforce_workspace import TrainManipForceWorkspace
        cls = TrainManipForceWorkspace

    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None, strict=False)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    policy.eval()
    policy.to(device)
    return policy, cfg


def single_step_predict(policy, obs_tokens, cond_data, cond_mask, noise, t_value):
    """같은 noise에서 특정 t 값으로 1-step DDIM 수행, pred_x0 반환"""
    scheduler = policy.noise_scheduler
    alphas_cumprod = scheduler.alphas_cumprod.to(noise.device)
    final_alpha_cumprod = scheduler.final_alpha_cumprod.to(noise.device)

    trajectory = noise.clone()
    trajectory[cond_mask] = cond_data[cond_mask]

    model_output = policy.model(trajectory, t_value, obs_tokens)

    alpha_prod_t = alphas_cumprod[t_value]
    beta_prod_t = 1 - alpha_prod_t

    if scheduler.config.prediction_type == "epsilon":
        pred_x0 = (trajectory - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    elif scheduler.config.prediction_type == "sample":
        pred_x0 = model_output
    else:
        raise ValueError(f"Unsupported: {scheduler.config.prediction_type}")

    if scheduler.config.clip_sample:
        pred_x0 = pred_x0.clamp(-scheduler.config.clip_sample_range, scheduler.config.clip_sample_range)

    # unnormalize
    pred_x0_unnorm = policy.normalizer['action'].unnormalize(pred_x0)
    return pred_x0_unnorm.detach().cpu()


def full_16step_predict(policy, obs_tokens, cond_data, cond_mask, noise):
    """같은 noise에서 16-step DDIM 수행, 최종 action 반환"""
    scheduler = policy.noise_scheduler
    scheduler.set_timesteps(16)
    trajectory = noise.clone()

    for t in scheduler.timesteps:
        trajectory[cond_mask] = cond_data[cond_mask]
        model_output = policy.model(trajectory, t, obs_tokens)
        trajectory = scheduler.step(model_output, t, trajectory).prev_sample

    trajectory[cond_mask] = cond_data[cond_mask]
    action = policy.normalizer['action'].unnormalize(trajectory)
    return action.detach().cpu()


def load_obs_from_episode(episode_path, frame_indices, shape_meta):
    """실제 에피소드 데이터에서 obs_dict 구성 (이미지 + FT)"""
    episode_path = Path(episode_path)

    with open(episode_path / 'pose_data.json') as f:
        pose_data = json.load(f)

    # FT 데이터 로드
    ft_csv_path = list(episode_path.glob('ft_data_*.csv'))[0]
    ft_timestamps_str = []
    ft_values = []
    with open(ft_csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            ft_timestamps_str.append(row[0])
            ft_values.append([float(x) for x in row[1:7]])
    ft_values = np.array(ft_values, dtype=np.float32)

    obs_shape_meta = shape_meta['obs']
    # 이미지 resolution 파악
    for key, attr in obs_shape_meta.items():
        if attr.get('type') == 'rgb':
            _, target_h, target_w = attr['shape']
            break

    # 이미지 로드
    handeye_imgs = []
    additional_imgs = []
    img_timestamps = []

    for idx in frame_indices:
        frame = pose_data[idx]
        img_file = frame['image_file']

        # handeye camera
        img_path = episode_path / 'images' / 'handeye' / img_file
        img = cv2.imread(str(img_path))
        hi, wi = img.shape[:2]
        tf = get_image_transform(input_res=(wi, hi), output_res=(target_w, target_h), bgr_to_rgb=True)
        img_out = tf(img).astype(np.float32) / 255.0
        img_out = np.moveaxis(img_out, -1, 0)  # HWC -> CHW
        handeye_imgs.append(img_out)

        # additional camera
        img_add_path = episode_path / 'images' / 'additional_cam' / img_file
        img_add = cv2.imread(str(img_add_path))
        hi2, wi2 = img_add.shape[:2]
        tf2 = get_image_transform(input_res=(wi2, hi2), output_res=(target_w, target_h), bgr_to_rgb=True)
        img_add_out = tf2(img_add).astype(np.float32) / 255.0
        img_add_out = np.moveaxis(img_add_out, -1, 0)
        additional_imgs.append(img_add_out)

        img_timestamps.append(frame['timestamp'])

    # FT 데이터: 이미지 타임스탬프 사이의 8프레임 추출
    target_ft_frames = 8
    # 간단하게: 두 번째 이미지 기준으로 직전 8프레임
    # (실제 eval에서도 비슷하게 최근 FT 데이터를 사용)
    ft_slice = ft_values[-target_ft_frames:] if len(ft_values) >= target_ft_frames else ft_values
    if len(ft_slice) < target_ft_frames:
        pad = np.tile(ft_slice[-1:], (target_ft_frames - len(ft_slice), 1))
        ft_slice = np.vstack([ft_slice, pad])

    # obs_dict 구성 (batch dim 포함)
    obs_dict = {
        'handeye_cam_1': torch.from_numpy(np.stack(handeye_imgs)).float().unsqueeze(0),      # (1, 2, 3, H, W)
        'handeye_cam_2': torch.from_numpy(np.stack(additional_imgs)).float().unsqueeze(0),    # (1, 2, 3, H, W)
        'ft_data': torch.from_numpy(ft_slice).float().unsqueeze(0),                           # (1, 8, 6)
        'ft_timestamps': torch.from_numpy(np.array(img_timestamps)).float().unsqueeze(0),     # (1, 2)
        'img_timestamps': torch.from_numpy(np.array(img_timestamps)).float().unsqueeze(0),    # (1, 2)
    }
    return obs_dict


def collect_real_obs_tokens(policy, cfg, data_dir, n_samples=50, device='cuda'):
    """여러 에피소드에서 실제 obs를 로드하고 obs_encoder를 통해 obs_tokens 생성"""
    data_dir = Path(data_dir)
    episodes = sorted([d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('episode_')])
    print(f"Found {len(episodes)} episodes in {data_dir}")

    shape_meta = OmegaConf.to_container(cfg.task.shape_meta, resolve=True)
    obs_tokens_list = []

    n_per_episode = max(1, n_samples // len(episodes))
    count = 0

    for ep_path in episodes:
        if count >= n_samples:
            break

        with open(ep_path / 'pose_data.json') as f:
            pose_data = json.load(f)
        n_frames = len(pose_data)
        if n_frames < 10:
            continue

        # 에피소드 내 여러 시점에서 샘플링
        sample_starts = np.linspace(5, n_frames - 5, n_per_episode, dtype=int)
        for start_idx in sample_starts:
            if count >= n_samples:
                break
            # horizon=2: 2프레임 사용
            frame_indices = [max(0, start_idx - 3), start_idx]

            try:
                obs_dict = load_obs_from_episode(ep_path, frame_indices, shape_meta)
                # obs_dict를 device로 이동
                obs_dict_dev = {}
                for k, v in obs_dict.items():
                    if isinstance(v, torch.Tensor):
                        obs_dict_dev[k] = v.to(device)
                    else:
                        obs_dict_dev[k] = v

                # normalizer 적용 (이미지 이외)
                obs_for_norm = {k: v for k, v in obs_dict_dev.items()
                                if k not in ['ft_timestamps', 'img_timestamps', 'ft_data']}
                nobs = policy.normalizer.normalize(obs_for_norm)
                if 'ft_data' in obs_dict_dev:
                    nobs['ft_data'] = obs_dict_dev['ft_data']
                    nobs['ft_timestamps'] = obs_dict_dev['ft_timestamps']

                # obs_encoder로 obs_tokens 생성
                obs_tokens = policy.obs_encoder(nobs)  # (1, N, n_emb)
                obs_tokens_list.append(obs_tokens.detach())
                count += 1
            except Exception as e:
                print(f"  Skip {ep_path.name} frame {start_idx}: {e}")
                continue

        if count % 10 == 0 and count > 0:
            print(f"  Collected {count}/{n_samples} obs_tokens")

    print(f"Total obs_tokens collected: {len(obs_tokens_list)}")
    return obs_tokens_list


@torch.no_grad()
def run_experiment_with_real_obs(policy, cfg, data_dir, n_trials=50, device='cuda'):
    """실제 obs로 같은 noise + 다른 t 실험"""
    test_timesteps = [99, 95, 90, 85, 80, 75, 70, 60, 50, 40, 30, 20, 10, 5, 0]

    # 실제 obs_tokens 수집
    obs_tokens_list = collect_real_obs_tokens(policy, cfg, data_dir, n_samples=n_trials, device=device)
    n_trials = len(obs_tokens_list)
    if n_trials == 0:
        raise RuntimeError("No obs_tokens collected. Check data_dir.")

    action_dim = policy.action_dim
    action_horizon = getattr(policy, 'action_horizon', getattr(policy, 'horizon', 8))
    B = 1

    cond_data = torch.zeros(B, action_horizon, action_dim, device=device)
    cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

    print(f"Running {n_trials} trials with REAL obs, {len(test_timesteps)} timesteps each...")

    from scipy.spatial.transform import Rotation as R

    all_pos_errors = {t: [] for t in test_timesteps}
    all_rot_errors = {t: [] for t in test_timesteps}

    for trial in range(n_trials):
        obs_tokens = obs_tokens_list[trial]

        # 같은 noise를 모든 t에 사용
        gen = torch.Generator(device=device).manual_seed(trial)
        noise = torch.randn(B, action_horizon, action_dim, device=device, generator=gen)

        # 16-step baseline
        action_16 = full_16step_predict(policy, obs_tokens, cond_data, cond_mask, noise)
        final_action = action_16[0, 0, :].numpy()  # first timestep of horizon

        for t in test_timesteps:
            pred = single_step_predict(policy, obs_tokens, cond_data, cond_mask, noise, t)
            pred_action = pred[0, 0, :].numpy()

            # position error (mm)
            pos_err = np.linalg.norm(pred_action[:3] - final_action[:3]) * 1000
            all_pos_errors[t].append(pos_err)

            # rotation error (deg)
            if len(pred_action) >= 7:
                try:
                    r_pred = R.from_quat(pred_action[3:7])
                    r_final = R.from_quat(final_action[3:7])
                    rot_err = np.degrees((r_pred.inv() * r_final).magnitude())
                except Exception:
                    rot_err = 0.0
            else:
                rot_err = 0.0
            all_rot_errors[t].append(rot_err)

        if (trial + 1) % 10 == 0:
            print(f"  trial {trial+1}/{n_trials}")

    return test_timesteps, all_pos_errors, all_rot_errors


@torch.no_grad()
def run_experiment_with_dummy_obs(policy, n_trials=50, device='cuda'):
    """Dummy obs로 같은 noise + 다른 t 실험 (비교용)"""
    test_timesteps = [99, 95, 90, 85, 80, 75, 70, 60, 50, 40, 30, 20, 10, 5, 0]

    B = 1
    cond_pos_shape = policy.model.cond_pos_emb.shape
    n_emb = cond_pos_shape[2]
    total_tokens = cond_pos_shape[1]
    action_dim = policy.action_dim
    action_horizon = getattr(policy, 'action_horizon', getattr(policy, 'horizon', 8))
    n_obs_tokens = total_tokens - action_horizon

    cond_data = torch.zeros(B, action_horizon, action_dim, device=device)
    cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

    torch.manual_seed(0)
    obs_tokens = torch.randn(B, n_obs_tokens, n_emb, device=device)

    print(f"Running {n_trials} trials with DUMMY obs, {len(test_timesteps)} timesteps each...")
    from scipy.spatial.transform import Rotation as R

    all_pos_errors = {t: [] for t in test_timesteps}
    all_rot_errors = {t: [] for t in test_timesteps}

    for trial in range(n_trials):
        gen = torch.Generator(device=device).manual_seed(trial)
        noise = torch.randn(B, action_horizon, action_dim, device=device, generator=gen)

        action_16 = full_16step_predict(policy, obs_tokens, cond_data, cond_mask, noise)
        final_action = action_16[0, 0, :].numpy()

        for t in test_timesteps:
            pred = single_step_predict(policy, obs_tokens, cond_data, cond_mask, noise, t)
            pred_action = pred[0, 0, :].numpy()

            pos_err = np.linalg.norm(pred_action[:3] - final_action[:3]) * 1000
            all_pos_errors[t].append(pos_err)

            if len(pred_action) >= 7:
                try:
                    r_pred = R.from_quat(pred_action[3:7])
                    r_final = R.from_quat(final_action[3:7])
                    rot_err = np.degrees((r_pred.inv() * r_final).magnitude())
                except Exception:
                    rot_err = 0.0
            else:
                rot_err = 0.0
            all_rot_errors[t].append(rot_err)

        if (trial + 1) % 10 == 0:
            print(f"  trial {trial+1}/{n_trials}")

    return test_timesteps, all_pos_errors, all_rot_errors


def plot_comparison(test_timesteps, real_pos, real_rot, dummy_pos, dummy_rot, out_path):
    """Real obs vs Dummy obs 비교: obs에 무관하게 동일한 패턴임을 보여주는 figure"""
    alphas = get_cosine_alphas_cumprod()
    alpha_values = [alphas[t] for t in test_timesteps]

    real_means_p = [np.mean(real_pos[t]) for t in test_timesteps]
    real_stds_p = [np.std(real_pos[t]) for t in test_timesteps]
    dummy_means_p = [np.mean(dummy_pos[t]) for t in test_timesteps]
    dummy_stds_p = [np.std(dummy_pos[t]) for t in test_timesteps]

    real_means_r = [np.mean(real_rot[t]) for t in test_timesteps]
    real_stds_r = [np.std(real_rot[t]) for t in test_timesteps]
    dummy_means_r = [np.mean(dummy_rot[t]) for t in test_timesteps]
    dummy_stds_r = [np.std(dummy_rot[t]) for t in test_timesteps]

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    ax1, ax2, ax3 = axes

    # --- Position Error: Real vs Dummy (bar chart) ---
    x = np.arange(len(test_timesteps))
    w = 0.35
    t_labels = [str(t) for t in test_timesteps]

    bars1 = ax1.bar(x - w/2, real_means_p, w, color='#4C72B0', alpha=0.8, label='Real Obs')
    ax1.errorbar(x - w/2, real_means_p, yerr=real_stds_p, fmt='none', color='#2c4a7c', capsize=2)
    bars2 = ax1.bar(x + w/2, dummy_means_p, w, color='#DD8452', alpha=0.8, label='Random Obs')
    ax1.errorbar(x + w/2, dummy_means_p, yerr=dummy_stds_p, fmt='none', color='#a0522d', capsize=2)

    # sweet spot 영역 표시
    sweet_indices = [i for i, t in enumerate(test_timesteps) if 50 <= t <= 80]
    if sweet_indices:
        ax1.axvspan(sweet_indices[0] - 0.5, sweet_indices[-1] + 0.5, color='green', alpha=0.08)

    ax1.set_xticks(x)
    ax1.set_xticklabels(t_labels, fontsize=8)
    ax1.set_xlabel('Timestep $t$')
    ax1.set_ylabel('Position Error vs 16-step (mm)')
    ax1.set_title('Position Error')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- Rotation Error: Real vs Dummy (bar chart) ---
    bars3 = ax2.bar(x - w/2, real_means_r, w, color='#4C72B0', alpha=0.8, label='Real Obs')
    ax2.errorbar(x - w/2, real_means_r, yerr=real_stds_r, fmt='none', color='#2c4a7c', capsize=2)
    bars4 = ax2.bar(x + w/2, dummy_means_r, w, color='#DD8452', alpha=0.8, label='Random Obs')
    ax2.errorbar(x + w/2, dummy_means_r, yerr=dummy_stds_r, fmt='none', color='#a0522d', capsize=2)

    if sweet_indices:
        ax2.axvspan(sweet_indices[0] - 0.5, sweet_indices[-1] + 0.5, color='green', alpha=0.08)

    ax2.set_xticks(x)
    ax2.set_xticklabels(t_labels, fontsize=8)
    ax2.set_xlabel('Timestep $t$')
    ax2.set_ylabel('Rotation Error vs 16-step (deg)')
    ax2.set_title('Rotation Error')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # --- Cosine Schedule + error overlay on alpha_bar scale ---
    t_all = np.arange(100)
    ax3.plot(t_all, alphas, color='#2ca02c', linewidth=2, alpha=0.4, label=r'$\bar{\alpha}_t$')
    ax3.fill_between(t_all, 0, alphas, color='#2ca02c', alpha=0.05)

    # position error를 cosine schedule 위에 scatter
    # normalize error to [0,1] for overlay
    max_err = max(max(real_means_p), max(dummy_means_p))
    real_norm = [e / max_err for e in real_means_p]
    dummy_norm = [e / max_err for e in dummy_means_p]

    ax3.scatter(test_timesteps, real_norm, color='#4C72B0', s=60, zorder=5, label='Real Obs (pos err, norm.)')
    ax3.scatter(test_timesteps, dummy_norm, color='#DD8452', s=60, marker='x', zorder=5, label='Random Obs (pos err, norm.)')

    # 두 곡선 연결
    ax3.plot(test_timesteps, real_norm, color='#4C72B0', alpha=0.5, linewidth=1.5)
    ax3.plot(test_timesteps, dummy_norm, color='#DD8452', alpha=0.5, linewidth=1.5, linestyle='--')

    ax3.set_xlabel('Timestep $t$')
    ax3.set_ylabel(r'$\bar{\alpha}_t$ / Normalized Error')
    ax3.set_xlim(0, 99)
    ax3.set_ylim(0, 1.1)
    ax3.invert_xaxis()
    ax3.set_title(r'Error Pattern Follows $\bar{\alpha}_t$, Not Obs Content')
    ax3.legend(fontsize=8, loc='center right')
    ax3.grid(True, alpha=0.3)
    ax3.spines['top'].set_visible(False)

    fig.suptitle('1-step DDIM Error is Determined by Cosine Schedule, Not Observation Content\n'
                 'Real observations and random tokens produce the same error pattern',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out_path}")


def get_cosine_alphas_cumprod(num_train_timesteps=100):
    """squaredcos_cap_v2 schedule 재현"""
    def alpha_bar(t):
        return np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2
    betas = []
    for i in range(num_train_timesteps):
        t1 = i / num_train_timesteps
        t2 = (i + 1) / num_train_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
    betas = np.array(betas)
    return np.cumprod(1.0 - betas)


def plot_timestep_effect(test_timesteps, pos_errors, rot_errors, out_path):
    """같은 noise, 다른 t에서의 action error + cosine schedule overlay"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    ax1, ax2, ax3 = axes

    x = np.arange(len(test_timesteps))
    t_labels = [f't={t}' for t in test_timesteps]

    # === Position Error ===
    means_p = [np.mean(pos_errors[t]) for t in test_timesteps]
    stds_p = [np.std(pos_errors[t]) for t in test_timesteps]
    ax1.bar(x, means_p, color='#4C72B0', alpha=0.7, width=0.6)
    ax1.errorbar(x, means_p, yerr=stds_p, fmt='none', color='black', capsize=3)
    for i, (m, s) in enumerate(zip(means_p, stds_p)):
        ax1.annotate(f'{m:.1f}', (i, m + s), textcoords='offset points',
                     xytext=(0, 4), ha='center', fontsize=7, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(t_labels, fontsize=8, rotation=45)
    ax1.set_ylabel('Position Error vs 16-step (mm)')
    ax1.set_title('Position Error')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # === Rotation Error ===
    means_r = [np.mean(rot_errors[t]) for t in test_timesteps]
    stds_r = [np.std(rot_errors[t]) for t in test_timesteps]
    ax2.bar(x, means_r, color='#DD8452', alpha=0.7, width=0.6)
    ax2.errorbar(x, means_r, yerr=stds_r, fmt='none', color='black', capsize=3)
    for i, (m, s) in enumerate(zip(means_r, stds_r)):
        ax2.annotate(f'{m:.1f}', (i, m + s), textcoords='offset points',
                     xytext=(0, 4), ha='center', fontsize=7, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(t_labels, fontsize=8, rotation=45)
    ax2.set_ylabel('Rotation Error vs 16-step (deg)')
    ax2.set_title('Rotation Error')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # === Cosine Schedule: alpha_bar & SNR ===
    alphas = get_cosine_alphas_cumprod()
    t_all = np.arange(100)
    snr = alphas / (1 - alphas + 1e-8)
    snr_db = 10 * np.log10(snr + 1e-8)

    # alpha_bar curve
    ax3.plot(t_all, alphas, color='#2ca02c', linewidth=2, label=r'$\bar{\alpha}_t$ (signal ratio)')
    ax3.fill_between(t_all, 0, alphas, color='#2ca02c', alpha=0.1)
    ax3.set_xlabel('Timestep t')
    ax3.set_ylabel(r'$\bar{\alpha}_t$', color='#2ca02c')
    ax3.tick_params(axis='y', labelcolor='#2ca02c')
    ax3.set_xlim(0, 99)
    ax3.set_ylim(0, 1.05)
    ax3.invert_xaxis()

    # SNR on secondary axis
    ax3r = ax3.twinx()
    ax3r.plot(t_all, snr_db, color='#9467bd', linewidth=2, linestyle='--', label='SNR (dB)')
    ax3r.set_ylabel('SNR (dB)', color='#9467bd')
    ax3r.tick_params(axis='y', labelcolor='#9467bd')

    # mark test timesteps
    for t in test_timesteps:
        ax3.axvline(x=t, color='gray', alpha=0.3, linewidth=0.8)

    # highlight sweet spot
    ax3.axvspan(45, 65, color='green', alpha=0.1)
    ax3.text(55, 0.95, 'sweet\nspot', ha='center', va='top', fontsize=9, color='green', fontweight='bold')

    ax3.set_title('Cosine Schedule')
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3r.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='center right')
    ax3.grid(True, alpha=0.3)
    ax3.spines['top'].set_visible(False)

    fig.suptitle('Same Noise, Different Timestep t: 1-step Action vs 16-step Action\n'
                 'Cosine schedule determines noise level at each t',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out_path}")

    # === 추가: x축을 alpha_bar (signal ratio)로 변환한 그래프 ===
    alphas = get_cosine_alphas_cumprod()
    alpha_values = [alphas[t] for t in test_timesteps]

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Position - x축: alpha_bar
    ax1.scatter(alpha_values, means_p, color='#4C72B0', s=80, zorder=5)
    ax1.errorbar(alpha_values, means_p, yerr=stds_p, fmt='none', color='black', capsize=3, zorder=4)
    for i, t in enumerate(test_timesteps):
        ax1.annotate(f't={t}', (alpha_values[i], means_p[i]),
                     textcoords='offset points', xytext=(5, 5), fontsize=7)
    ax1.set_xlabel(r'$\bar{\alpha}_t$ (signal ratio, 1=clean, 0=noise)')
    ax1.set_ylabel('Position Error vs 16-step (mm)')
    ax1.set_title('Position Error vs Signal Ratio')
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Rotation - x축: alpha_bar
    ax2.scatter(alpha_values, means_r, color='#DD8452', s=80, zorder=5)
    ax2.errorbar(alpha_values, means_r, yerr=stds_r, fmt='none', color='black', capsize=3, zorder=4)
    for i, t in enumerate(test_timesteps):
        ax2.annotate(f't={t}', (alpha_values[i], means_r[i]),
                     textcoords='offset points', xytext=(5, 5), fontsize=7)
    ax2.set_xlabel(r'$\bar{\alpha}_t$ (signal ratio, 1=clean, 0=noise)')
    ax2.set_ylabel('Rotation Error vs 16-step (deg)')
    ax2.set_title('Rotation Error vs Signal Ratio')
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    fig2.suptitle('Action Error vs Signal Ratio (cosine nonlinearity removed)\n'
                  r'x-axis = $\bar{\alpha}_t$ from cosine schedule',
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    out_path2 = out_path.replace('.png', '_linear_scale.png')
    plt.savefig(out_path2, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {out_path2}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', required=True, help='eval config yaml')
    parser.add_argument('--data_dir', '-d',
                        default='/home/ailab-2204/Workspace/ManipForce_v2/data/gear_assem',
                        help='episode data directory')
    parser.add_argument('--output', '-o', default='timestep_effect_real.png')
    parser.add_argument('--n_trials', '-n', type=int, default=50)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    ckpt_path = config['paths']['model_checkpoint']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading model from {ckpt_path}...")
    policy, cfg = load_policy(ckpt_path, device)
    print("Model loaded.")

    test_timesteps, pos_errors, rot_errors = run_experiment_with_real_obs(
        policy, cfg, args.data_dir, n_trials=args.n_trials, device=device)
    plot_timestep_effect(test_timesteps, pos_errors, rot_errors, args.output)
