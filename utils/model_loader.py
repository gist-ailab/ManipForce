"""Model loading utilities for eval_robot."""

import torch
import dill
import hydra.utils
from omegaconf import OmegaConf, DictConfig, ListConfig
from diffusion_policy.workspace.base_workspace import BaseWorkspace


def load_model(ckpt_path, config, state):
    """Load model checkpoint and return (policy, cfg, rotation_repr, device).

    Args:
        ckpt_path: Path to model checkpoint file.
        config: Eval config dict.
        state: EvalState instance (updates gui_mode_label, gui_timesteps_str).
    """
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
                        print(f"[Patch] Changing {v} -> {new_v}")
                        config_obj[k] = new_v
                else:
                    patch_config(v)
        elif isinstance(config_obj, (list, ListConfig)):
            for item in config_obj:
                patch_config(item)

    patch_config(cfg)

    # Remove low_dim obs keys without weights
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
        print(f"[Patch] Removing low_dim obs '{obs_key}' (no weights in checkpoint)")
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
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None, strict=False)

    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    policy.num_inference_steps = config['model']['num_inference_steps']
    custom_ts = config.get('model', {}).get('custom_timesteps', None)
    num_steps = config['model']['num_inference_steps']

    if custom_ts is not None:
        policy.custom_timesteps = list(custom_ts)
        print(f"[INFO] Custom DDIM timesteps: {policy.custom_timesteps}")
        state.gui_mode_label = "c2f"
        state.gui_timesteps_str = f"timesteps: {custom_ts}"
    else:
        state.gui_mode_label = f"baseline {num_steps} step"
        state.gui_timesteps_str = f"DDIM {num_steps} steps"

    rotation_repr = getattr(cfg.task, 'rotation_repr', 'rotation_6d')
    print(f"[Config] rotation_repr={rotation_repr}, action_shape={cfg.task.shape_meta.action.shape}")

    device = torch.device('cuda')
    policy.eval().to(device)
    return policy, cfg, rotation_repr, device
