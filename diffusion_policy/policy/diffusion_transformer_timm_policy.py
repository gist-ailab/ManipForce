from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_action_diffusion import TransformerForActionDiffusion
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.vision.fmt_obs_encoder import FMTObsEncoder


class DiffusionTransformerTimmPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: FMTObsEncoder,
            num_inference_steps=None,
            input_pertub=0.1,
            # arch
            n_layer=7,
            n_head=8,
            n_emb=768,
            p_drop_attn=0.1,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']
        
        obs_shape = obs_encoder.output_shape()
        assert obs_shape[-1] == n_emb
        obs_tokens = obs_shape[-2]
        
        model = TransformerForActionDiffusion(
            input_dim=action_dim,
            output_dim=action_dim,
            action_horizon=action_horizon,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            max_cond_tokens=obs_tokens+1, # obs tokens + 1 token for time
            p_drop_attn=p_drop_attn
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.input_pertub = input_pertub
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.custom_timesteps = None  # 사용자 지정 timestep (예: [90, 0])
    
    # ========= inference  ============
    def conditional_sample(self,
            condition_data, condition_mask,
            cond=None, generator=None,
            return_intermediates=False,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        # 중간 결과 저장용
        intermediates = [] if return_intermediates else None

        if self.custom_timesteps is not None:
            # Custom timesteps: 직접 DDIM step 수행 (scheduler.step의 prev_timestep 계산이 고정 간격이라 맞지 않음)
            timesteps = self.custom_timesteps
            alphas_cumprod = scheduler.alphas_cumprod.to(device=condition_data.device)
            final_alpha_cumprod = scheduler.final_alpha_cumprod.to(device=condition_data.device)

            for i, t in enumerate(timesteps):
                # 1. apply conditioning
                trajectory[condition_mask] = condition_data[condition_mask]

                # 2. predict model output
                model_output = model(trajectory, t, cond)

                # 3. 올바른 prev_timestep 계산: timestep 리스트의 다음 값 사용
                if i + 1 < len(timesteps):
                    prev_t = timesteps[i + 1]
                else:
                    prev_t = -1  # 마지막 step

                # DDIM step 수식 직접 적용
                alpha_prod_t = alphas_cumprod[t]
                alpha_prod_t_prev = alphas_cumprod[prev_t] if prev_t >= 0 else final_alpha_cumprod
                beta_prod_t = 1 - alpha_prod_t

                # predicted x_0
                if scheduler.config.prediction_type == "epsilon":
                    pred_original_sample = (trajectory - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
                elif scheduler.config.prediction_type == "sample":
                    pred_original_sample = model_output
                else:
                    raise ValueError(f"Unsupported prediction_type: {scheduler.config.prediction_type}")

                # clip predicted x_0
                if scheduler.config.clip_sample:
                    pred_original_sample = pred_original_sample.clamp(
                        -scheduler.config.clip_sample_range, scheduler.config.clip_sample_range)

                # recompute epsilon from clipped x_0
                pred_epsilon = (trajectory - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5

                # direction pointing to x_t (eta=0, deterministic DDIM)
                pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * pred_epsilon

                # x_{t-1}
                trajectory = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

                if return_intermediates:
                    intermediates.append({
                        'step': i,
                        'timestep': int(t),
                        'x_t': trajectory.detach().cpu(),
                        'pred_x0': pred_original_sample.detach().cpu(),
                        'pred_epsilon': model_output.detach().cpu(),
                    })
        else:
            # 기존 방식: scheduler.set_timesteps + scheduler.step
            scheduler.set_timesteps(self.num_inference_steps)

            for i, t in enumerate(scheduler.timesteps):
                trajectory[condition_mask] = condition_data[condition_mask]
                model_output = model(trajectory, t, cond)

                if return_intermediates:
                    # pred_x0 계산
                    alpha_prod_t = scheduler.alphas_cumprod[t]
                    beta_prod_t = 1 - alpha_prod_t
                    if scheduler.config.prediction_type == "epsilon":
                        pred_x0 = (trajectory - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
                    else:
                        pred_x0 = model_output
                    intermediates.append({
                        'step': i,
                        'timestep': int(t),
                        'x_t': trajectory.detach().cpu(),
                        'pred_x0': pred_x0.detach().cpu(),
                        'pred_epsilon': model_output.detach().cpu(),
                    })

                trajectory = scheduler.step(
                    model_output, t, trajectory,
                    generator=generator,
                    **kwargs
                    ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        if return_intermediates:
            return trajectory, intermediates
        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], return_intermediates=False) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        obs_for_normalizer = {k: v for k, v in obs_dict.items()
                              if k not in ['ft_timestamps', 'img_timestamps', 'ft_data']}
        nobs = self.normalizer.normalize(obs_for_normalizer)
        if 'ft_data' in obs_dict:
            nobs['ft_data'] = obs_dict['ft_data']  # use unnormalized original data
            nobs['ft_timestamps'] = obs_dict['ft_timestamps']

        B = next(iter(nobs.values())).shape[0]

        # process input
        obs_tokens = self.obs_encoder(nobs)
        # (B, N, n_emb)

        # empty data for action
        cond_data = torch.zeros(size=(B, self.action_horizon, self.action_dim), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        sample_result = self.conditional_sample(
            condition_data=cond_data,
            condition_mask=cond_mask,
            cond=obs_tokens,
            return_intermediates=return_intermediates,
            **self.kwargs)

        if return_intermediates:
            nsample, intermediates = sample_result
        else:
            nsample = sample_result

        # unnormalize prediction
        assert nsample.shape == (B, self.action_horizon, self.action_dim)
        action_pred = self.normalizer['action'].unnormalize(nsample)

        result = {
            'action': action_pred,
            'action_pred': action_pred
        }

        if return_intermediates:
            # unnormalize intermediate pred_x0 values too
            for entry in intermediates:
                entry['pred_x0_unnorm'] = self.normalizer['action'].unnormalize(
                    entry['pred_x0'].to(self.device)).detach().cpu()
            result['intermediates'] = intermediates

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, 
            lr: float,
            weight_decay: float,
            obs_encoder_lr: float,
            obs_encoder_weight_decay: float,
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=weight_decay)
        
        backbone_params = list()
        other_obs_params = list()
        for key, value in self.obs_encoder.named_parameters():
            if key.startswith('key_model_map'):
                backbone_params.append(value)
            else:
                other_obs_params.append(value)
        optim_groups.append({
            "params": backbone_params,
            "weight_decay": obs_encoder_weight_decay,
            "lr": obs_encoder_lr # for fine tuning
        })
        optim_groups.append({
            "params": other_obs_params,
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=lr, betas=betas
        )
        return optimizer

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        obs = batch['obs']
        
        # 🆕 Masking flags and ft_data are excluded from normalization
        obs_for_normalizer = {
            k: v for k, v in obs.items() 
            if k not in ['ft_timestamps', 'img_timestamps', 'ft_data']  # Add masking flag exclusion
        }
        
        nobs = self.normalizer.normalize(obs_for_normalizer)

        # ft_data and masking info are added separately
        if 'ft_data' in obs:
            nobs['ft_data'] = obs['ft_data']
            nobs['ft_timestamps'] = obs['ft_timestamps']

        nactions = self.normalizer['action'].normalize(batch['action'])
        trajectory = nactions
        
        # process input
        obs_tokens = self.obs_encoder(nobs)
        # (B, N, n_emb)

        # 🆕 Store original global_cond for force prediction (before loop) # TODO: might use later
        # if self.ft_pred_for_residual:
            # obs_tokens_original = obs_tokens
        
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # input perturbation by adding additonal noise to alleviate exposure bias
        # reference: https://github.com/forever208/DDPM-IP
        noise_new = noise + self.input_pertub * torch.randn(trajectory.shape, device=trajectory.device)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (nactions.shape[0],), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps)
        
        # Predict the noise residual
        pred = self.model(
            noisy_trajectory,
            timesteps, 
            cond=obs_tokens
        )

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        return loss

    def forward(self, batch):
        return self.compute_loss(batch)