"""Noisy Regression Policy: pure noise input, no timestep conditioning, 1-step prediction.

Same transformer architecture as diffusion version, but:
- Input: pure random noise (like diffusion)
- No timestep embedding (T removed) — model doesn't know noise level
- Directly predicts action from noise + obs in one forward pass
- Training: MSE between predicted action and ground truth (no diffusion loss)
"""
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.vision.fmt_obs_encoder import FMTObsEncoder


class TransformerForNoisyRegression(ModuleAttrMixin):
    """Same transformer decoder as diffusion, but timestep embedding replaced with
    a fixed learnable token (no timestep information)."""

    def __init__(self,
        input_dim: int,
        output_dim: int,
        action_horizon: int,
        n_layer: int = 7,
        n_head: int = 8,
        n_emb: int = 768,
        max_cond_tokens: int = 800,
        p_drop_attn: float = 0.1,
    ) -> None:
        super().__init__()

        # input embedding (noisy action → embedding, same as diffusion)
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.randn(1, action_horizon, n_emb))
        # fixed learnable token replacing timestep embedding
        self.fixed_token = nn.Parameter(torch.randn(1, 1, n_emb))
        # condition position embedding (+1 for fixed token replacing time token)
        self.cond_pos_emb = nn.Parameter(torch.randn(1, max_cond_tokens + 1, n_emb))

        # decoder (identical architecture)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_layer
        )

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)

        self.action_horizon = action_horizon

        self.apply(self._init_weights)

    def _init_weights(self, module):
        ignore_types = (nn.Dropout,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential,
            nn.Embedding)
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForNoisyRegression):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.fixed_token, mean=0.0, std=0.02)
            if module.cond_pos_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def get_optim_groups(self, weight_decay: float = 1e-3):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn
                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_emb")
        no_decay.add("fixed_token")
        no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, \
            "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, \
            "parameters %s were not separated into either decay/no_decay set!" % (
                str(param_dict.keys() - union_params),)

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return optim_groups

    def forward(self, noisy_input: torch.Tensor, cond: torch.Tensor):
        """
        noisy_input: (B, T, input_dim) — pure random noise
        cond: (B, N, n_emb) — observation tokens
        output: (B, T, output_dim) — predicted actions
        """
        B = noisy_input.shape[0]

        # condition encoding: obs tokens + fixed token (replacing timestep)
        fixed = self.fixed_token.expand(B, -1, -1)
        cond_emb = torch.cat([cond, fixed], dim=1)
        tc = cond_emb.shape[1]
        cond_pos_emb = self.cond_pos_emb[:, :tc, :]
        cond_emb = cond_emb + cond_pos_emb

        # input: embed noisy action
        input_emb = self.input_emb(noisy_input)
        t = input_emb.shape[1]
        pos_emb = self.pos_emb[:, :t, :]
        input_emb = input_emb + pos_emb

        # transformer decode
        x = self.decoder(tgt=input_emb, memory=cond_emb)
        x = self.ln_f(x)
        x = self.head(x)
        return x


class NoisyRegressionPolicy(BaseImagePolicy):
    """Pure noise input + no timestep → direct action prediction.

    Tests whether noise as input stochasticity is useful WITHOUT
    the diffusion framework (no scheduler, no T conditioning).
    """

    def __init__(self,
            shape_meta: dict,
            obs_encoder: FMTObsEncoder,
            n_layer=7,
            n_head=8,
            n_emb=768,
            p_drop_attn=0.1,
            **kwargs):
        super().__init__()

        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']

        obs_shape = obs_encoder.output_shape()
        assert obs_shape[-1] == n_emb
        obs_tokens = obs_shape[-2]

        model = TransformerForNoisyRegression(
            input_dim=action_dim,
            output_dim=action_dim,
            action_horizon=action_horizon,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            max_cond_tokens=obs_tokens,
            p_drop_attn=p_drop_attn
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.normalizer = LinearNormalizer()
        self.action_dim = action_dim
        self.action_horizon = action_horizon

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert 'past_action' not in obs_dict

        obs_for_normalizer = {k: v for k, v in obs_dict.items()
                              if k not in ['ft_timestamps', 'img_timestamps', 'ft_data']}
        nobs = self.normalizer.normalize(obs_for_normalizer)
        if 'ft_data' in obs_dict:
            nobs['ft_data'] = obs_dict['ft_data']
            nobs['ft_timestamps'] = obs_dict['ft_timestamps']

        B = next(iter(nobs.values())).shape[0]
        obs_tokens = self.obs_encoder(nobs)

        # pure random noise as input (same as diffusion's starting point)
        noise = torch.randn(
            size=(B, self.action_horizon, self.action_dim),
            device=self.device, dtype=self.dtype)

        # 1-step: noise + obs → action (no timestep)
        nsample = self.model(noise, obs_tokens)

        action_pred = self.normalizer['action'].unnormalize(nsample)

        return {
            'action': action_pred,
            'action_pred': action_pred
        }

    # ========= training ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        obs = batch['obs']
        obs_for_normalizer = {
            k: v for k, v in obs.items()
            if k not in ['ft_timestamps', 'img_timestamps', 'ft_data']
        }
        nobs = self.normalizer.normalize(obs_for_normalizer)
        if 'ft_data' in obs:
            nobs['ft_data'] = obs['ft_data']
            nobs['ft_timestamps'] = obs['ft_timestamps']

        nactions = self.normalizer['action'].normalize(batch['action'])
        B = nactions.shape[0]

        obs_tokens = self.obs_encoder(nobs)

        # pure random noise input
        noise = torch.randn_like(nactions)

        # predict action directly from noise
        pred = self.model(noise, obs_tokens)

        loss = F.mse_loss(pred, nactions, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss

    def forward(self, batch):
        return self.compute_loss(batch)
