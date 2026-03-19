from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.vision.fmt_obs_encoder import FMTObsEncoder


class TransformerForActionRegression(ModuleAttrMixin):
    """Same transformer decoder as diffusion version, but without timestep conditioning.
    Uses learnable action query tokens instead of noisy action input."""

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

        # learnable action query tokens (replaces noisy input + timestep)
        self.action_query = nn.Parameter(torch.randn(1, action_horizon, n_emb))
        self.pos_emb = nn.Parameter(torch.randn(1, action_horizon, n_emb))
        # condition position embedding (no +1 for time token since we don't use timestep)
        self.cond_pos_emb = nn.Parameter(torch.randn(1, max_cond_tokens, n_emb))

        # decoder (same architecture)
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

        # init
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
        elif isinstance(module, TransformerForActionRegression):
            torch.nn.init.normal_(module.action_query, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
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

        no_decay.add("action_query")
        no_decay.add("pos_emb")
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

    def forward(self, cond: torch.Tensor):
        """
        cond: (B, N, n_emb) observation tokens
        output: (B, T, output_dim) predicted actions
        """
        B = cond.shape[0]

        # condition encoding
        tc = cond.shape[1]
        cond_pos_emb = self.cond_pos_emb[:, :tc, :]
        cond_emb = cond + cond_pos_emb

        # action queries
        query = self.action_query.expand(B, -1, -1) + self.pos_emb

        # transformer decode
        x = self.decoder(tgt=query, memory=cond_emb)
        x = self.ln_f(x)
        x = self.head(x)
        return x


class RegressionTransformerTimmPolicy(BaseImagePolicy):
    """Direct action regression using the same transformer architecture.
    No diffusion, no noise scheduler - just obs_encoder → transformer → action."""

    def __init__(self,
            shape_meta: dict,
            obs_encoder: FMTObsEncoder,
            # arch
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

        model = TransformerForActionRegression(
            input_dim=action_dim,
            output_dim=action_dim,
            action_horizon=action_horizon,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            max_cond_tokens=obs_tokens,  # no +1 since no time token
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

        # encode observations
        obs_tokens = self.obs_encoder(nobs)

        # directly predict actions (no diffusion)
        nsample = self.model(obs_tokens)

        # unnormalize
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

        obs_tokens = self.obs_encoder(nobs)
        pred = self.model(obs_tokens)

        loss = F.mse_loss(pred, nactions, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss

    def forward(self, batch):
        return self.compute_loss(batch)
