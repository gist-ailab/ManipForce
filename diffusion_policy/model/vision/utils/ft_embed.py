import torch
import torch.nn as nn

import logging
import torchcde

logger = logging.getLogger(__name__)


class FTEmbed(nn.Module):
    def __init__(
        self,
        ft_dim,
        hidden_channels,
        norm_type="group",
        act="gelu",
        alpha_init=1e-2,
        groups=None
    ):
        super().__init__()
        # Organize and clarify variables
        self.ft_dim = ft_dim  # input FT dimension
        self.hidden_channels = hidden_channels  # number of channels after embedding
        self.norm_type = norm_type  # normalization type: batch/group/layer
        self.groups = groups  # number of groups for group norm
        self.alpha_init = alpha_init  # initial value for alpha

        # 1D convolution: FT dimension -> hidden channels
        self.conv = nn.Conv1d(
            in_channels=self.ft_dim,
            out_channels=self.hidden_channels,
            kernel_size=3,
            padding=1
        )

        # Select normalization layer
        if self.norm_type == "batch":
            self.norm_conv = nn.BatchNorm1d(self.hidden_channels)
        elif self.norm_type == "group":
            if self.groups is None:
                self.groups = self.hidden_channels // 64
            self.norm_conv = nn.GroupNorm(num_groups=self.groups, num_channels=self.hidden_channels)
        else:  # "layer"
            self.norm_conv = nn.LayerNorm(self.hidden_channels)

        # Select activation function
        if act == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU(inplace=True)

        # Residual projection (only if dimensions differ)
        if self.ft_dim != self.hidden_channels:
            self.res_proj = nn.Linear(self.ft_dim, self.hidden_channels)
        else:
            self.res_proj = nn.Identity()

        # Learnable scaling parameter
        self.alpha = nn.Parameter(torch.tensor(self.alpha_init, dtype=torch.float32))

    def forward(self, x):  # x: [B, T, C]
        B, T, C = x.shape

        # Conv: [B, C, T] -> [B, H, T]
        z = x.transpose(1, 2)
        z = self.conv(z)

        # Post-norm after Conv
        if self.norm_type == "batch" or self.norm_type == "group":
            z = self.norm_conv(z)                   # BN1d expects [B, C, T]
            z = self.act(z)
            z = z.transpose(1, 2)                  # [B, T, H]
        else:
            z = z.transpose(1, 2)                  # [B, T, H] for LN/GN
            z = self.norm_conv(z)                  # LN/GN over channel dim
            z = self.act(z)

        out = self.res_proj(x) + self.alpha.to(z.dtype) * z
        return out


class VectorField(nn.Module):
    def __init__(self, hidden_channels: int, ft_dim: int, act="GELU"):
        """
        Args:
          hidden_channels: CDE hidden dimension (e.g., 512)
          ft_dim: FT sensor data dimension (e.g., 6)
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.ft_dim = ft_dim
        
        # ───────────────────────────────────────────────────────────────
        # (1) Small MLP that transforms t into hidden_channels dimension
        # ───────────────────────────────────────────────────────────────
        # Input: t (scalar or (batch,) form)
        # -> (batch, 1) -> Linear(1 -> hidden_channels) -> LayerNorm -> ReLU → (batch, hidden_channels)
        self.time_proj = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.LayerNorm(hidden_channels),
            getattr(nn, act)()
        )
        
        # ───────────────────────────────────────────────────────────────
        # (2) Linear layer for the original z(t) -> f(z(t)) calculation
        #       Input: (batch, hidden_channels)
        #       Output: (batch, hidden_channels * (ft_dim + 1))
        # ───────────────────────────────────────────────────────────────
        self.linear = nn.Linear(hidden_channels, hidden_channels * (ft_dim + 1))
        # Weight initialization: Xavier (gain=0.1)
        nn.init.xavier_uniform_(self.linear.weight, gain=0.1)
        nn.init.zeros_(self.linear.bias)
        
        # (3) LayerNorm + ReLU after linear output
        self.norm = nn.LayerNorm(hidden_channels * (ft_dim + 1))
        self.act = getattr(nn, act)()

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
          t: scalar or (batch,) form of timestamp
          z: [batch, hidden_channels] or [hidden_channels] form of hidden state
        Returns:
          [batch, hidden_channels, ft_dim+1] form of tensor
        """
        
        batch_size = z.size(0)
        
        # 2) Reshape t to (batch, 1)
        #    - If t is a scalar (0-dim), reshape to (1,) then expand by batch size
        if t.dim() == 0:
            # example: t = torch.tensor(0.5)
            t = t.unsqueeze(0).expand(batch_size)  # → (batch,)
        #    - Now t is in (batch,) form. convert to (batch, 1) and send to MLP
        if t.dim() == 1:
            t_in = t.unsqueeze(-1)  # (batch, 1)
        else:
            # If already (batch, 1), leave as is
            t_in = t
        
        # 3) t embedding: (batch, 1) → (batch, hidden_channels)
        t_emb = self.time_proj(t_in)  # (batch, hidden_channels)
        
        # 4) Add time information to z
        #    (batch, hidden_channels) + (batch, hidden_channels) → (batch, hidden_channels)
        z_time = z + t_emb
        
        # 5) Final linear layer + activation
        #    (batch, hidden_channels) → Linear → (batch, hidden_channels*(ft_dim+1))
        out = self.linear(z_time)
        out = self.norm(out)
        out = self.act(out)
        
        # 6) [batch, hidden_channels*(ft_dim+1)] → [batch, hidden_channels, ft_dim+1]
        return out.view(batch_size, self.hidden_channels, self.ft_dim + 1)


class FTNeuralCDEEncoder(nn.Module):
    def __init__(
        self,
        ft_dim,
        embed_dim,
        initial_dim=None,
        act="GELU",
    ):
        super().__init__()
        self.ft_dim = ft_dim
        self.embed_dim = embed_dim

        assert initial_dim is not None, "initial_dim must be provided"
        
        # initial state encoder (img_feat -> hidden_channels)
        if embed_dim != initial_dim:
            self.initial_encoder = nn.Sequential(
                nn.Linear(initial_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                getattr(nn, act)()
            )
            logger.info(f"Created single-stage dimension reducer from {initial_dim} to {embed_dim}")
        else:
            logger.warning("No initial encoder is not required for FTNeuralCDEEncoder")
            self.initial_encoder = None
            
        self.vector_field = VectorField(embed_dim, ft_dim, act=act)
    
    def forward(self, img_feat, ft_data, ft_timestamps=None):
        """
        Args:
            img_feat: (batch, img_feat_dim) form of image features
            ft_data: (batch, time, ft_dim) form of FT data
            ft_timestamps: (batch, time) form of FT timestamps, assume equal interval if None
        
        Returns:
            z_all: (batch, time, hidden_channels) form of final hidden state
        """

        batch_size, seq_len, _ = ft_data.shape
        # Default to equal interval if timestamps are missing
        if ft_timestamps is None:
            ft_timestamps = torch.linspace(0, 1, seq_len, device=ft_data.device)
            ft_timestamps = ft_timestamps.expand(batch_size, seq_len)
        
        # Normalize timestamps (to [0, 1] range)
        if ft_timestamps.max() > 1.0:
            # Compute min and max per batch at once
            t_min = ft_timestamps.min(dim=-1, keepdim=True)[0]  # (B, 1)
            t_max = ft_timestamps.max(dim=-1, keepdim=True)[0]  # (B, 1)
            ft_timestamps_normalized = (ft_timestamps - t_min) / (t_max - t_min + 1e-10)
        else:
            ft_timestamps_normalized = ft_timestamps
        
        ft_feat_with_time = torch.cat([
            ft_timestamps_normalized.unsqueeze(-1),
            ft_data
        ], dim=-1)
        
        # Compute Hermite spline coefficients
        coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(ft_feat_with_time)
        X = torchcde.CubicSpline(coeffs)
        
        if self.initial_encoder is not None:
            z0 = self.initial_encoder(img_feat)  # (batch, hidden_channels)
        else:
            z0 = img_feat

        z_all = torchcde.cdeint(
            X=X,
            func=self.vector_field,
            z0=z0,
            t=X.interval,
            # t=t_tensor,
            method='rk4',
            atol=1e-3,
            rtol=1e-3,
        )
        
        z_final = z_all

        # NaN/Inf check
        if torch.isnan(z_final).any():
            logger.warning("[FT-CDE] z_final contains NaN!")
        if torch.isinf(z_final).any():
            logger.warning("[FT-CDE] z_final contains Inf!")
            
        return z_final


# for debug
# if __name__ == "__main__":
#     model = FTNeuralCDEEncoder(ft_dim=6, embed_dim=768, initial_dim=768)
#     img_feat = torch.randn(1, 768)
#     ft_feat = torch.randn(1, 8, 6)
#     z_final = model(img_feat, ft_feat)
#     print(z_final.shape)