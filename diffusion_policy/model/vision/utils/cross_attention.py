import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(
        self,
        model_dim=256,
        num_heads=4,
        attn_drop=0.0,
        drop=0.0,
        use_mlp=True,
        mlp_ratio=4,
    ):
        super().__init__()

        self.drop = nn.Dropout(drop)
        self.use_mlp = use_mlp

        self.attn = nn.MultiheadAttention(model_dim, num_heads, dropout=attn_drop, batch_first=True)
        self.norm1 = nn.LayerNorm(model_dim)

        if use_mlp:
            h = model_dim * mlp_ratio
            self.ffn = nn.Sequential(nn.Linear(model_dim, h), nn.GELU(), nn.Linear(h, model_dim))
            self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, img_feat, ft_feat, img_key_padding_mask=None, ft_key_padding_mask=None):
        """
        img_feat: (B, L_img, D), ft_feat: (B, L_ft, D)
        *_key_padding_mask: (B, L) bool, True = pad
        """
        # Change computation order according to 'order'
        # 1) img <- ft  (image: Query, FT: Key/Value)
        attn_img, _ = self.attn(
            query=img_feat,
            key=ft_feat,
            value=ft_feat,
            key_padding_mask=ft_key_padding_mask,
            need_weights=True,
        )
        img_out = self.norm1(img_feat + self.drop(attn_img))        # residual connection

        if self.use_mlp:
            img_out = self.norm2(img_out + self.drop(self.ffn(img_out)))  # FFN residual connection

        return img_out