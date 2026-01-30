import copy

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging
import torchcde
from einops import rearrange

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.vision.utils import AttentionPool1d, CrossAttention, FTEmbed, FTNeuralCDEEncoder
from diffusion_policy.model.vision.backbones.dino_v2_model_zoo import model_dict as dino_model_dict

logger = logging.getLogger(__name__)


class FMTObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            model_name: str,
            pretrained: bool,
            frozen: bool,
            global_pool: str,
            transforms: list,
            use_cls_token: bool=False,
            use_spatial_embed: bool=False,
            use_temporal_embed: bool=False,
            use_modal_embed: bool=False,
            use_attn_pool: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            imagenet_norm: bool=False,
            downsample_ratio: int=32,
            # FT Neural CDE related parameters (same as backup)
            use_ft_ncde: bool=False,
            ft_ncde_hidden_dim: int=256,  # Changed from 128 → 256
            ft_key: str='ft_data',
            ft_timestamp_key: str='ft_timestamps',
            img_timestamp_key: str='img_timestamps',
            # 🆕 Add Cross Attention parameters
            use_cross_attention: bool=False,
            cross_attention_type: str='sequential', # sequential, parallel
            cross_attention_modals: list=['img', 'ft'],
            cross_attention_num_heads: int=4,
            # 🆕 Add only FT prediction related parameters
            use_ft_prediction: bool=False,
            ft_prediction_dim: int=None,
            proj_dim: int=256,
            load_from: str='checkpoints/dinov2-B_psz-16_pretrain.pth'
        ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()
        
        rgb_keys = list()
        ft_keys = list()  # add ft_keys
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()
        feature_dim = None

        assert global_pool == ''

        if 'dino' in model_name:
            model = dino_model_dict[model_name](
                img_size=shape_meta['obs']['handeye_cam_1']['shape'][1],
                pretrained=pretrained,
                load_from=load_from
            )
            feature_dim = model.embed_dim
        else:
            model = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                global_pool=global_pool, # '' means no pooling
                num_classes=0            # remove classification layer
            )
            
        self.frozen = frozen
        if frozen and 'adapter' not in model_name:
            for param in model.parameters():
                param.requires_grad = False
        
        # Multi-modal observation data processing: Classify RGB images, FT sensors, and low-dimensional data by type and assign appropriate models/transforms
        image_shape = None
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]
        if transforms is not None and not isinstance(transforms[0], torch.nn.Module):
            assert transforms[0].type == 'RandomCrop'
            ratio = transforms[0].ratio
            transforms = [
                torchvision.transforms.RandomCrop(size=int(image_shape[0] * ratio)),
                torchvision.transforms.Resize(size=image_shape[0], antialias=True)
            ] + transforms[1:]
        transform = nn.Identity() if transforms is None else torch.nn.Sequential(*transforms)

        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
                model_key = key if not share_rgb_model else key.rsplit('_', 1)[0] if '_' in key else key

                this_model = copy.deepcopy(model)
                key_model_map[model_key] = this_model

                this_transform = transform
                key_transform_map[key] = this_transform
            elif type == 'ft':  # add 'ft' type processing
                ft_keys.append(key)
            elif type == 'low_dim':
                if not attr.get('ignore_by_policy', False):
                    low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        
        # 🆕 Auto-add ee_pose key (process as low_dim)
        if 'ee_pose' not in low_dim_keys:
            low_dim_keys.append('ee_pose')
        
        feature_map_shape = [x // downsample_ratio for x in image_shape]
        
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        logger.info(f'rgb_keys: {rgb_keys}')
        logger.info(f'ft_keys: {ft_keys}')
        logger.info(f'low_dim_keys: {low_dim_keys}')  # 🆕 also output low_dim_keys

        self.model_name = model_name
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.ft_keys = ft_keys  # add ft_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.ft_ncde_hidden_dim = ft_ncde_hidden_dim  # add Neural CDE dimension
        self.use_ft_ncde = use_ft_ncde  # add whether to use Neural CDE
        self.ft_key = ft_key  # add FT key
        self.ft_timestamp_key = ft_timestamp_key  # add timestamp key
        self.img_timestamp_key = img_timestamp_key  # add image timestamp key
        self.use_ft_prediction = use_ft_prediction
        self.frozen = frozen
        self.use_cls_token = use_cls_token
        self.use_spatial_embed = use_spatial_embed
        self.use_temporal_embed = use_temporal_embed
        self.use_modal_embed = use_modal_embed
        self.use_attn_pool = use_attn_pool

        if 'vit' in model_name:
            feature_dim = 768  # standard ViT
            logger.info(f"ViT model detected, assuming feature_dim={feature_dim}")

        elif model_name.startswith('resnet'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 512
            elif downsample_ratio == 16:
                modules = list(model.children())[:-3]
                model = torch.nn.Sequential(*modules)
                feature_dim = 256
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")

        elif model_name.startswith('convnext'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 1024
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")

        self.feature_dim = feature_dim
        
        self.use_cross_attention = use_cross_attention
        self.cross_attention_type = cross_attention_type
        self.cross_attention_modals = cross_attention_modals

        if self.use_cross_attention:
            if 'img' in self.cross_attention_modals:
                self.img_cross_attention = CrossAttention(model_dim=proj_dim, num_heads=cross_attention_num_heads)
            if 'ft' in self.cross_attention_modals:
                self.ft_cross_attention = CrossAttention(model_dim=proj_dim, num_heads=cross_attention_num_heads)

        if self.use_spatial_embed:
            spatial_embed_dict = {}
            for _, rgb_key in enumerate(self.rgb_keys):
                spatial_embed_dict[rgb_key] = nn.Parameter(torch.zeros(1, feature_map_shape[0] * feature_map_shape[1], proj_dim))
            self.spatial_embed = nn.ParameterDict(spatial_embed_dict)
        else:
            self.spatial_embed = None

        if self.use_temporal_embed:
            ft_horizon = shape_meta['obs'][self.ft_key]['horizon']
            rgb_horizon = shape_meta['obs'][self.rgb_keys[0]]['horizon']
            if self.use_ft_ncde:
                self.temporal_embed = nn.Parameter(torch.zeros(1, rgb_horizon, proj_dim))
            else:
                self.temporal_embed = nn.Parameter(torch.zeros(1, ft_horizon, proj_dim))
        else:
            self.temporal_embed = None
        
        if self.use_modal_embed:
            modal_embed_dict = {}
            for _, rgb_key in enumerate(self.rgb_keys):
                modal_embed_dict[rgb_key] = nn.Parameter(torch.zeros(1, 1, proj_dim))
            modal_embed_dict[self.ft_key] = nn.Parameter(torch.zeros(1, 1, proj_dim))
            self.modal_embed = nn.ParameterDict(modal_embed_dict)
        else:
            self.modal_embed = None
        
        # FT encoder
        if self.use_ft_ncde:
            self.ft_embed = FTNeuralCDEEncoder(
                ft_dim=shape_meta['obs'][self.ft_key]['shape'][-1],
                embed_dim=proj_dim,
                initial_dim=proj_dim * len(self.rgb_keys)
            )
            self.init_pool = AttentionPool1d(
                seq_len=feature_map_shape[0] * feature_map_shape[1],
                embed_dim=proj_dim,
                num_heads=proj_dim // 64,
                output_dim=proj_dim
            )
        else:
            self.ft_embed = FTEmbed(
                ft_dim=shape_meta['obs'][self.ft_key]['shape'][-1],
                hidden_channels=proj_dim,
            )

        self.img_proj = nn.Linear(feature_dim, proj_dim)
        self.ft_proj = nn.Linear(proj_dim, proj_dim)

        self.img_norm = nn.LayerNorm(proj_dim)
        self.ft_norm = nn.LayerNorm(proj_dim)

        if self.use_attn_pool:
            self.img_attn_pool = AttentionPool1d(
                seq_len=feature_map_shape[0] * feature_map_shape[1],
                embed_dim=proj_dim,
                num_heads=proj_dim // 64,
                output_dim=proj_dim
            )
            self.ft_attn_pool = AttentionPool1d(
                seq_len=shape_meta['obs'][self.ft_key]['horizon'],
                embed_dim=proj_dim,
                num_heads=proj_dim // 64,
                output_dim=proj_dim
            )
        
        final_dim = proj_dim * (len(self.rgb_keys) * rgb_horizon + len(self.ft_keys)) if self.use_attn_pool else proj_dim
        self.final_proj = nn.Linear(final_dim, final_dim)
        self.final_norm = nn.LayerNorm(final_dim)

        # init embed
        if self.use_spatial_embed:
            for key in self.spatial_embed.keys():
                nn.init.normal_(self.spatial_embed[key], mean=0.0, std=0.02)
        if self.use_temporal_embed:
            nn.init.normal_(self.temporal_embed, mean=0.0, std=0.02)
        if self.use_modal_embed:
            for key in self.modal_embed.keys():
                nn.init.normal_(self.modal_embed[key], mean=0.0, std=0.02)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _interpolate(self, x, size):  # x: (B, L, C)
        if x.shape[1] == size:
            return x
        
        x_ch_first = x.transpose(1, 2)                       # (B, C, L)
        y = F.interpolate(x_ch_first, size=size,
                        mode='linear', align_corners=False) # (B, C, 2L)
        return y.transpose(1, 2)
    
    def extract_img_feat(self, img_1, img_2):
        if self.share_rgb_model:
            model_key = list(self.key_model_map.keys())[0]
            img_1_out = self.key_model_map[model_key](img_1)  # (B*T, L=197, D=768)
            img_2_out = self.key_model_map[model_key](img_2)  # (B*T, L=197, D=768)
        else:
            img_1_out = self.key_model_map[self.rgb_keys[0]](img_1)  # (B*T, L=197, D=768)
            img_2_out = self.key_model_map[self.rgb_keys[1]](img_2)  # (B*T, L=197, D=768)
        return img_1_out, img_2_out
        
    def forward(self, obs_dict: dict, predict_ft: bool = False, verbose: bool = False) -> torch.Tensor:

        # 1) Extract RGB features from all cameras
        # self.rgb_keys = ['handeye_cam_1', 'handeye_cam_2']
        rgb_key_1, rgb_key_2 = self.rgb_keys[0], self.rgb_keys[1]
        img_1, img_2 = obs_dict[rgb_key_1], obs_dict[rgb_key_2]
        B, T_img, _, _, _ = img_1.shape

        img_1 = rearrange(img_1, 'b t c h w -> (b t) c h w')
        img_2 = rearrange(img_2, 'b t c h w -> (b t) c h w')

        img_1 = self.key_transform_map[rgb_key_1](img_1)
        img_2 = self.key_transform_map[rgb_key_2](img_2)
        
        if self.frozen and 'adapter' not in self.model_name:
            with torch.no_grad():
                img_feat_1, img_feat_2 = self.extract_img_feat(img_1, img_2)
        else:
            img_feat_1, img_feat_2 = self.extract_img_feat(img_1, img_2)

        if not self.use_cls_token:
            if verbose:
                logger.info(f"DEBUG: use_cls_token is False")
            img_feat_1 = img_feat_1[:, 1:, :] # (B*T, L=196, D=768)
            img_feat_2 = img_feat_2[:, 1:, :] # (B*T, L=196, D=768)
        else:
            if verbose:
                logger.info(f"DEBUG: use_cls_token is True")
        
        img_feat_seq_1 = rearrange(img_feat_1, '(b t) l d -> b t l d', b=B, t=T_img) # (B, T=2, L=196, D=768) # (B*T, L, D) -> (B, T, L, D)
        img_feat_seq_2 = rearrange(img_feat_2, '(b t) l d -> b t l d', b=B, t=T_img) # (B, T=2, L=196, D=768) # (B*T, L, D) -> (B, T, L, D)

        ft = obs_dict[self.ft_key].to(self.device)
        if self.use_ft_ncde:
            if verbose:
                logger.info(f"DEBUG: use_ft_ncde is True")
            init_feat_1 = self.init_pool(img_feat_seq_1[:, 0, :, :])
            init_feat_2 = self.init_pool(img_feat_seq_2[:, 0, :, :])
            init_feat = torch.cat([init_feat_1, init_feat_2], dim=-1)
            ft_timestamps = obs_dict.get(self.ft_timestamp_key, None)
            if ft_timestamps is None:
                logger.warning(f"DEBUG: ft_timestamps is None")
            ft_feat = self.ft_embed(init_feat, ft, ft_timestamps=ft_timestamps) # (B, T=8, D=768)
        else:
            if verbose:
                logger.info(f"DEBUG: use_ft_ncde is False")
            ft_feat = self.ft_embed(ft) # (B, T=8, D=768)

        L = img_feat_seq_1.shape[2]
        T_ft = ft_feat.shape[1]
        
        img_feat_seq_1 = self.img_proj(img_feat_seq_1)
        img_feat_seq_2 = self.img_proj(img_feat_seq_2)
        ft_feat = self.ft_proj(ft_feat)

        if self.use_spatial_embed:
            if verbose:
                logger.info(f"DEBUG: use_spatial_embed is True")
            img_feat_seq_1 = img_feat_seq_1 + self.spatial_embed[rgb_key_1].unsqueeze(1).expand(-1, T_img, -1, -1)
            img_feat_seq_2 = img_feat_seq_2 + self.spatial_embed[rgb_key_2].unsqueeze(1).expand(-1, T_img, -1, -1)
        else:
            if verbose:
                logger.info(f"DEBUG: use_spatial_embed is False")

        if self.use_temporal_embed:
            if verbose:
                logger.info(f"DEBUG: use_time_embed is True")
            img_feat_seq_1 = img_feat_seq_1 + self._interpolate(self.temporal_embed, size=T_img).unsqueeze(2).expand(-1, -1, L, -1)
            img_feat_seq_2 = img_feat_seq_2 + self._interpolate(self.temporal_embed, size=T_img).unsqueeze(2).expand(-1, -1, L, -1)
            ft_feat = ft_feat + self.temporal_embed
        else:
            if verbose:
                logger.info(f"DEBUG: use_time_embed is False")

        if self.use_modal_embed:
            if verbose:
                logger.info(f"DEBUG: use_modal_embed is True")
            img_feat_seq_1 = img_feat_seq_1 + self.modal_embed[rgb_key_1].unsqueeze(1).expand(-1, T_img, L, -1)
            img_feat_seq_2 = img_feat_seq_2 + self.modal_embed[rgb_key_2].unsqueeze(1).expand(-1, T_img, L, -1)
            ft_feat = ft_feat + self.modal_embed[self.ft_key].expand(-1, T_ft, -1)
        else:
            if verbose:
                logger.info(f"DEBUG: use_modal_embed is False")
        
        img_feat_seq_1 = self.img_norm(img_feat_seq_1)
        img_feat_seq_2 = self.img_norm(img_feat_seq_2)
        ft_feat = self.ft_norm(ft_feat)
        
        img_feat_1 = rearrange(img_feat_seq_1, 'b t l d -> b (t l) d') # (B, T, L, D) -> (B, T*L, D)
        img_feat_2 = rearrange(img_feat_seq_2, 'b t l d -> b (t l) d') # (B, T, L, D) -> (B, T*L, D)

        combined_img_feat = torch.cat([img_feat_1, img_feat_2], dim=1)
        if self.use_cross_attention:
            if verbose:
                logger.info(f"DEBUG: use_cross_attention is True")
            if self.cross_attention_type == 'sequential':
                if verbose:
                    logger.info(f"DEBUG: cross_attention_type is sequential")
                assert len(self.cross_attention_modals) == 2
                if self.cross_attention_modals[0] == 'img':
                    if verbose:
                        logger.info(f"DEBUG: cross_attention_modals is [img, ft]")
                    enhanced_img_feat = self.img_cross_attention(combined_img_feat, ft_feat)
                    enhanced_ft_feat = self.ft_cross_attention(ft_feat, enhanced_img_feat)
                elif self.cross_attention_modals[0] == 'ft':
                    if verbose:
                        logger.info(f"DEBUG: cross_attention_modals is [ft, img]")
                    enhanced_ft_feat = self.ft_cross_attention(ft_feat, combined_img_feat)
                    enhanced_img_feat = self.img_cross_attention(combined_img_feat, enhanced_ft_feat)
                else:
                    raise ValueError(f"Unsupported cross attention modality: {self.cross_attention_modals[0]}")
            elif self.cross_attention_type == 'parallel':
                if verbose:
                    logger.info(f"DEBUG: cross_attention_type is parallel")
                if 'img' in self.cross_attention_modals:
                    if verbose:
                        logger.info(f"DEBUG: cross_attention_modals is img")
                    enhanced_img_feat = self.img_cross_attention(combined_img_feat, ft_feat)
                else:
                    enhanced_img_feat = combined_img_feat
                if 'ft' in self.cross_attention_modals:
                    if verbose:
                        logger.info(f"DEBUG: cross_attention_modals is ft")
                    enhanced_ft_feat = self.ft_cross_attention(ft_feat, combined_img_feat)
                else:
                    enhanced_ft_feat = ft_feat
            else:
                raise ValueError(f"Unsupported cross attention type: {self.cross_attention_type}")
        else:
            if verbose:
                logger.info(f"DEBUG: use_cross_attention is False")
            enhanced_img_feat = combined_img_feat
            enhanced_ft_feat = ft_feat
        
        if self.use_attn_pool:
            if verbose:
                logger.info(f"DEBUG: use_attn_pool is True, use unet-dp")
            enhanced_img_feat_1, enhanced_img_feat_2 = enhanced_img_feat.split([img_feat_1.shape[1], img_feat_2.shape[1]], dim=1)
            enhanced_img_feat_1 = rearrange(enhanced_img_feat_1, 'b (t l) d -> (b t) l d', b=B, t=T_img)
            enhanced_img_feat_2 = rearrange(enhanced_img_feat_2, 'b (t l) d -> (b t) l d', b=B, t=T_img)
            
            pooled_img_feat_1 = self.img_attn_pool(enhanced_img_feat_1) # (b*t, d)
            pooled_img_feat_2 = self.img_attn_pool(enhanced_img_feat_2) # (b*t, d)

            pooled_img_feat_1 = rearrange(pooled_img_feat_1, '(b t) d -> b (t d)', b=B, t=T_img)
            pooled_img_feat_2 = rearrange(pooled_img_feat_2, '(b t) d -> b (t d)', b=B, t=T_img)

            pooled_ft_feat = self.ft_attn_pool(enhanced_ft_feat) # (b, d)

            final_feat = torch.cat([pooled_img_feat_1, pooled_img_feat_2, pooled_ft_feat], dim=-1) # (b, 5*d)
        else:
            if verbose:
                logger.info(f"DEBUG: use_attn_pool is False, use transformer-dp")
            final_feat = torch.cat([enhanced_img_feat, enhanced_ft_feat], dim=1)

        final_feat = self.final_norm(self.final_proj(final_feat))
        if verbose:
            logger.info(f"DEBUG: final_feat.shape: {final_feat.shape}")

        return final_feat
    
    @torch.no_grad()
    def output_shape(self):
        self.to('cuda')
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (1, attr['horizon']) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        
        example_output = self.forward(example_obs_dict, verbose=True)
        # assert len(example_output.shape) == 2
        assert example_output.shape[0] == 1
        
        return example_output.shape


# if __name__=='__main__':
#     timm_obs_encoder = TimmObsEncoder(
#         shape_meta=None,
#         model_name='resnet18.a1_in1k',
#         pretrained=False,
#         global_pool='',
#         transforms=None
#     )
