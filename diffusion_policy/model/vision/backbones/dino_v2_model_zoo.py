from .dino_v2 import create_dino_vits16, create_dino_vitb16, create_dino_vitl16, create_dino_vitg16

model_dict = {
    'dino_vits16': create_dino_vits16,
    'dino_vitb16': create_dino_vitb16,
    'dino_vitl16': create_dino_vitl16,
    'dino_vitg16': create_dino_vitg16,
}