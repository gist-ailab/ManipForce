import argparse
import os.path as osp

import torch
import torch.nn.functional as F
import numpy as np

url_dict = {
    'small': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth',
    'base': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth',
    'large': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth',
    'giant': 'https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth'
}

def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--variant", default="base", type=str)
    args.add_argument("--kernel", default=16, type=int)
    args.add_argument("--height", default=256, type=int)
    args.add_argument("--width", default=256, type=int)
    args.add_argument("--split-qkv", "--split_qkv", action="store_true", dest="split_qkv", 
                     help="Split QKV weights into separate Q, K, V weights")
    return args.parse_args()


def load_weight(pretrained_path):
    if not osp.isfile(pretrained_path):
        raise FileNotFoundError(
            f"{pretrained_path} dont exist(absolute path: {osp.abspath(pretrained_path)})"
        )
    weight = torch.load(pretrained_path, map_location="cpu")
    if len(weight.keys()) <= 10:
        print(f"The read weights may be abnormal, as shown below:")
        print(weight.keys())
        raise KeyError()
    return weight


def interpolate_patch_embed_(weight, key="patch_embed.proj.weight", kernel_conv=16):
    assert key in weight, f"{key} must in {weight.keys()}"
    ori_shape = weight[key].shape
    weight[key] = F.interpolate(
        weight[key].float(),
        size=(kernel_conv, kernel_conv),
        mode="bicubic",
        align_corners=False,
    )
    dst_shape = weight[key].shape
    print(f"Convert conv kernel in patch embed layer: {ori_shape} -> {dst_shape}")


def interpolate_pos_embed_(
    weight: dict, key="pos_embed", crop_size=(512, 512), kernel_conv=16
):
    pos_cls, pos_tokens = weight[key][:, :1, :], weight["pos_embed"][:, 1:, :]
    embed_dim = pos_tokens.shape[-1]
    orig_size = int(pos_tokens.shape[-2] ** 0.5)
    orig_shape = (-1, orig_size, orig_size, embed_dim)
    crop_size = tuple(L // kernel_conv for L in crop_size)
    resized_pos_tokens = F.interpolate(
        pos_tokens.reshape(*orig_shape).permute(0, 3, 1, 2),
        size=crop_size,
        mode="bicubic",
        align_corners=False,
    )
    dst_shape = resized_pos_tokens.shape
    resized_pos_tokens = resized_pos_tokens.permute(0, 2, 3, 1).reshape(
        -1, np.prod(crop_size), embed_dim
    )
    weight[key] = torch.cat((pos_cls, resized_pos_tokens), dim=1)
    print(
        f"Convert pos embedding: {pos_tokens.shape} -> {orig_shape} -> {dst_shape} -> {resized_pos_tokens.shape}"
    )


def split_qkv_weights(weight):
    qkv_keys = [key for key in weight.keys() if "qkv" in key and "weight" in key]
    for key in qkv_keys:
        qkv_weight = weight[key]
        dim = qkv_weight.shape[1]
        q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=0)
        assert dim == q_weight.shape[0]
        
        q_key = key.replace("qkv", "q")
        k_key = key.replace("qkv", "k")
        v_key = key.replace("qkv", "v")
        
        weight[q_key] = q_weight
        weight[k_key] = k_weight
        weight[v_key] = v_weight
        
        del weight[key]
        print(f"Split {key} into {q_key}, {k_key}, and {v_key}")
    
    qkv_bias_keys = [key for key in weight.keys() if "qkv" in key and "bias" in key]
    for key in qkv_bias_keys:
        qkv_bias = weight[key]
        q_bias, k_bias, v_bias = qkv_bias.chunk(3, dim=0)
        
        q_key = key.replace("qkv", "q")
        k_key = key.replace("qkv", "k")
        v_key = key.replace("qkv", "v")
        
        weight[q_key] = q_bias
        weight[k_key] = k_bias
        weight[v_key] = v_bias
        
        del weight[key]
        print(f"Split {key} into {q_key}, {k_key}, and {v_key}")


def main():
    args = parse_args()
    variant = args.variant
    kernel_conv = args.kernel
    crop_size = (args.height, args.width)
    split_qkv = args.split_qkv

    model_url = url_dict[variant]
    model_name = osp.basename(model_url)

    if crop_size == (512, 512):
        converted_name = model_name.replace('14_', f'{kernel_conv}_')
    else:
        converted_name = model_name.replace('14_', f'{kernel_conv}_{crop_size[0]}x{crop_size[1]}_')
    
    if split_qkv:
        converted_name = converted_name.replace('.pth', '_split_qkv.pth')
    
    model_name = osp.join('./pretrained', model_name)
    converted_name = osp.join('./pretrained', converted_name)

    # Download dinov2 model file if it doesn't exist
    if not osp.exists(model_name):
        import urllib.request
        
        print(f"The file {model_name} does not exist. Starting download...")
        download_url = model_url
        
        try:
            def report_progress(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 / total_size)
                print(f"\rDownload progress: {percent:.2f}% ({downloaded} / {total_size} bytes)", end="")
            
            print(f"Starting download from {download_url}...")
            urllib.request.urlretrieve(download_url, model_name, reporthook=report_progress)
            print(f"\nDownload completed: {model_name}")
        except Exception as e:
            print(f"Download failed: {e}")
            raise

    weight = load_weight(model_name)
    print("Load from", model_name)
    interpolate_patch_embed_(weight, kernel_conv=kernel_conv)
    interpolate_pos_embed_(weight, crop_size=crop_size, kernel_conv=kernel_conv)
    
    if split_qkv:
        split_qkv_weights(weight)
        print("QKV weights have been split into separate Q, K, V weights")
    
    torch.save(weight, converted_name)
    print("Save to", converted_name)
    return args


if __name__ == "__main__":
    main()