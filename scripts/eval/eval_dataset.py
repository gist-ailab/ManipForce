
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import time
import cv2
import numpy as np
import torch
import argparse

from omegaconf import OmegaConf
import hydra.utils

import dill
import os
import json
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt, lfilter
from scipy.spatial.transform import Slerp

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from umi.real_world.real_inference_util import (get_real_gumi_obs_dict,
                                              get_real_gumi_action)
from diffusion_policy.dataset.gumi_dataset_w_ft import GumiDatasetWithFT
from torch.utils.data import DataLoader

# 전역 변수 선언
prev_frames = []
prev_timestamps = []
current_img_idx = 0

# 데이터 수집 주기가 30Hz인 경우
dt = 1.0 / 30.0  # 30Hz

def convert_action_10d_to_8d(action_10d):
    position = action_10d[..., :3]
    rotation_6d = action_10d[..., 3:9]
    gripper = action_10d[..., 9:10]

    first_col = rotation_6d[..., :3]
    second_col = rotation_6d[..., 3:6]

    first_col = first_col /  np.linalg.norm(first_col, axis=-1, keepdims=True)
    second_col = second_col - np.sum(first_col * second_col, axis=-1, keepdims=True) * first_col
    second_col = second_col / np.linalg.norm(second_col, axis=-1, keepdims=True)
    third_col = np.cross(first_col, second_col)

    rotation_matrix = np.stack([first_col, second_col, third_col], axis=-1)
    quat = R.from_matrix(rotation_matrix).as_quat()  # [x, y, z, w] 순서
    # 7D 벡터로 조합
    action_8d = np.concatenate([
        position,    # 위치 (3)
        quat,       # 쿼터니언 (4)
        gripper     # 그리퍼 (1)
    ], axis=-1)
    
    return action_8d

def butter_lowpass(cutoff: float, fs: float, order: int = 5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    return butter(order, normal_cutoff, btype='low', analog=False)


def lowpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 5) -> np.ndarray:
    b, a = butter_lowpass(cutoff, fs, order)
    return lfilter(b, a, data, axis=0)

def moving_average_filter(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """이동 평균 필터 - 오프셋 문제 없음"""
    if len(data) < window_size:
        return data
    
    # 패딩을 위해 양 끝에 반사
    pad_size = window_size // 2
    padded_data = np.pad(data, (pad_size, pad_size), mode='edge')
    
    # 이동 평균 계산
    smoothed = np.zeros_like(data)
    for i in range(len(data)):
        smoothed[i] = np.mean(padded_data[i:i+window_size])
    
    return smoothed

def savgol_filter(data: np.ndarray, window_size: int = 5, poly_order: int = 2) -> np.ndarray:
    """Savitzky-Golay 필터 - 스케일 보존하면서 스무딩"""
    if len(data) < window_size:
        return data
    
    # window_size가 짝수면 홀수로 만들기
    if window_size % 2 == 0:
        window_size += 1
    
    # poly_order는 window_size보다 작아야 함
    poly_order = min(poly_order, window_size - 1)
    
    try:
        from scipy.signal import savgol_filter as scipy_savgol
        return scipy_savgol(data, window_size, poly_order)
    except ImportError:
        # scipy가 없으면 이동 평균으로 대체
        print("Warning: scipy not available, using moving average instead")
        return moving_average_filter(data, window_size)

def weighted_moving_average_filter(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """가중 이동 평균 필터 - 중앙값에 더 높은 가중치"""
    if len(data) < window_size:
        return data
    
    # 가중치 생성 (중앙에 높은 가중치)
    weights = np.exp(-0.5 * np.linspace(-2, 2, window_size)**2)
    weights = weights / np.sum(weights)  # 정규화
    
    # 패딩을 위해 양 끝에 반사
    pad_size = window_size // 2
    padded_data = np.pad(data, (pad_size, pad_size), mode='edge')
    
    # 가중 이동 평균 계산
    smoothed = np.zeros_like(data)
    for i in range(len(data)):
        window_data = padded_data[i:i+window_size]
        smoothed[i] = np.sum(window_data * weights)
    
    return smoothed


def slerp_quaternions(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    key_rots = R.from_quat([q1, q2])
    s = Slerp([0, 1], key_rots)
    return s(t).as_quat()

def get_obs(data_iter, episode_start_idx=None, episode_end_idx=None, current_global_idx=0):
    try:
        batch = next(data_iter)
    except StopIteration:
        return None, None, None
    
    # 특정 에피소드 범위를 벗어나면 None 반환
    if episode_start_idx is not None and episode_end_idx is not None:
        if current_global_idx < episode_start_idx or current_global_idx >= episode_end_idx:
            return None, None, None

    # ---- pull out two cameras ----
    cam1_tchw = batch['obs']['handeye_cam_1'].squeeze(0).cpu().numpy()  # (T,C,H,W)
    cam2_tchw = batch['obs']['handeye_cam_2'].squeeze(0).cpu().numpy()  # (T,C,H,W)

    # BGR → RGB 변환 (학습 데이터와 일치시키기 위해) - TCHW 형태에서 변환
    cam1_thwc = cam1_tchw.transpose(0, 2, 3, 1)  # (T,H,W,C)로 임시 변환
    cam2_thwc = cam2_tchw.transpose(0, 2, 3, 1)  # (T,H,W,C)로 임시 변환
    
    # BGR → RGB 변환
    cam1_rgb = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in cam1_thwc])
    cam2_rgb = np.stack([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in cam2_thwc])
    
    # 다시 TCHW 형태로 변환
    cam1_tchw_rgb = cam1_rgb.transpose(0, 3, 1, 2)  # (T,C,H,W)
    cam2_tchw_rgb = cam2_rgb.transpose(0, 3, 1, 2)  # (T,C,H,W)

    # ---- pull out FT window ----
    ft_data = batch['obs']['ft_data'].squeeze(0).cpu().numpy()          # (T_ft, D_ft)
    
    # FT 타임스탬프도 포함
    ft_timestamps = None
    if 'ft_timestamps' in batch['obs']:
        ft_timestamps = batch['obs']['ft_timestamps'].squeeze(0).cpu().numpy()
    
    obs_np = {
        'handeye_cam_1': cam1_tchw_rgb,  # (T,C,H,W) 형태 유지
        'handeye_cam_2': cam2_tchw_rgb,  # (T,C,H,W) 형태 유지
        'ft_data':       ft_data,
    }
    
    # FT 타임스탬프가 있으면 추가
    if ft_timestamps is not None:
        obs_np['ft_timestamps'] = ft_timestamps

    # ground-truth action sequence
    gt = None
    if 'action' in batch:
        gt = batch['action'].squeeze(0).cpu().numpy()  # (T_a, D_a)

    # for display, pick last cam1 frame (RGB로 변환된 것 사용)
    frame = cam1_rgb[-1]  # (H,W,C) 형태
    return obs_np, frame, gt

def setup_data(
    cfg: OmegaConf,
    dataset_path: str,
    episode_idx: int = 0
):
    """
    Returns:
      data_iter: 지정한 episode 부터 읽는 iterator
      episode_len: 해당 episode의 프레임 수
    """
    # 1) 전체 dataset & replay_buffer 가져오기
    dataset = GumiDatasetWithFT(
        shape_meta=cfg.shape_meta,
        dataset_path=dataset_path,
        pose_repr={
            'obs_pose_repr':    cfg.task.pose_repr.obs_pose_repr,
            'action_pose_repr': cfg.task.pose_repr.action_pose_repr
        }
    )
    # 각 에피소드 길이 리스트
    ep_lengths = dataset.replay_buffer.episode_lengths  
    print(f"Total episodes in dataset: {len(ep_lengths)}")
    print(f"Episode lengths: {ep_lengths[:10]}...")  # 처음 10개만 출력
    assert 0 <= episode_idx < len(ep_lengths), \
        f"episode_idx out of range: must be 0 <= idx < {len(ep_lengths)}"

    # 목표 episode 길이 & 시작 인덱스
    episode_len = ep_lengths[episode_idx]
    start_idx = sum(ep_lengths[:episode_idx])
    end_idx = start_idx + episode_len
    print(f"Target episode {episode_idx}: length={episode_len}, start_idx={start_idx}, end_idx={end_idx}")

    # 2) 전체 데이터셋을 사용하되, 특정 에피소드만 처리하도록 수정
    # DataLoader iterator 생성 (전체 데이터셋 사용)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
    data_iter = iter(loader)
    
    # 에피소드 시작 인덱스 저장 (나중에 필터링에 사용)
    episode_start_idx = start_idx
    episode_end_idx = end_idx

    return data_iter, episode_len, episode_start_idx, episode_end_idx


def main():
    # Command line arguments 파싱
    parser = argparse.ArgumentParser(description='GUMI dataset multimodal evaluation')
    parser.add_argument('--dataset_path', type=str, 
                       default="/home/ailab-2204/Workspace/gail-umi/data/LanPort_Insertion_0810.zarr",
                       help='Path to the dataset (.zarr file)')
    parser.add_argument('--model_checkpoint_path', type=str,
                       default="/home/ailab-2204/Workspace/gail-umi/data/outputs/adapter_dinov2-B_unet-dp_lanport/checkpoints/epoch=0109-val_action_error=0.00040.ckpt",
                       help='Path to the model checkpoint file')
    parser.add_argument('--episode_idx', type=int, default=10,
                       help='Episode index to evaluate')
    parser.add_argument('--smoothing', action='store_true', default=True,
                       help='Enable smoothing for predictions')
    parser.add_argument('--filter_type', type=str, default='savgol', 
                       choices=['moving_avg', 'weighted_avg', 'savgol', 'lowpass'],
                       help='Type of smoothing filter')
    parser.add_argument('--window_size', type=int, default=5,
                       help='Window size for smoothing filter')
    parser.add_argument('--cutoff_freq', type=float, default=5.0,
                       help='Cutoff frequency for low-pass filter (Hz)')
    parser.add_argument('--sampling_freq', type=float, default=30.0,
                       help='Sampling frequency (Hz)')
    parser.add_argument('--pred_scale', type=float, default=1.0,
                       help='Scale factor for predictions (multiply predictions by this value)')
    
    args = parser.parse_args()

    # load checkpoint
    ckpt_path = args.model_checkpoint_path
    payload = torch.load(open(ckpt_path, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cfg.policy.obs_encoder.pretrained = False # pretrained 설정 추가
    cls = hydra.utils.get_class(cfg._target_)
    
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    policy.num_inference_steps = 16 # DDIM inference iterations
    obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
    action_pose_repr = cfg.task.pose_repr.action_pose_repr

    device = torch.device('cuda')
    policy.eval().to(device)
    
    dataset_path = args.dataset_path
    print(f"Loading episode {args.episode_idx} from dataset: {dataset_path}")
    data_iter, episode_len, episode_start_idx, episode_end_idx = setup_data(cfg, dataset_path, episode_idx=args.episode_idx)
    print(f"Episode {args.episode_idx} loaded with {episode_len} frames")
    
    # 스무딩 파라미터
    smoothing_enabled = args.smoothing
    filter_type = args.filter_type  # 필터 타입
    window_size = args.window_size  # 윈도우 크기
    cutoff_freq = args.cutoff_freq  # Hz (낮을수록 더 부드러워짐)
    sampling_freq = args.sampling_freq  # Hz (데이터 수집 주파수)
    pred_scale = args.pred_scale  # 예측값 스케일 팩터
    
    if pred_scale != 1.0:
        print(f"Prediction scale factor: {pred_scale}")
    

    pred_pos_list, gt_pos_list = [], []
    pred_quat_list, gt_quat_list = [], []
    pred_euler_list, gt_euler_list = [], []
    pred_gripper_list, gt_gripper_list = [], []  # gripper 리스트 추가
    cur_pred_pos = np.zeros(3)
    cur_gt_pos   = np.zeros(3)


    current_global_idx = 0
    for _ in range(episode_len):
        obs_np, frame, gt_act = get_obs(data_iter, episode_start_idx, episode_end_idx, current_global_idx)
        current_global_idx += 1
        # (Pdb) obs_np['handeye_cam_1'].shape (2, 224, 224, 3)
        # (Pdb) obs_np['ft_data'].shape(4, 6)
        if obs_np is None:
            print("All data processed.")
            break

        # print(">>> raw ft_data:", obs_np['ft_data'].shape)
        # print(obs_np['ft_data'])  # (T_ft, D_ft) 형태의 넘파이 배열
        # 전처리 & 예측
        obs_dict = get_real_gumi_obs_dict(
            env_obs=obs_np,
            shape_meta=cfg.task.shape_meta,
            obs_pose_repr=cfg.task.pose_repr.obs_pose_repr
        )
        obs_t = dict_apply(
            obs_dict,
            lambda x: torch.from_numpy(x).unsqueeze(0).to(device)
        )
        with torch.no_grad():
            res = policy.predict_action(obs_t)
        act10 = res['action_pred'][0].cpu().numpy()
        
        if gt_act is not None:
            # 예측과 GT의 같은 시점 비교 (첫 번째 스텝)
            pred_act8 = convert_action_10d_to_8d(act10[0])  # 예측의 첫 번째 스텝
            gt_act8 = convert_action_10d_to_8d(gt_act[0])   # GT의 첫 번째 스텝
            
            # Predicted pos, quat & gripper (스케일 적용)
            p = pred_act8[:3] * pred_scale  # 위치에 스케일 적용
            q = pred_act8[3:7]  # 회전은 스케일 적용하지 않음 (정규화된 쿼터니언)
            gripper_pred = pred_act8[7]  # 그리퍼에 스케일 적용

            # GT pos, quat & gripper
            g_p = gt_act8[:3]
            g_q = gt_act8[3:7]
            gripper_gt = gt_act8[7]

            # 상대적 액션을 직접 비교 (누적하지 않음)
            # cur_pred_pos += p
            # cur_gt_pos   += g_p

            # 쿼터니언 보간
            if pred_quat_list:
                prev_q = pred_quat_list[-1]
                sm_q = slerp_quaternions(prev_q, q, 0.8)
            else:
                sm_q = q
            sm_q = sm_q / np.linalg.norm(sm_q)

            # 리스트에 추가 (상대적 액션 직접 저장)
            pred_pos_list.append(p.copy())
            gt_pos_list.append(g_p.copy())
            pred_quat_list.append(sm_q.copy())
            gt_quat_list.append(g_q.copy())
            pred_euler_list.append(R.from_quat(sm_q).as_euler('xyz', True))
            gt_euler_list.append(R.from_quat(g_q).as_euler('xyz', True))
            pred_gripper_list.append(gripper_pred)
            gt_gripper_list.append(gripper_gt)

        # 화면에 표시
        cv2.imshow('Observation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # Plot & 저장
    if pred_pos_list:
        pp = np.array(pred_pos_list)
        gp = np.array(gt_pos_list)
        po = np.array(pred_euler_list)
        go = np.array(gt_euler_list)
        pg = np.array(pred_gripper_list)
        gg = np.array(gt_gripper_list)
        
        # 예측값 스무딩 적용
        if smoothing_enabled and len(pp) > window_size:  # 최소 window_size 이상의 데이터가 있을 때만 적용
            print(f"Applying {filter_type} smoothing with window_size={window_size}")
            
            # 필터 함수 선택
            if filter_type == 'moving_avg':
                filter_func = lambda x: moving_average_filter(x, window_size)
            elif filter_type == 'weighted_avg':
                filter_func = lambda x: weighted_moving_average_filter(x, window_size)
            elif filter_type == 'savgol':
                filter_func = lambda x: savgol_filter(x, window_size, poly_order=2)
            elif filter_type == 'lowpass':
                filter_func = lambda x: lowpass_filter(x, cutoff_freq, sampling_freq, order=3)
            else:
                filter_func = lambda x: moving_average_filter(x, window_size)
            
            # 위치 스무딩
            pp_smooth = np.zeros_like(pp)
            for i in range(3):  # X, Y, Z 각각에 대해
                pp_smooth[:, i] = filter_func(pp[:, i])
            
            # 회전 스무딩
            po_smooth = np.zeros_like(po)
            for i in range(3):  # roll, pitch, yaw 각각에 대해
                po_smooth[:, i] = filter_func(po[:, i])
            
            # 그리퍼 스무딩
            pg_smooth = filter_func(pg)
            
            # 스무딩된 결과를 원본에 할당
            pp = pp_smooth
            po = po_smooth
            pg = pg_smooth
            
            print(f"{filter_type} smoothing applied successfully!")

        plt.figure(figsize=(12, 12))
        # Position (Relative Action)
        plt.subplot(3, 1, 1)
        for i, lbl in enumerate(['X','Y','Z']):
            plt.plot(pp[:,i], label=f'Pred {lbl}')
            plt.plot(gp[:,i], '--', label=f'GT {lbl}')
        title_suffix = ""
        if smoothing_enabled:
            title_suffix += f" ({filter_type.capitalize()} Smoothed)"
        if pred_scale != 1.0:
            title_suffix += f" (Scale={pred_scale:.2f})"
        plt.title(f'Relative Position Action{title_suffix}')
        plt.legend(); plt.grid(True)
        
        # Orientation
        plt.subplot(3, 1, 2)
        for i, lbl in enumerate(['roll','pitch','yaw']):
            plt.plot(po[:,i], label=f'Pred {lbl}')
            plt.plot(go[:,i], '--', label=f'GT {lbl}')
        plt.title(f'Orientation (degrees){title_suffix}')
        plt.legend(); plt.grid(True)
        
        # Gripper
        plt.subplot(3, 1, 3)
        plt.plot(pg, label='Pred Gripper', linewidth=2)
        plt.plot(gg, '--', label='GT Gripper', linewidth=2)
        plt.title(f'Gripper State{title_suffix}')
        plt.ylabel('Gripper Value')
        plt.xlabel('Time Steps')
        plt.legend(); plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('pose_comparison.png')
        plt.close()
        print("Saved 'pose_comparison.png' with position, orientation, and gripper plots.")

if __name__ == '__main__':
    main()