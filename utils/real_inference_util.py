from typing import Dict, Callable, Tuple, List
import numpy as np
from scipy.spatial.transform import Rotation as R  # 추가

import collections
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.common.pose_repr_util import (
    compute_relative_pose, 
    convert_pose_mat_rep
)
from utils.pose_util import (
    pose_to_mat, mat_to_pose,
    mat_to_pose10d, pose10d_to_mat)
from diffusion_policy.model.common.rotation_transformer import \
    RotationTransformer

def get_real_obs_resolution(
        shape_meta: dict
        ) -> Tuple[int, int]:
    out_res = None
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            co,ho,wo = shape
            if out_res is None:
                out_res = (wo, ho)
            assert out_res == (wo, ho)
    return out_res


def get_real_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            this_imgs_in = env_obs[key]
            t,hi,wi,ci = this_imgs_in.shape
            co,ho,wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi,hi), 
                    output_res=(wo,ho), 
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
        elif type == 'low_dim':
            this_data_in = env_obs[key]
            obs_dict_np[key] = this_data_in
    return obs_dict_np


def get_real_gumi_obs_dict(
    env_obs: Dict[str, np.ndarray], 
    shape_meta: dict,
    obs_pose_repr: str='abs'
) -> Dict[str, np.ndarray]:
    obs_dict_np: Dict[str, np.ndarray] = {}
    images = env_obs
    obs_shape_meta = shape_meta['obs']

    # 1) RGB streams만 처리
    for key, attr in obs_shape_meta.items():
        if attr.get('type') != 'rgb':
            continue

        expected_C, expected_H, expected_W = attr['shape']
        im = images[key]  # (T,C,H,W) 형태로 들어옴

        # shape 체크만 하고 변환하지 않음
        if im.ndim == 4:
            T, C, H, W = im.shape
            assert C == expected_C, f"{key}: channel mismatch {C}!={expected_C}"
            obs_dict_np[key] = im  # 그대로 TCHW 형태 유지

    # 2) FT 및 기타 저차원 스트림 추가
    for low_key in ['ft_data', 'ft_timestamps', 'state', 'img_timestamps']:
        if low_key in images:
            arr = images[low_key]
            obs_dict_np[low_key] = arr.astype(np.float32)

    # 3) pose_wrt_start를 gumi_dataset_w_ft.py와 같은 방식으로 정규화를 해줘야함
    if 'pose_wrt_start' in obs_shape_meta and 'pose_wrt_start' in env_obs:
        pose_data = env_obs['pose_wrt_start']  # (T, 7)
        # 정규화 없이 그대로 전달 (policy.predict_action에서 정규화 수행)
        obs_dict_np['pose_wrt_start'] = pose_data.astype(np.float32)

    return obs_dict_np

def get_real_umi_obs_dict(
        env_obs: Dict[str, np.ndarray], 
        shape_meta: dict,
        obs_pose_repr: str='abs',
        tx_robot1_robot0: np.ndarray=None,
        episode_start_pose: List[np.ndarray]=None,
        ) -> Dict[str, np.ndarray]:
    obs_dict_np = dict()
    # process non-pose
    obs_shape_meta = shape_meta['obs']
    robot_prefix_map = collections.defaultdict(list)
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = attr.get('shape')
        if type == 'rgb':
            this_imgs_in = env_obs[key]
            t,hi,wi,ci = this_imgs_in.shape
            co,ho,wo = shape
            assert ci == co
            out_imgs = this_imgs_in
            if (ho != hi) or (wo != wi) or (this_imgs_in.dtype == np.uint8):
                tf = get_image_transform(
                    input_res=(wi,hi), 
                    output_res=(wo,ho), 
                    bgr_to_rgb=False)
                out_imgs = np.stack([tf(x) for x in this_imgs_in])
                if this_imgs_in.dtype == np.uint8:
                    out_imgs = out_imgs.astype(np.float32) / 255
            # THWC to TCHW
            obs_dict_np[key] = np.moveaxis(out_imgs,-1,1)
        elif type == 'low_dim' and ('eef' not in key):
            this_data_in = env_obs[key]
            obs_dict_np[key] = this_data_in
            # handle multi-robots
            ks = key.split('_')
            if ks[0].startswith('robot'):
                robot_prefix_map[ks[0]].append(key)

    # generate relative pose
    for robot_prefix in robot_prefix_map.keys():
        # convert pose to mat
        pose_mat = pose_to_mat(np.concatenate([
            env_obs[robot_prefix + '_eef_pos'],
            env_obs[robot_prefix + '_eef_rot_axis_angle']
        ], axis=-1))

        # solve reltaive obs
        obs_pose_mat = convert_pose_mat_rep(
            pose_mat, 
            base_pose_mat=pose_mat[-1],
            pose_rep=obs_pose_repr,
            backward=False)

        obs_pose = mat_to_pose10d(obs_pose_mat)
        obs_dict_np[robot_prefix + '_eef_pos'] = obs_pose[...,:3]
        obs_dict_np[robot_prefix + '_eef_rot_axis_angle'] = obs_pose[...,3:]
    
    # generate pose relative to other robot
    n_robots = len(robot_prefix_map)
    for robot_id in range(n_robots):
        # convert pose to mat
        assert f'robot{robot_id}' in robot_prefix_map
        tx_robota_tcpa = pose_to_mat(np.concatenate([
            env_obs[f'robot{robot_id}_eef_pos'],
            env_obs[f'robot{robot_id}_eef_rot_axis_angle']
        ], axis=-1))
        for other_robot_id in range(n_robots):
            if robot_id == other_robot_id:
                continue
            tx_robotb_tcpb = pose_to_mat(np.concatenate([
                env_obs[f'robot{other_robot_id}_eef_pos'],
                env_obs[f'robot{other_robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            tx_robota_robotb = tx_robot1_robot0
            if robot_id == 0:
                tx_robota_robotb = np.linalg.inv(tx_robot1_robot0)
            tx_robota_tcpb = tx_robota_robotb @ tx_robotb_tcpb

            rel_obs_pose_mat = convert_pose_mat_rep(
                tx_robota_tcpa,
                base_pose_mat=tx_robota_tcpb[-1],
                pose_rep='relative',
                backward=False)
            rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
            obs_dict_np[f'robot{robot_id}_eef_pos_wrt{other_robot_id}'] = rel_obs_pose[:,:3]
            obs_dict_np[f'robot{robot_id}_eef_rot_axis_angle_wrt{other_robot_id}'] = rel_obs_pose[:,3:]

    # generate relative pose with respect to episode start
    if episode_start_pose is not None:
        for robot_id in range(n_robots):        
            # convert pose to mat
            pose_mat = pose_to_mat(np.concatenate([
                env_obs[f'robot{robot_id}_eef_pos'],
                env_obs[f'robot{robot_id}_eef_rot_axis_angle']
            ], axis=-1))
            
            # get start pose
            start_pose = episode_start_pose[robot_id]
            start_pose_mat = pose_to_mat(start_pose)
            rel_obs_pose_mat = convert_pose_mat_rep(
                pose_mat,
                base_pose_mat=start_pose_mat,
                pose_rep='relative',
                backward=False)
            
            rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
            # obs_dict_np[f'robot{robot_id}_eef_pos_wrt_start'] = rel_obs_pose[:,:3]
            obs_dict_np[f'robot{robot_id}_eef_rot_axis_angle_wrt_start'] = rel_obs_pose[:,3:]

    return obs_dict_np

def get_real_umi_action(
        action: np.ndarray,
        env_obs: Dict[str, np.ndarray], 
        action_pose_repr: str='abs'
    ):

    n_robots = int(action.shape[-1] // 10)
    env_action = list()
    for robot_idx in range(n_robots):
        # convert pose to mat
        pose_mat = pose_to_mat(np.concatenate([
            env_obs[f'robot{robot_idx}_eef_pos'][-1],
            env_obs[f'robot{robot_idx}_eef_rot_axis_angle'][-1]
        ], axis=-1))

        start = robot_idx * 10
        action_pose10d = action[..., start:start+9]
        action_grip = action[..., start+9:start+10]
        action_pose_mat = pose10d_to_mat(action_pose10d)

        # solve relative action
        action_mat = convert_pose_mat_rep(
            action_pose_mat, 
            base_pose_mat=pose_mat,
            pose_rep=action_pose_repr,
            backward=True)

        # convert action to pose
        action_pose = mat_to_pose(action_mat)
        env_action.append(action_pose)
        env_action.append(action_grip)

    env_action = np.concatenate(env_action, axis=-1)
    return env_action

def get_real_gumi_action(
        action: np.ndarray,
        env_obs: Dict[str, np.ndarray], 
        action_pose_repr: str='abs'
    ):

    # 단일 로봇 가정 (n_robots = 1)
    action_pose10d = action[..., :9]  # 위치 + rotation_6d
    action_grip = action[..., 9:10]   # 그리퍼
    
    # 10차원 액션을 7차원으로 변환
    action_8d = convert_action_10d_to_8d(action_pose10d)
    
    # 위치 + 오일러 + 그리퍼
    env_action = np.concatenate([action_8d, action_grip], axis=-1)
    
    return env_action

def rotation_6d_to_euler(rotation_6d):
    """
    rotation_6d 표현을 오일러 각(roll, pitch, yaw)으로 변환
    
    입력:
    - rotation_6d: 형태 (B, 6) 또는 (6,) rotation_6d 표현 배열
    
    출력:
    - euler_angles: 형태 (B, 3) 또는 (3,) 오일러 각 배열 (라디안)
    """
    # 입력이 벡터인지 배치인지 확인
    is_vector = len(rotation_6d.shape) == 1
    if is_vector:
        rotation_6d = rotation_6d.reshape(1, 6)
    
    # 회전 행렬 초기화
    batch_size = rotation_6d.shape[0]
    rotation_matrix = np.zeros((batch_size, 3, 3))
    
    # 첫 두 열 가져오기
    x_axis = rotation_6d[:, 0:3]
    y_axis = rotation_6d[:, 3:6]
    
    # 정규화
    x_axis = x_axis / np.linalg.norm(x_axis, axis=1, keepdims=True)
    
    # y_axis를 x_axis에 직교하도록 정규화
    y_axis = y_axis - np.sum(x_axis * y_axis, axis=1, keepdims=True) * x_axis
    y_axis = y_axis / np.linalg.norm(y_axis, axis=1, keepdims=True)
    
    # z_axis는 x_axis와 y_axis의 외적
    z_axis = np.cross(x_axis, y_axis)
    
    # 회전 행렬 구성
    rotation_matrix[:, :, 0] = x_axis
    rotation_matrix[:, :, 1] = y_axis
    rotation_matrix[:, :, 2] = z_axis
    
    # 회전 행렬을 오일러 각으로 변환 (xyz 순서)
    euler_angles = R.from_matrix(rotation_matrix).as_euler('xyz')
    
    # 벡터 입력이었으면 벡터로 반환
    if is_vector:
        euler_angles = euler_angles.squeeze()
    
    return euler_angles

def convert_action_10d_to_7d(action_10d):
    """
    10차원 액션 [위치(3) + rotation_6d(6) + 그리퍼(1)]을
    7차원 액션 [위치(3) + 오일러(3) + 그리퍼(1)]로 변환
    """
    position = action_10d[..., :3]           # 위치 (변경 없음)
    rotation_6d = action_10d[..., 3:9]       # rotation_6d
    gripper = action_10d[..., 9:10]          # 그리퍼
    
    # rotation_6d -> 오일러 각
    euler_angles = rotation_6d_to_euler(rotation_6d)
    
    # 벡터/배치 처리
    if len(euler_angles.shape) == 1:
        euler_angles = euler_angles.reshape(1, 3)
        euler_angles = euler_angles[0]  # 다시 벡터로 변환
    
    # 결합하여 7차원 액션 생성
    if len(position.shape) == 1:
        # 단일 액션인 경우
        action_7d = np.concatenate([position, euler_angles, gripper])
    else:
        # 배치 액션인 경우
        action_7d = np.concatenate([position, euler_angles, gripper], axis=-1)
    
    return action_7d

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