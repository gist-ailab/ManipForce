import copy
import os
import pathlib
import shutil
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
import zarr
from PIL import Image
from filelock import FileLock
from scipy.spatial.transform import Rotation as R
from threadpoolctl import threadpool_limits
from tqdm import tqdm, trange

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.multimodal_replay_buffer import MultiModalReplayBuffer
from diffusion_policy.common.normalize_util import (
    array_to_stats,
    concatenate_normalizer,
    get_identity_normalizer_from_stat,
    get_image_identity_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    ManipForceSequenceSampler,
    get_val_mask,
)
from diffusion_policy.dataset.base_dataset import BaseDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


register_codecs()

def convert_action_8d_to_10d(action_8d):
    """Convert 8D action (7D pose + 1D gripper) to 10D"""
    position = action_8d[..., :3]  # Keep position as is
    quat = action_8d[..., 3:7]     # Quaternion
    gripper_state = action_8d[..., 7:8]  # Gripper state
    
    # Convert quaternion to rotation matrix
    from scipy.spatial.transform import Rotation as R
    rot = R.from_quat(quat)  # Need to check [x, y, z, w] order
    rotation_matrix = rot.as_matrix()
    
    # Use the first two columns of the rotation matrix (6D)
    rotation_6d = np.concatenate([
        rotation_matrix[..., :3, 0],  # First column
        rotation_matrix[..., :3, 1]   # Second column
    ], axis=-1)
    
    # Create final 10D vector (position + rotation_6d + gripper_state)
    action_10d = np.concatenate([
        position,      # 3D
        rotation_6d,   # 6D
        gripper_state  # 1D (actual gripper state)
    ], axis=-1)
    return action_10d

def resize_with_padding(img, target_size=224):
    """
    Maintain the aspect ratio of the image while adding padding to make it square
    img: (H, W, C) uint8
    return: (224, 224, C) uint8
    """
    h, w = img.shape[:2]
    aspect = w / h
    
    if aspect > 1:  # If width is longer (1280x800)
        new_w = target_size
        new_h = int(target_size / aspect)
    else:  # If height is longer (640x480)
        new_h = target_size
        new_w = int(target_size * aspect)
    
    # Resize with PIL
    pil_img = Image.fromarray(img)
    resized = pil_img.resize((new_w, new_h))
    
    # Padding
    new_img = Image.new('RGB', (target_size, target_size), (0, 0, 0))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    new_img.paste(resized, (paste_x, paste_y))
    
    return np.array(new_img)

# def normalize_ft_window(ft_window: np.ndarray, method: str = "mean_diff"):
#     W, D = ft_window.shape
#     if method in ("mean", "mean_diff"):
#         mean_vec = np.mean(ft_window, axis=0, keepdims=True)
#         ft_mc = ft_window - mean_vec
#     else:
#         ft_mc = ft_window
#     if method in ("diff", "mean_diff"):
#         ft_norm = ft_mc[1:, :] - ft_mc[:-1, :]
#     else:
#         ft_norm = ft_mc
#     return ft_norm

# def ft_to_ee(ft_s: np.ndarray) -> np.ndarray:
#     """
#     ft_s : (...,6)  [Fx, Fy, Fz, Tx, Ty, Tz]  sensor-based
#     return: (...,6)  EE-based
#     Coordinate transformation: [x,y,z] → [-y, x, -z]
#     """
#     R_align = np.array([[0,-1,0],   # x → -y
#                         [1,0,0],    # y → x  
#                         [0,0,-1]], dtype=ft_s.dtype)  # z → -z

#     f   = (R_align @ ft_s[..., :3 ].T).T       # Force: [Fx, Fy, Fz] → [-Fy, Fx, -Fz]
#     tau = (R_align @ ft_s[..., 3: ].T).T       # Torque: [Tx, Ty, Tz] → [-Ty, Tx, -Tz]
    
#     # Set tau to 0 for testing
#     tau = np.zeros_like(tau)
    
#     return np.concatenate([f, tau], axis=-1)

class ManipForceDataset(BaseDataset):
    def __init__(self,
        shape_meta: dict,
        dataset_path: str,
        cache_dir: Optional[str]=None,
        pose_repr: dict={},
        action_padding: bool=False,
        temporally_independent_normalization: bool=False,
        repeat_frame_prob: float=0.0,
        modal_masking_prob: float=0.0,  # disabled by default
        mask_strategy: str="random_timesteps",  # "random_timesteps" or "full_modal"
        seed: int=42,
        val_ratio: float=0.0,
        max_duration: Optional[float]=None,
        ft_hz: float=200.0
    ):
        self.ft_hz = ft_hz
        self.pose_repr = pose_repr
        self.obs_pose_repr = self.pose_repr.get('obs_pose_repr', 'rel')
        self.action_pose_repr = self.pose_repr.get('action_pose_repr', 'rel')
        
        if cache_dir is None:
            # Modify to handle both directory and zip files
            if dataset_path.endswith('.zarr.zip'):
                store = zarr.ZipStore(dataset_path, mode='r')
            else:
                store = zarr.DirectoryStore(dataset_path)            
            replay_buffer = MultiModalReplayBuffer.copy_from_store(
                src_store=store, 
                store=zarr.MemoryStore(),
                ft_store=zarr.MemoryStore(),  # Separate store for FT data
                image_keys=['handeye_cam_1', 'handeye_cam_2', 'action', 'state', 'img_timestamps', 'pose_wrt_start'],  # image related data
                ft_keys=['ft_data', 'ft_timestamps']  # FT related data
            )            
        
            # Only close ZipStore
            if isinstance(store, zarr.ZipStore):
                store.close()
        else:
            # TODO: refactor into a stand alone function?
            # determine path name
            mod_time = os.path.getmtime(dataset_path)
            stamp = datetime.fromtimestamp(mod_time).isoformat()
            stem_name = os.path.basename(dataset_path).split('.')[0]
            cache_name = '_'.join([stem_name, stamp])
            cache_dir = pathlib.Path(os.path.expanduser(cache_dir))
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir.joinpath(cache_name + '.zarr.mdb')
            lock_path = cache_dir.joinpath(cache_name + '.lock')
            # load cached file
            print('Acquiring lock on cache.')
            with FileLock(lock_path):
                # cache does not exist
                if not cache_path.exists():
                    try:
                        with zarr.LMDBStore(str(cache_path),     
                            writemap=True, metasync=False, sync=False, map_async=True, lock=False
                            ) as lmdb_store:
                            with zarr.ZipStore(dataset_path, mode='r') as zip_store:
                                print(f"Copying data to {str(cache_path)}")
                                ReplayBuffer.copy_from_store(
                                    src_store=zip_store,
                                    store=lmdb_store
                                )
                        print("Cache written to disk!")
                    except Exception as e:
                        shutil.rmtree(cache_path)
                        raise e
            
            # open read-only lmdb store
            store = zarr.LMDBStore(str(cache_path), readonly=True, lock=False)
            replay_buffer = ReplayBuffer.create_from_group(
                group=zarr.group(store)
            )
        
        '''
        replay_buffer data structure (varies per episode):
        
        === Dataset Information ===
        Source data keys: ['action', 'ft_data', 'ft_timestamps', 'img', 'img_timestamps', 'state']
        Source meta keys: ['episode_ends', 'episode_ft_ends', 'episode_img_ends']
        
        === Image Related Data === 
        replay_buffer['action']: action data, shape (142, 7), dtype float32
        replay_buffer['img']: image data, shape (142, 800, 1280, 3), dtype uint8
        replay_buffer['img_timestamps']: image timestamps, shape (142,), dtype float64
        replay_buffer['state']: state data, shape (142, 7), dtype float32
        
        === Metadata ===
        replay_buffer.meta['episode_ends']: episode transitions, shape (5,), dtype int64
        replay_buffer.meta['episode_ft_ends']: FT episode transitions, shape (5,), dtype int64
        replay_buffer.meta['episode_img_ends']: image episode transitions, shape (5,), dtype int64
        
        === FT Data ===
        replay_buffer.ft_data['ft_data']: FT sensor data, shape (522, 7), dtype float32
        replay_buffer.ft_data['ft_timestamps']: FT timestamps, shape (522,), dtype float64
        
        === Data Access Methods ===
        1. image/action/state data: direct access with replay_buffer['key']
        2. FT data: access with replay_buffer.ft_data['key']
        3. metadata: access with replay_buffer.meta['key']
        4. FT data matching a specific image: replay_buffer.get_matching_ft_data(img_idx, window_size)
        
        >>> Note <<<
        image data (142 frames) and FT data (522 frames) have different lengths
        Matching must be performed using timestamps
        
        '''
        
        self.num_robot = 0
        rgb_keys = list()
        ft_keys = list()
        low_dim_keys = list() # added
        img_timestamp_keys = list()
        ft_timestamp_keys = list()
        key_horizon = dict()
        key_down_sample_steps = dict()
        key_latency_steps = dict()
        obs_shape_meta = shape_meta['obs']
        # Check dataset structure (actual Zarr file)
        print("Checking dataset structure:")
        print(f"Keys in replay_buffer: {list(replay_buffer.keys())}")
        if 'meta' in replay_buffer:
            print(f"Meta keys: {list(replay_buffer['meta'].keys())}")
        
        # Check FT data structure
        print("\n=== Checking FT data ===")
        has_ft_attr = hasattr(replay_buffer, 'ft_data')

        # Check new data structure - modified version
        has_ft_data = False
        has_ft_timestamps = False

        if has_ft_attr and replay_buffer.ft_data is not None:
            has_ft_data = 'ft_data' in replay_buffer.ft_data
            has_ft_timestamps = 'ft_timestamps' in replay_buffer.ft_data
            
        has_img_timestamps = 'img_timestamps' in replay_buffer
        has_separate_episode_ends = False

        if 'meta' in replay_buffer:
            has_separate_episode_ends = ('episode_ft_ends' in replay_buffer.meta and
                                       'episode_img_ends' in replay_buffer.meta)
        # Extract key info from shape_meta
        for key, attr in obs_shape_meta.items():
            # Classify keys by data type
            type = attr.get('type', 'ft')  # default was 'ft'
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'ft':
                ft_keys.append(key)
            elif type == 'low_dim':  # relative_ee_pose should go here
                if not attr.get('ignore_by_policy', False):
                    low_dim_keys.append(key)
            elif type == 'img_timestamp' or type == 'img_timestamps':
                img_timestamp_keys.append(key)
            elif type == 'ft_timestamp' or type == 'ft_timestamps':
                ft_timestamp_keys.append(key)
            
            self.low_dim_keys = low_dim_keys

            if key.endswith('eef_pos'):
                self.num_robot += 1

            # solve obs_horizon
            horizon = shape_meta['obs'][key]['horizon']
            key_horizon[key] = horizon

            # solve latency_steps
            latency_steps = shape_meta['obs'][key]['latency_steps']
            key_latency_steps[key] = latency_steps

            # solve down_sample_steps
            down_sample_steps = shape_meta['obs'][key]['down_sample_steps']
            key_down_sample_steps[key] = down_sample_steps

        # solve action
        key_horizon['action'] = shape_meta['action']['horizon']
        key_latency_steps['action'] = shape_meta['action']['latency_steps']
        key_down_sample_steps['action'] = shape_meta['action']['down_sample_steps']

        # Timestamp processing - identify and add img_timestamps key
        if has_img_timestamps:
            correct_key = 'img_timestamps'  # actual data key name (plural)
            
            cam_key     = rgb_keys[0]
            cam_meta    = shape_meta['obs'][cam_key]
            cam_horizon = cam_meta['horizon']
            cam_latency = cam_meta['latency_steps']
            cam_down    = cam_meta['down_sample_steps']

            # step setup for img_timestamps
            key_horizon[correct_key]         = cam_horizon
            key_latency_steps[correct_key]   = cam_latency
            key_down_sample_steps[correct_key]= cam_down

            # Refresh timestamp keys list
            img_timestamp_keys[:] = [correct_key]
        
        # Check and add ft_timestamps key (only if FT data exists)
        if has_ft_timestamps:
            correct_key = 'ft_timestamps'  # actual data key name (plural)
            
            # Modify if timestamp key is missing or singular
            if not ft_timestamp_keys or ('ft_timestamp' in ft_timestamp_keys and correct_key not in ft_timestamp_keys):
                if 'ft_timestamp' in ft_timestamp_keys:
                    ft_timestamp_keys.remove('ft_timestamp')
                
                # print(f"Unified FT timestamp key to plural: ft_timestamps")
                ft_timestamp_keys.append(correct_key)
                
                # Update key name and horizon value settings
                if 'ft_timestamp' in key_horizon:
                    key_horizon[correct_key] = key_horizon.pop('ft_timestamp')
                else:
                    key_horizon[correct_key] = key_horizon.get('ft_data', 1)
                
                if 'ft_timestamp' in key_latency_steps:
                    key_latency_steps[correct_key] = key_latency_steps.pop('ft_timestamp')
                else:
                    key_latency_steps[correct_key] = key_latency_steps.get('ft_data', 0)
                
                if 'ft_timestamp' in key_down_sample_steps:
                    key_down_sample_steps[correct_key] = key_down_sample_steps.pop('ft_timestamp')
                else:
                    key_down_sample_steps[correct_key] = key_down_sample_steps.get('ft_data', 1)

        # Add ft_data key if missing
        if has_ft_data and 'ft_data' not in ft_keys:
            # print("Add 'ft_data' key to ft_keys")
            ft_keys.append('ft_data')
            if 'ft_data' not in key_horizon:
                key_horizon['ft_data'] = -1  # Special value for dynamic sampling
            if 'ft_data' not in key_latency_steps:
                key_latency_steps['ft_data'] = 0
            if 'ft_data' not in key_down_sample_steps:
                key_down_sample_steps['ft_data'] = down_sample_steps

        # Setup episode_ends
        episode_ends = None
        if has_separate_episode_ends:
            episode_ends = replay_buffer['meta']['episode_ft_ends']
        else:
            episode_ends = replay_buffer.episode_ends

        # Create validation mask
        val_mask = get_val_mask(
            n_episodes=len(episode_ends), 
            val_ratio=val_ratio,
            seed=seed
        )
        train_mask = ~val_mask

        # Create list of keys to include (only FT keys without 'wrt')
        self.sampler_ft_keys = list()
        for key in ft_keys:
            if not 'wrt' in key:
                self.sampler_ft_keys.append(key)

        # Create multimodal sampler
        sampler = ManipForceSequenceSampler(
            shape_meta=shape_meta,
            replay_buffer=replay_buffer,
            rgb_keys=rgb_keys,
            lowdim_keys=self.sampler_ft_keys + img_timestamp_keys + ft_timestamp_keys + low_dim_keys,
            key_horizon=key_horizon,
            key_latency_steps=key_latency_steps,
            key_down_sample_steps=key_down_sample_steps,
            episode_mask=train_mask,
            action_padding=action_padding,
            repeat_frame_prob=repeat_frame_prob,
            max_duration=max_duration,
            img_hz=30.0,  # image sampling rate (Hz)
            ft_hz=self.ft_hz   # FT sensor sampling rate (Hz)
        )
        # Add debug output
        
        # Debug index creation process
        valid_indices = 0
        skipped_indices = 0
        for current_idx in range(len(episode_ends)):
            if not action_padding and len(episode_ends) < current_idx + (key_horizon['action'] - 1) * key_down_sample_steps['action'] + 1:
                skipped_indices += 1
                continue
            valid_indices += 1
        
        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.rgb_keys = rgb_keys
        self.ft_keys = self.sampler_ft_keys
        self.img_timestamp_keys = img_timestamp_keys
        self.ft_timestamp_keys = ft_timestamp_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        self.val_mask = val_mask
        self.action_padding = action_padding
        self.repeat_frame_prob = repeat_frame_prob
        self.max_duration = max_duration
        self.sampler = sampler
        self.episode_ends = episode_ends
        self.temporally_independent_normalization = temporally_independent_normalization
        self.threadpool_limits_is_applied = False
        self.has_ft_data = has_ft_data
        self.has_ft_timestamps = has_ft_timestamps
        self.has_img_timestamps = has_img_timestamps

        # Set default robot count to 1 if not specified
        self.num_robot = 0
        for key in ft_keys:
            if key.endswith('eef_pos'):
                self.num_robot += 1
        if self.num_robot == 0:
            # Infer number of robots from action dimension
            if 'action' in self.replay_buffer:
                action_dim = self.replay_buffer['action'].shape[-1]
                if action_dim == 7:  # typical 7D robot arm vector
                    self.num_robot = 1
                    
        # Dataset summary output
        print("\n" + "="*50)
        print("ManipForceDataset Summary")
        print("="*50)
        print(f"Total number of episodes: {len(episode_ends)}")
        
        # Calculate episode length stats
        episode_lengths = []
        for i in range(len(episode_ends)):
            start_idx = episode_ends[i-1] if i > 0 else 0
            end_idx = episode_ends[i]
            episode_lengths.append(end_idx - start_idx)
        
        # Episode length stats
        avg_length = np.mean(episode_lengths)
        min_length = np.min(episode_lengths)
        max_length = np.max(episode_lengths)
        total_steps = sum(episode_lengths)
        
        print(f"Total timesteps: {total_steps}")
        print(f"Average episode length: {avg_length:.1f} timesteps")
        print(f"Min/Max episode length: {min_length}/{max_length} timesteps")
        
        # Dataset components
        print(f"\nObservation Data:")
        print(f"  - RGB Keys ({len(self.rgb_keys)}): {', '.join(self.rgb_keys)}")
        print(f"  - FT Keys ({len(self.ft_keys)}): {', '.join(self.ft_keys)}")
        print(f"  - Image Timestamp Keys: {', '.join(self.img_timestamp_keys)}")
        print(f"  - FT Timestamp Keys: {', '.join(self.ft_timestamp_keys)}")
        
        # Action info
        if 'action' in self.replay_buffer:
            action_shape = self.replay_buffer['action'].shape
            print(f"\nAction Information:")
            print(f"  - Dimension: {action_shape[-1]}")
        
        # Train/Val split
        train_episodes = np.sum(~self.val_mask)
        val_episodes = np.sum(self.val_mask)
        print(f"\nData Split:")
        print(f"  - Training Episodes: {train_episodes} ({train_episodes/len(episode_ends)*100:.1f}%)")
        print(f"  - Validation Episodes: {val_episodes} ({val_episodes/len(episode_ends)*100:.1f}%)")
        
        # Sampler info
        print(f"\nSequence Sampler:")
        print(f"  - Training Samples: {len(self.sampler)}")
        print(f"  - Action Padding: {self.action_padding}")
        print("="*50 + "\n")

        # 🆕 Add masking related attributes
        self.modal_masking_prob = modal_masking_prob
        self.mask_strategy = mask_strategy
        self.mask_rng = np.random.RandomState(seed + 1000)  # separate seed for independence

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = ManipForceSequenceSampler(
            shape_meta=self.shape_meta,
            replay_buffer=self.replay_buffer,
            rgb_keys=self.rgb_keys,
            lowdim_keys=self.ft_keys + self.img_timestamp_keys + self.ft_timestamp_keys + self.low_dim_keys,
            key_horizon=self.key_horizon,
            key_latency_steps=self.key_latency_steps,
            key_down_sample_steps=self.key_down_sample_steps,
            episode_mask=self.val_mask,
            action_padding=self.action_padding,
            repeat_frame_prob=self.repeat_frame_prob,
            max_duration=self.max_duration,
            img_hz=30.0,  # image sampling rate (Hz)
            ft_hz=self.ft_hz   # FT sensor sampling rate (Hz)
        )
        val_set.val_mask = ~self.val_mask
        return val_set
    
    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # collect ft_data, action, and pose data
        data_cache = {'ft_data': [], 'action': [], 'pose_wrt_start': []}

        # temporarily ignore RGB to speed things up
        self.sampler.ignore_rgb(True)
        loader = torch.utils.data.DataLoader(self, batch_size=64, num_workers=32)
        for batch in tqdm(loader, desc='collecting for norm'):
            # ft_data processing (legacy code)
            ft_np = batch['obs']['ft_data'].detach().cpu().numpy()
            for ft_seq in ft_np:
                data_cache['ft_data'].append(ft_seq)
                
            # action processing (legacy code)
            ac_np = batch['action'].detach().cpu().numpy()
            for ac_seq in ac_np:
                data_cache['action'].append(ac_seq)
            
            # add pose_wrt_start processing
            if 'pose_wrt_start' in batch['obs']:
                pose_np = batch['obs']['pose_wrt_start'].detach().cpu().numpy()
                for pose_seq in pose_np:
                    data_cache['pose_wrt_start'].append(pose_seq)

        self.sampler.ignore_rgb(False)

        # # FT normalization (legacy code)
        # arr_ft = np.stack(data_cache['ft_data'], axis=0)
        # flat_ft = arr_ft.reshape(-1, arr_ft.shape[-1])
        # normalizer['ft_data'] = get_range_normalizer_from_stat(array_to_stats(flat_ft))

        # Action normalization (legacy code)
        arr_ac = np.concatenate(data_cache['action'], axis=0)
        normalizer['action'] = get_range_normalizer_from_stat(array_to_stats(arr_ac))

        # add pose_wrt_start normalization
        if data_cache['pose_wrt_start']:
            pose_data = np.stack(data_cache['pose_wrt_start'], axis=0)
            # Separate position(3) and rotation(4) for normalization
            pos_normalizer = get_range_normalizer_from_stat(array_to_stats(pose_data[...,:3]))
            rot_normalizer = get_identity_normalizer_from_stat(array_to_stats(pose_data[...,3:]))
            normalizer['pose_wrt_start'] = concatenate_normalizer([pos_normalizer, rot_normalizer])
            
        # RGB normalization (legacy code)
        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
        
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.threadpool_limits_is_applied:
            threadpool_limits(1)
            self.threadpool_limits_is_applied = True

        # 1) Sample data from sampler
        data = self.sampler.sample_sequence(idx)
        obs_dict: Dict[str, np.ndarray] = {}

        # Added 0813, fixed by SJ
        # Process only if pose_wrt_start exists in shape_meta
        if 'pose_wrt_start' in self.shape_meta['obs'] and 'pose_wrt_start' in data:
            # position(3) + orientation(4) = 7D
            # relative_pose = np.concatenate([
            #     np.array(data['pose_wrt_start']['position']),
            #     np.array(data['pose_wrt_start']['orientation'])
            # ], axis=-1).astype(np.float32)
            obs_dict['pose_wrt_start'] = data['pose_wrt_start']

        # 3) RGB image processing + masking
        for key in self.rgb_keys:
            if key in data:
                img = np.moveaxis(data[key], -1, 1).astype(np.float32) / 255.0  # (T, C, H, W)
                
                # Legacy resizing logic
                expected_shape = self.shape_meta['obs'][key]['shape']
                if img.shape[2:] != expected_shape[1:]:
                    resized_imgs = []
                    for t in range(img.shape[0]):
                        curr_img = np.moveaxis(img[t], 0, -1)
                        curr_img_uint8 = (curr_img * 255).astype(np.uint8)
                        resized_pil = resize_with_padding(curr_img_uint8, target_size=expected_shape[1])
                        resized_np = resized_pil.astype(np.float32) / 255.0
                        resized_np = np.moveaxis(resized_np, -1, 0)
                        resized_imgs.append(resized_np)
                    img = np.stack(resized_imgs, axis=0)
                
                obs_dict[key] = img

        # 4) FT data processing + masking
        if 'ft_data' in data and 'ft_timestamps' in data:
            raw_ft = data['ft_data'].astype(np.float32)  # (8,6)
            obs_dict['ft_data'] = raw_ft
            
            if 'ft_timestamps' in data:
                # obs_dict['ft_timestamps'] = data['ft_timestamps'][1:]
                obs_dict['ft_timestamps'] = data['ft_timestamps']

        # 6) Action processing
        if 'action' in data:
            action_data = data['action'].astype(np.float32)
            # 🆕 Expand 7D action to 8D (add gripper dummy)
            if action_data.shape[-1] == 7:
                T = action_data.shape[0]
                # Gripper status dummy value (default: 1.0 = open)
                dummy_gripper = np.ones((T, 1), dtype=np.float32)  # (T, 1)
                action_data = np.concatenate([action_data, dummy_gripper], axis=-1)  # (T, 8)
            elif action_data.shape[-1] == 8:
                pass
            else:
                raise ValueError(f"Unsupported action dimension: {action_data.shape[-1]}. Expected 7 or 8.")
                
        else:
            return {'obs': obs_dict, 'action': torch.zeros((8, 10))}

        # 8) numpy → torch conversion
        torch_obs = dict_apply(obs_dict, lambda x: torch.from_numpy(x).float() if isinstance(x, np.ndarray) else torch.tensor(x))
        # 9) Action 8D → 10D conversion (pose + gripper)
        if action_data.shape[-1] == 8:
            action_data = convert_action_8d_to_10d(action_data)
        torch_action = torch.from_numpy(action_data).float()

        final_result = {
            'obs': torch_obs,
            'action': torch_action
        }
        return final_result