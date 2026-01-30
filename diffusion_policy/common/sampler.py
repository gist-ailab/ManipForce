from typing import Optional, Union
import numpy as np
import random
import scipy.interpolate as si
import scipy.spatial.transform as st
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.multimodal_replay_buffer import MultiModalReplayBuffer


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


class SequenceSampler:
    def __init__(self,
        shape_meta: dict,
        replay_buffer: ReplayBuffer,
        rgb_keys: list,
        lowdim_keys: list,
        key_horizon: dict,
        key_latency_steps: dict,
        key_down_sample_steps: dict,
        episode_mask: Optional[np.ndarray]=None,
        action_padding: bool=False,
        repeat_frame_prob: float=0.0,
        max_duration: Optional[float]=None
    ):
        episode_ends = replay_buffer.episode_ends[:]

        # load gripper_width
        gripper_width = replay_buffer['robot0_gripper_width'][:, 0]
        gripper_width_threshold = 0.08
        self.repeat_frame_prob = repeat_frame_prob

        # create indices, including (current_idx, start_idx, end_idx)
        indices = list()
        for i in range(len(episode_ends)):
            before_first_grasp = True # initialize for each episode
            if episode_mask is not None and not episode_mask[i]:
                # skip episode
                continue
            start_idx = 0 if i == 0 else episode_ends[i-1]
            end_idx = episode_ends[i]
            if max_duration is not None:
                end_idx = min(end_idx, max_duration * 60)
            for current_idx in range(start_idx, end_idx):
                if not action_padding and end_idx < current_idx + (key_horizon['action'] - 1) * key_down_sample_steps['action'] + 1:
                    continue
                if gripper_width[current_idx] < gripper_width_threshold:
                    before_first_grasp = False
                indices.append((current_idx, start_idx, end_idx, before_first_grasp))
        
        # load low_dim to memory and keep rgb as compressed zarr array
        self.replay_buffer = dict()
        self.num_robot = 0
        for key in lowdim_keys:
            if key.endswith('eef_pos'):
                self.num_robot += 1

            if key.endswith('pos_abs'):
                axis = shape_meta['obs'][key]['axis']
                if isinstance(axis, int):
                    axis = [axis]
                self.replay_buffer[key] = replay_buffer[key[:-4]][:, list(axis)]
            elif key.endswith('quat_abs'):
                axis = shape_meta['obs'][key]['axis']
                if isinstance(axis, int):
                    axis = [axis]
                # HACK for hybrid abs/relative proprioception
                rot_in = replay_buffer[key[:-4]][:]
                rot_out = st.Rotation.from_quat(rot_in).as_euler('XYZ')
                self.replay_buffer[key] = rot_out[:, list(axis)]
            elif key.endswith('axis_angle_abs'):
                axis = shape_meta['obs'][key]['axis']
                if isinstance(axis, int):
                    axis = [axis]
                rot_in = replay_buffer[key[:-4]][:]
                rot_out = st.Rotation.from_rotvec(rot_in).as_euler('XYZ')
                self.replay_buffer[key] = rot_out[:, list(axis)]
            else:
                self.replay_buffer[key] = replay_buffer[key][:]
        for key in rgb_keys:
            self.replay_buffer[key] = replay_buffer[key]
        
        
        if 'action' in replay_buffer:
            self.replay_buffer['action'] = replay_buffer['action'][:]
        else:
            # construct action (concatenation of [eef_pos, eef_rot, gripper_width])
            actions = list()
            for robot_idx in range(self.num_robot):
                for cat in ['eef_pos', 'eef_rot_axis_angle', 'gripper_width']:
                    key = f'robot{robot_idx}_{cat}'
                    if key in self.replay_buffer:
                        actions.append(self.replay_buffer[key])
            self.replay_buffer['action'] = np.concatenate(actions, axis=-1)

        self.action_padding = action_padding
        self.indices = indices
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        
        self.ignore_rgb_is_applied = False # speed up the interation when getting normalizaer

    def __len__(self):
        return len(self.indices)
    
    def sample_sequence(self, idx):
        current_idx, start_idx, end_idx, before_first_grasp = self.indices[idx]

        result = dict()

        obs_keys = self.rgb_keys + self.lowdim_keys
        if self.ignore_rgb_is_applied:
            obs_keys = self.lowdim_keys

        # observation
        for key in obs_keys:
            input_arr = self.replay_buffer[key]
            this_horizon = self.key_horizon[key]
            this_latency_steps = self.key_latency_steps[key]
            this_downsample_steps = self.key_down_sample_steps[key]
            
            if key in self.rgb_keys:
                assert this_latency_steps == 0
                num_valid = min(this_horizon, (current_idx - start_idx) // this_downsample_steps + 1)
                slice_start = current_idx - (num_valid - 1) * this_downsample_steps

                output = input_arr[slice_start: current_idx + 1: this_downsample_steps]
                assert output.shape[0] == num_valid
                
                # solve padding
                if output.shape[0] < this_horizon:
                    padding = np.repeat(output[:1], this_horizon - output.shape[0], axis=0)
                    output = np.concatenate([padding, output], axis=0)
            else:
                idx_with_latency = np.array(
                    [current_idx - idx * this_downsample_steps + this_latency_steps for idx in range(this_horizon)],
                    dtype=np.float32)
                idx_with_latency = idx_with_latency[::-1]
                idx_with_latency = np.clip(idx_with_latency, start_idx, end_idx - 1)
                interpolation_start = max(int(idx_with_latency[0]) - 5, start_idx)
                interpolation_end = min(int(idx_with_latency[-1]) + 2 + 5, end_idx)

                if 'rot' in key:
                    # rotation
                    rot_preprocess, rot_postprocess = None, None
                    if key.endswith('quat'):
                        rot_preprocess = st.Rotation.from_quat
                        rot_postprocess = st.Rotation.as_quat
                    elif key.endswith('axis_angle'):
                        rot_preprocess = st.Rotation.from_rotvec
                        rot_postprocess = st.Rotation.as_rotvec
                    else:
                        raise NotImplementedError
                    slerp = st.Slerp(
                        times=np.arange(interpolation_start, interpolation_end),
                        rotations=rot_preprocess(input_arr[interpolation_start: interpolation_end]))
                    output = rot_postprocess(slerp(idx_with_latency))
                else:
                    interp = si.interp1d(
                        x=np.arange(interpolation_start, interpolation_end),
                        y=input_arr[interpolation_start: interpolation_end],
                        axis=0, assume_sorted=True)
                    output = interp(idx_with_latency)
                
            result[key] = output

        # repeat frame before first grasp
        if self.repeat_frame_prob != 0.0:
            if before_first_grasp and random.random() < self.repeat_frame_prob:
                for key in obs_keys:
                    result[key][:-1] = result[key][-1:]

        # aciton
        input_arr = self.replay_buffer['action']
        action_horizon = self.key_horizon['action']
        action_latency_steps = self.key_latency_steps['action']
        assert action_latency_steps == 0
        action_down_sample_steps = self.key_down_sample_steps['action']
        slice_end = min(end_idx, current_idx + (action_horizon - 1) * action_down_sample_steps + 1)
        output = input_arr[current_idx: slice_end: action_down_sample_steps]
        # solve padding
        if not self.action_padding:
            assert output.shape[0] == action_horizon
        elif output.shape[0] < action_horizon:
            padding = np.repeat(output[-1:], action_horizon - output.shape[0], axis=0)
            output = np.concatenate([output, padding], axis=0)
        result['action'] = output

        return result
    
    def ignore_rgb(self, apply=True):
        self.ignore_rgb_is_applied = apply
        

class ManipForceSequenceSampler:
    def __init__(self,
        shape_meta: dict,
        replay_buffer: Union[ReplayBuffer, MultiModalReplayBuffer],
        rgb_keys: list,
        lowdim_keys: list,
        key_horizon: dict,
        key_latency_steps: dict,
        key_down_sample_steps: dict,
        episode_mask: Optional[np.ndarray]=None,
        action_padding: bool=False,
        repeat_frame_prob: float=0.0,
        max_duration: Optional[float]=None,
        img_hz: float=30.0,  # image sampling rate (Hz)
        ft_hz: float=200.0   # FT sensor sampling rate (Hz)
    ):
        self.episode_ends = replay_buffer.episode_ends[:]
        
        episode_ends = self.episode_ends  # keep compatibility with legacy code
        
        # load gripper_width
        if 'robot0_gripper_width' in replay_buffer:
            gripper_width = replay_buffer['robot0_gripper_width'][:, 0]
        else:
            # Use default value higher than threshold (assume always open condition)
            gripper_width = np.ones(len(replay_buffer['action'])) * 0.1
            
        gripper_width_threshold = 0.08
        self.repeat_frame_prob = repeat_frame_prob

        # create indices, including (current_idx, start_idx, end_idx)
        indices = list()
        for i in range(len(episode_ends)):
            before_first_grasp = True # initialize for each episode
            if episode_mask is not None and not episode_mask[i]:
                # skip episode
                continue
            start_idx = 0 if i == 0 else episode_ends[i-1]
            end_idx = episode_ends[i]
            if max_duration is not None:
                end_idx = min(end_idx, max_duration * 60)
            # Reduce sample size by generating indices at down_sample_steps intervals
            step = key_down_sample_steps['action']
            for current_idx in range(start_idx, end_idx, step):
            # for current_idx in range(start_idx, end_idx):
                if not action_padding and end_idx < current_idx + (key_horizon['action'] - 1) * key_down_sample_steps['action'] + 1:
                    continue
                if gripper_width[current_idx] < gripper_width_threshold:
                    before_first_grasp = False
                # Also store episode index
                indices.append((current_idx, start_idx, end_idx, before_first_grasp, i))  # Add episode index i
        
        # load low_dim to memory and keep rgb as compressed zarr array
        self.replay_buffer = dict()
        self.num_robot = 0

        # Add FT data processing
        self.has_ft_data = False
        self.original_replay_buffer = replay_buffer  # Store original buffer
        
        # Check FT data access in MultiModalReplayBuffer
        if hasattr(replay_buffer, 'ft_data') and replay_buffer.ft_data is not None:
            self.has_ft_data = True
            
            # Add FT data directly to self.replay_buffer
            if hasattr(replay_buffer.ft_data, 'keys'):
                if 'ft_data' in replay_buffer.ft_data:
                    self.replay_buffer['ft_data'] = replay_buffer.ft_data['ft_data']
                
                if 'ft_timestamps' in replay_buffer.ft_data:
                    self.replay_buffer['ft_timestamps'] = replay_buffer.ft_data['ft_timestamps']
            
                
        # Add image timestamps (critical)
        if 'img_timestamps' in replay_buffer:
            self.replay_buffer['img_timestamps'] = replay_buffer['img_timestamps'][:]
        
        # RGB data processing
        for key in rgb_keys:
            self.replay_buffer[key] = replay_buffer[key]
        
        # Action data processing
        if 'action' in replay_buffer:
            self.replay_buffer['action'] = replay_buffer['action'][:]
        else:
            # construct action (concatenation of [eef_pos, eef_rot, gripper_width])
            actions = list()
            for robot_idx in range(self.num_robot):
                for cat in ['eef_pos', 'eef_rot_axis_angle', 'gripper_width']:
                    key = f'robot{robot_idx}_{cat}'
                    if key in self.replay_buffer:
                        actions.append(self.replay_buffer[key])
            self.replay_buffer['action'] = np.concatenate(actions, axis=-1)
        
        if 'pose_wrt_start' in replay_buffer:
            self.replay_buffer['pose_wrt_start'] = replay_buffer['pose_wrt_start'][:]

        self.action_padding = action_padding
        
        # Reference
        # Calculate number of FT frames (FT data count within image interval, including start point)
        self.target_ft_frames = int(round(ft_hz / img_hz)) + 1
        
        # Limit to min 2, max 20
        self.target_ft_frames = max(2, min(20, self.target_ft_frames))
        self.indices = indices
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        
        self.ignore_rgb_is_applied = False # speed up the interation when getting normalizer

        # Check if indices are empty and add default index with warning
        if len(indices) == 0:
            print("WARNING: No valid indices created based on current criteria!")
            print("Adding default index to avoid empty dataset error...")
            
            # Add the beginning of the first episode as the default index
            if len(episode_ends) > 0:
                start_idx = 0
                end_idx = episode_ends[0]
                current_idx = start_idx
                indices.append((current_idx, start_idx, end_idx, True, 0))  # Add episode index
                print(f"Added default index: {indices[0]}")
        
        print(f"Final number of indices: {len(self.indices)}")


    def __len__(self):
        return len(self.indices)
    
    def sample_sequence(self, idx):
        # Get index information (including episode index)
        current_idx, start_idx, end_idx, before_first_grasp, episode_idx = self.indices[idx]

        # Initialize result dictionary and add metadata
        result = dict()
        # Add episode index and first observation index (for FT data matching)
        result['episode_idx'] = episode_idx
        result['first_obs_idx'] = current_idx

        obs_keys = self.rgb_keys + self.lowdim_keys
        if self.ignore_rgb_is_applied:
            obs_keys = self.lowdim_keys

        # observation
        for key in obs_keys:
            input_arr = self.replay_buffer[key]
            this_horizon = self.key_horizon[key]
            this_latency_steps = self.key_latency_steps[key]
            this_downsample_steps = self.key_down_sample_steps[key]
            
            if key in self.rgb_keys:
                assert this_latency_steps == 0
                num_valid = min(this_horizon, (current_idx - start_idx) // this_downsample_steps + 1)
                slice_start = current_idx - (num_valid - 1) * this_downsample_steps

                output = input_arr[slice_start: current_idx + 1: this_downsample_steps]
                assert output.shape[0] == num_valid
                
                # solve padding
                if output.shape[0] < this_horizon:
                    padding = np.repeat(output[:1], this_horizon - output.shape[0], axis=0)
                    output = np.concatenate([padding, output], axis=0)
            else:
                idx_with_latency = np.array(
                    [current_idx - idx * this_downsample_steps + this_latency_steps for idx in range(this_horizon)],
                    dtype=np.float32)
                idx_with_latency = idx_with_latency[::-1]
                idx_with_latency = np.clip(idx_with_latency, start_idx, end_idx - 1)
                interpolation_start = max(int(idx_with_latency[0]) - 5, start_idx)
                interpolation_end = min(int(idx_with_latency[-1]) + 2 + 5, end_idx)

                if 'rot' in key:
                    # rotation
                    rot_preprocess, rot_postprocess = None, None
                    if key.endswith('quat'):
                        rot_preprocess = st.Rotation.from_quat
                        rot_postprocess = st.Rotation.as_quat
                    elif key.endswith('axis_angle'):
                        rot_preprocess = st.Rotation.from_rotvec
                        rot_postprocess = st.Rotation.as_rotvec
                    else:
                        raise NotImplementedError
                    slerp = st.Slerp(
                        times=np.arange(interpolation_start, interpolation_end),
                        rotations=rot_preprocess(input_arr[interpolation_start: interpolation_end]))
                    output = rot_postprocess(slerp(idx_with_latency))
                else:
                    interp = si.interp1d(
                        x=np.arange(interpolation_start, interpolation_end),
                        y=input_arr[interpolation_start: interpolation_end],
                        axis=0, assume_sorted=True)
                    output = interp(idx_with_latency)
                
            result[key] = output

        # repeat frame before first grasp
        if self.repeat_frame_prob != 0.0:
            if before_first_grasp and random.random() < self.repeat_frame_prob:
                for key in obs_keys:
                    result[key][:-1] = result[key][-1:]
        # Add image timestamps
        if 'img_timestamps' in self.replay_buffer:
            cam_key = self.rgb_keys[0]
            img_horizon = self.key_horizon[cam_key]
            img_down_sample_steps = self.key_down_sample_steps[cam_key]
            
            # Select two frames before and after the current index
            if current_idx == start_idx:
                next_idx = min(current_idx + img_down_sample_steps, end_idx-1)
                img_indices = [current_idx, next_idx]
            else:
                prev_idx = max(current_idx - img_down_sample_steps, start_idx)
                img_indices = [prev_idx, current_idx]

            img_ts = self.replay_buffer['img_timestamps'][img_indices]
            result['img_timestamps'] = img_ts


        if self.has_ft_data and 'ft_data' in self.replay_buffer and 'ft_timestamps' in self.replay_buffer:
            if 'img_timestamps' in result:
                img_ts = result['img_timestamps']
                
                # Calculate episode boundaries (once)
                ft_episode_start = 0
                ft_episode_end = len(self.replay_buffer['ft_timestamps'])
                
                if hasattr(self.original_replay_buffer, 'ft_episode_ends') and self.original_replay_buffer.ft_episode_ends is not None:
                    ft_episode_start = 0 if episode_idx == 0 else self.original_replay_buffer.ft_episode_ends[episode_idx-1]
                    ft_episode_end = self.original_replay_buffer.ft_episode_ends[episode_idx]
                elif hasattr(self.original_replay_buffer, 'meta') and 'episode_ft_ends' in self.original_replay_buffer.meta:
                    ft_episode_start = 0 if episode_idx == 0 else self.original_replay_buffer.meta['episode_ft_ends'][episode_idx-1]
                    ft_episode_end = self.original_replay_buffer.meta['episode_ft_ends'][episode_idx]
                else:
                    print(f"[Warning] FT episode boundary info missing. Searching entire FT dataset.")
                
                # Slice per-episode FT data (once)
                episode_ft_data = self.replay_buffer['ft_data'][ft_episode_start:ft_episode_end]
                episode_ft_timestamps = self.replay_buffer['ft_timestamps'][ft_episode_start:ft_episode_end]
                
                try:
                    # Timestamp conversion (once)
                    img_ts_int = np.array([int(ts) for ts in img_ts], dtype=np.int64)
                    ft_ts_int = np.array([int(ts) for ts in episode_ft_timestamps], dtype=np.int64)
                    
                    # Find index with searchsorted (once)
                    left_idx = np.searchsorted(ft_ts_int, img_ts_int[0], side='left')
                    right_idx = np.searchsorted(ft_ts_int, img_ts_int[1], side='right')
                    
                    # Verify and adjust boundaries
                    left_idx = max(0, left_idx)
                    right_idx = min(len(ft_ts_int), right_idx)
                    
                    # FT data slicing
                    ft_slice = episode_ft_data[left_idx:right_idx]
                    ft_ts_slice = episode_ft_timestamps[left_idx:right_idx]
                    n = len(ft_slice)
                    
                    # Normalize frame count (one-time processing)
                    if n < self.target_ft_frames:
                        if n == 0:
                            # If completely empty: create target_ft_frames with zero vectors
                            # print(f"[WARNING] FT data empty in episode {episode_idx}. Replacing with zero vectors.")
                            ft_slice = np.zeros((self.target_ft_frames, 6), dtype=np.float32)
                            ft_ts_slice = np.zeros(self.target_ft_frames, dtype=ft_ts_slice.dtype if len(ft_ts_slice) > 0 else np.float64)
                        else:
                            # If some data exists: pad with last frame
                            pad_n = self.target_ft_frames - n
                            last_frame = ft_slice[-1:].repeat(pad_n, axis=0)  # use .repeat() instead of np.repeat
                            last_ts = np.full(pad_n, ft_ts_slice[-1])  # use np.full instead of np.repeat
                            
                            # concatenate at once
                            ft_slice = np.vstack([ft_slice, last_frame])
                            ft_ts_slice = np.concatenate([ft_ts_slice, last_ts])
                    
                    elif n > self.target_ft_frames:
                        # Downsampling: create indices with linspace then slice at once
                        idxs = np.linspace(0, n-1, self.target_ft_frames, dtype=np.int32)  # Explicit dtype for memory efficiency
                        ft_slice = ft_slice[idxs]
                        ft_ts_slice = ft_ts_slice[idxs]
                    
                    # Result validation and storage
                    assert ft_slice.shape[0] == self.target_ft_frames, f"FT frame count mismatch: {ft_slice.shape[0]} != {self.target_ft_frames}"
                    
                    result['ft_data'] = ft_slice
                    result['ft_timestamps'] = ft_ts_slice
                    
                except (ValueError, TypeError) as e:
                    print(f"Timestamp conversion error: {e}")
                    # Set default values on error
                    result['ft_data'] = np.zeros((self.target_ft_frames, 6), dtype=np.float32)
                    result['ft_timestamps'] = np.zeros(self.target_ft_frames, dtype=np.float64)
        
        
        # action
        input_arr = self.replay_buffer['action']
        action_horizon = self.key_horizon['action']
        action_latency_steps = self.key_latency_steps['action']
        assert action_latency_steps == 0
        action_down_sample_steps = self.key_down_sample_steps['action']
        slice_end = min(end_idx, current_idx + (action_horizon - 1) * action_down_sample_steps + 1)
        output = input_arr[current_idx: slice_end: action_down_sample_steps]
        
        # Padding logic
        if output.shape[0] < action_horizon:
            padding = np.repeat(output[-1:], action_horizon - output.shape[0], axis=0)
            output = np.concatenate([output, padding], axis=0)
        elif output.shape[0] > action_horizon:
            output = output[:action_horizon]
        
        result['action'] = output

        return result
    
    def ignore_rgb(self, apply=True):
        self.ignore_rgb_is_applied = apply
        