
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import os
import json
import zarr
import numpy as np
import pandas as pd
from numcodecs import Blosc
from PIL import Image
import argparse
import datetime
from scipy.spatial.transform import Rotation as R
import cv2
from concurrent.futures import ThreadPoolExecutor
import time



def compensate_gripping_force_per_episode(all_episodes_ft_data, start_idx=5, end_idx=100):
    """
    Compute per-episode gripping force bias and subtract it.
    Bias is estimated from samples [start_idx, end_idx] of each episode.

    Args:
        all_episodes_ft_data: List of (ft_data, episode_info) tuples.
        start_idx: Start index for bias calculation (default: 5).
        end_idx:   End index for bias calculation (default: 100).

    Returns:
        compensated_ft_data_list: Bias-corrected FT data list.
        episode_biases: Per-episode bias vectors.
    """
    print(f"\n🔧 Per-Episode Gripping Force Compensation")
    print(f"   Bias calculation: samples {start_idx}~{end_idx} per episode")
    
    compensated_ft_data_list = []
    episode_biases = []
    
    for ft_data, episode_info in all_episodes_ft_data:
        if ft_data is None:
            compensated_ft_data_list.append((None, episode_info))
            episode_biases.append(None)
            continue
        
        # Extract initial segment for this episode
        actual_end_idx = min(end_idx, len(ft_data))
        actual_start_idx = min(start_idx, actual_end_idx - 5)
        
        if actual_end_idx > actual_start_idx:
            initial_ft = ft_data[actual_start_idx:actual_end_idx]
            episode_bias = np.mean(initial_ft, axis=0)
            
            # Apply bias to this episode
            compensated_ft = ft_data - episode_bias
            compensated_ft_data_list.append((compensated_ft, episode_info))
            episode_biases.append(episode_bias)
            
            print(f"   Episode {episode_info['name']}: bias={episode_bias[:3]}, force_mag={np.linalg.norm(episode_bias[:3]):.4f}")
        else:
            # Not enough initial data — skip compensation
            compensated_ft_data_list.append((ft_data, episode_info))
            episode_biases.append(None)
            print(f"   Episode {episode_info['name']}: insufficient initial data, no compensation")
    
    print(f"✅ Per-episode compensation applied to {len([b for b in episode_biases if b is not None])} episodes\n")
    return compensated_ft_data_list, episode_biases

def load_image_fast(img_path):
    """Fast image loading using OpenCV (faster than PIL). Returns RGB numpy array."""
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR -> RGB
    return img

def load_images_batch_parallel(image_paths, batch_size=32, max_workers=8):
    """Load images in parallel batches using ThreadPoolExecutor."""
    if not image_paths:
        return []
    
    print(f"    Loading {len(image_paths)} images in batches of {batch_size} with {max_workers} workers...")
    start_time = time.time()
    
    all_batches = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch_start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[batch_start:batch_start + batch_size]
            
            # Load images in parallel
            batch_imgs = list(executor.map(load_image_fast, batch_paths))
            batch_array = np.stack(batch_imgs, axis=0)
            all_batches.append(batch_array)
            
            if (batch_start // batch_size + 1) % 10 == 0:
                elapsed = time.time() - start_time
                processed = batch_start + len(batch_paths)
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"    Processed {processed}/{len(image_paths)} images ({rate:.1f} img/s)")
    
    total_time = time.time() - start_time
    print(f"    ✅ Loaded {len(image_paths)} images in {total_time:.1f}s ({len(image_paths)/total_time:.1f} img/s)")
    
    return all_batches

def extract_timestamp_from_filename(filename):
    """
    Extract a datetime object from an image filename.
    Example: '20250504_173116_860.jpg' -> datetime object.
    """
    base_name = os.path.splitext(filename)[0]
    parts = base_name.split('_')
    if len(parts) != 3:
        raise ValueError(f"Unexpected filename format: {filename}")
    
    date_str = parts[0]
    time_str = parts[1]
    millisec_str = parts[2]
    
    # Create datetime object
    year = int(date_str[0:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])
    hour = int(time_str[0:2])
    minute = int(time_str[2:4])
    second = int(time_str[4:6])
    millisec = int(millisec_str)
    
    dt = datetime.datetime(year, month, day, hour, minute, second, millisec * 1000)
    return dt

def read_ft_data(episode_dir, episode_number, regenerate_timestamps=False):
    """
    Read F/T data from an episode directory.
    
    Args:
        episode_dir (str): Path to the episode directory.
        episode_number (int): Episode number (for logging).
        regenerate_timestamps (bool): If True, regenerate timestamps at fixed 200 Hz intervals.
        
    Returns:
        tuple: (ft_data array, timestamps array)
    """
    import glob
    ft_pattern = os.path.join(episode_dir, "ft_data_episode_*.csv")
    ft_files = glob.glob(ft_pattern)
    
    if not ft_files:
        print(f"Warning: No FT data file found in {episode_dir} (pattern: ft_data_episode_*.csv)")
        return None, None
    
    # If multiple files exist, use the first one
    ft_csv_path = ft_files[0]
    if len(ft_files) > 1:
        print(f"Warning: Multiple FT files found in {episode_dir}, using {os.path.basename(ft_csv_path)}")
    
    print(f"Reading FT data from: {os.path.basename(ft_csv_path)}")
    
    try:
        df = pd.read_csv(ft_csv_path)
        if 'timestamp' not in df.columns or regenerate_timestamps:
            # Regenerate timestamps at fixed 200 Hz (5 ms intervals)
            print(f"    Regenerating timestamps for episode {episode_number} (200Hz = 5ms intervals)")
            timestamps = np.arange(len(df)) * 5.0  # 200Hz = 5ms interval
        else:
            if isinstance(df['timestamp'].iloc[0], str):
                # String timestamps: strip underscores and convert to float64 for precision
                timestamps = np.array([
                    float(ts.replace('_', '')) 
                    for ts in df['timestamp'].values
                ], dtype=np.float64)  # Store as float64 for precision
            else:
                timestamps = df['timestamp'].values.astype(np.float64)

        # Locate FT data columns (force_* and torque_*), excluding timestamp
        ft_columns = [col for col in df.columns if col.startswith('force_') or col.startswith('torque_')]
        if len(ft_columns) == 0:
            # Fallback: use numeric columns excluding timestamp
            numeric_cols = df.select_dtypes(include=np.number).columns
            if 'timestamp' in numeric_cols:
                numeric_cols = [col for col in numeric_cols if col != 'timestamp']
            ft_columns = numeric_cols[:6]
        
        if len(ft_columns) < 6:
            print(f"Warning: Found fewer than 6 FT columns in {ft_csv_path}: {ft_columns}")
            # Pad missing columns with zeros
            ft_data = np.zeros((len(df), 6), dtype=np.float32)
            for i, col in enumerate(ft_columns):
                ft_data[:, i] = df[col].values
        else:
            ft_data = df[ft_columns].values.astype(np.float32)
        
        return ft_data, timestamps
    except Exception as e:
        print(f"Error reading FT data from {ft_csv_path}: {e}")
        return None, None

def read_episode_data(episode_dir, regenerate_ft_timestamps=False):
    """Read all data from an episode directory (memory-efficient)."""
    final_pose_path = os.path.join(episode_dir, "pose_data.json")
    if not os.path.isfile(final_pose_path):
        print(f"Warning: Missing pose_data.json in {episode_dir}")
        return None, None, None, None, None, None, None, None
    
    episode_number = os.path.basename(episode_dir).split('_')[-1]

    ft_data, ft_timestamps = read_ft_data(episode_dir, episode_number, regenerate_timestamps=regenerate_ft_timestamps)
    
    with open(final_pose_path, 'r') as f:
        final_pose_data = json.load(f)
    
    # Index pose data by filename for O(1) lookup
    pose_dict = {}
    for pose_entry in final_pose_data:
        pose_dict[pose_entry['image_file']] = pose_entry
    
    # Camera directories
    cam1_dir = os.path.join(episode_dir, "images/handeye")
    cam2_dir = os.path.join(episode_dir, "images/additional_cam")
    
    # Camera 1 image files
    cam1_images = []
    if os.path.isdir(cam1_dir):
        cam1_images = sorted([f for f in os.listdir(cam1_dir) if f.endswith(('.jpg', '.png'))])
    
    # Camera 2 image files
    cam2_images = []
    if os.path.isdir(cam2_dir):
        cam2_images = sorted([f for f in os.listdir(cam2_dir) if f.endswith(('.jpg', '.png'))])
    
    if not cam1_images:
        print(f"Warning: No images found in {cam1_dir}")
        return None, None, None, None, None, None, None, None
    
    # Collect image info first, defer actual image loading
    image_info = []
    for idx, image_name in enumerate(cam1_images):
        image_path = os.path.join(cam1_dir, image_name)
        
        # Extract timestamp
        try:
            base_name = os.path.splitext(image_name)[0]
            numeric_timestamp = int(base_name.replace('_', ''))
        except Exception as e:
            print(f"Error extracting timestamp from {image_name}: {e}")
            numeric_timestamp = idx
        
        # Check for pose data
        has_pose = image_name in pose_dict
        
        image_info.append({
            'name': image_name,
            'path': image_path,
            'timestamp': numeric_timestamp,
            'has_pose': has_pose,
            'index': idx
        })
    
    # Process pose data without loading images
    actions, states = [], []
    image_timestamps = []
    last_action = None
    
    for img_info in image_info:
        image_name = img_info['name']
        image_timestamps.append(img_info['timestamp'])
        
        # Pose data (exists for all images)
        pose_entry = pose_dict[image_name]
        
        position = pose_entry["state"]["position"]
        orientation = pose_entry["state"]["orientation"]
        states.append(np.concatenate([position, orientation]))

        # Action data (includes gripper state)
        if "action" in pose_entry and pose_entry["action"] is not None:
            rel_position = pose_entry["action"]["relative_position"]
            rel_orientation = pose_entry["action"]["relative_orientation"]
            
            gripper_state = 1.0  # default: open
            if "pose" in pose_entry and "gripper_state" in pose_entry["pose"]:
                gripper_state = float(pose_entry["pose"]["gripper_state"])
            
            # 7D action + 1D gripper_state = 8D total
            action = np.concatenate([rel_position, rel_orientation, [gripper_state]])
            actions.append(action)
            last_action = action
        else:
            if last_action is not None:
                actions.append(last_action)
            else:
                # No action for this frame (first frame edge case)
                actions.append(np.concatenate([np.zeros(7), [1.0]]))
    
    # Convert to arrays
    actions_array = np.array(actions, dtype=np.float32)
    states_array = np.array(states, dtype=np.float32)
    img_timestamps_array = np.array(image_timestamps, dtype=np.float64)
    
    # Compute pose relative to the start frame (vectorized where possible)
    if states_array is not None and len(states_array) > 0:
        start_pos = states_array[0, :3]  # Starting position
        start_ori = states_array[0, 3:]  # Starting orientation (quaternion)
        
        # Create starting rotation object once
        start_rot = R.from_quat(start_ori)
        
        poses_wrt_start = np.zeros((len(states_array), 7), dtype=np.float32)
        
        # Optimized with vectorized calculation
        poses_wrt_start[:, :3] = states_array[:, :3] - start_pos  # Relative position
        
        # Rotation calculation (loop as it cannot be fully vectorized, but optimized)
        for i, state in enumerate(states_array):
            curr_ori = state[3:]
            curr_rot = R.from_quat(curr_ori)
            rel_rot = (start_rot.inv() * curr_rot).as_quat()
            poses_wrt_start[i, 3:] = rel_rot
    else:
        poses_wrt_start = None

    # Return image paths only (load later to save memory)
    cam1_image_paths = [img_info['path'] for img_info in image_info]
    cam2_image_paths = []
    if cam2_images:
        frames_to_load = min(len(cam1_image_paths), len(cam2_images))
        for i in range(frames_to_load):
            cam2_image_paths.append(os.path.join(cam2_dir, cam2_images[i]))

    return cam1_image_paths, cam2_image_paths, actions_array, states_array, ft_data, img_timestamps_array, ft_timestamps, poses_wrt_start

def save_to_zarr(data_root, output_path, ema_normalize=True, compensate_gripping_force=False, gripping_start_idx=10, gripping_end_idx=1000, batch_size=32, max_workers=8, compression_level=3, regenerate_ft_timestamps=False):
    """
    Save all episode data to a single Zarr file.
    Includes images from both cameras, actions, states, F/T data, and timestamps.
    
    Args:
        data_root (str): Root directory containing episode subdirectories.
        output_path (str): Path for the output .zarr file.
        ema_normalize (bool): Apply EMA normalization to FT data (default: True).
        compensate_gripping_force (bool): Subtract per-episode gripping force bias.
        gripping_start_idx (int): Start index for gripping bias window.
        gripping_end_idx (int): End index for gripping bias window.
        batch_size (int): Batch size for parallel image loading.
        max_workers (int): Number of parallel workers for image loading.
        compression_level (int): Zarr compression level (1-5).
        regenerate_ft_timestamps (bool): If True, regenerate FT timestamps.
    """
    data_root = os.path.abspath(data_root)
    output_path = os.path.abspath(output_path)

    print(f"Safety checks:")
    print(f"   Input data path: {data_root}")
    print(f"   Output zarr path: {output_path}")
    
    if not output_path.endswith('.zarr'):
        raise ValueError(f"❌ Output path must end with '.zarr' extension. Got: {output_path}")
    
    if output_path == data_root:
        raise ValueError(f"❌ Output path cannot be the same as input data path. This would overwrite your source data!")
    
    if output_path.startswith(data_root + os.sep):
        raise ValueError(f"❌ Output path cannot be inside the input data directory. This could cause conflicts!")
    
    if not os.path.exists(data_root):
        raise ValueError(f"❌ Input data directory does not exist: {data_root}")
    
    if os.path.exists(output_path):
        print(f"Warning: Output path already exists and will be overwritten: {output_path}")
        if os.path.isdir(output_path):
            try:
                files_count = len(os.listdir(output_path))
                print(f"   Existing directory contains {files_count} items")
            except:
                pass
    
    print(f"✅ Safety checks passed\n")
    
    # Create Zarr groups
    store = zarr.DirectoryStore(output_path)
    root = zarr.group(store, overwrite=True)
    
    # Create data groups
    data_group = root.require_group('data')
    meta_group = root.require_group('meta')
    
    # Episode directories list
    episode_dirs = sorted(
        [os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    )
    
    if not episode_dirs:
        raise ValueError("No episode directories found")
    
    all_episode_data = []
    episode_biases = None
    
    if compensate_gripping_force:
        print(f"🔧 Gripping force compensation enabled")
        print(f"   Bias calculation: samples {gripping_start_idx}~{gripping_end_idx}")
        
        # Load all episodes and collect FT data
        all_episodes_ft_data = []
        
        for i, episode_dir in enumerate(episode_dirs):
            print(f"Loading episode {i+1}/{len(episode_dirs)}: {os.path.basename(episode_dir)}")
            episode_data = read_episode_data(episode_dir, regenerate_ft_timestamps=regenerate_ft_timestamps)
            all_episode_data.append((episode_data, episode_dir))
            
            # Extract FT data
            cam1_image_paths, cam2_image_paths, actions, states, ft_data, img_timestamps, ft_timestamps, poses_wrt_start = episode_data
            if ft_data is not None:
                ft_data_feats = ft_data  # Includes all 6 axes (force_x, force_y, force_z, torque_x, torque_y, torque_z)
                episode_info = {'name': os.path.basename(episode_dir)}
                all_episodes_ft_data.append((ft_data_feats, episode_info))
            else:
                all_episodes_ft_data.append((None, {'name': os.path.basename(episode_dir)}))

        # Compute and apply per-episode biases
        compensated_ft_data_list, episode_biases = compensate_gripping_force_per_episode(
            all_episodes_ft_data, 
            start_idx=gripping_start_idx, 
            end_idx=gripping_end_idx
        )
        
        # Apply compensated FT data to the original data
        for i, ((ft_data_compensated, _), (episode_data, episode_dir)) in enumerate(zip(compensated_ft_data_list, all_episode_data)):
            cam1_image_paths, cam2_image_paths, actions, states, original_ft_data, img_timestamps, ft_timestamps, poses_wrt_start = episode_data
            if ft_data_compensated is not None:
                # Apply compensation to all 6 axes
                compensated_full_ft = ft_data_compensated
                all_episode_data[i] = ((cam1_image_paths, cam2_image_paths, actions, states, compensated_full_ft, img_timestamps, ft_timestamps, poses_wrt_start), episode_dir)
    else:
        # Standard load (no compensation)
        for i, episode_dir in enumerate(episode_dirs):
            print(f"Loading episode {i+1}/{len(episode_dirs)}: {os.path.basename(episode_dir)}")
            episode_data = read_episode_data(episode_dir, regenerate_ft_timestamps=regenerate_ft_timestamps)
            all_episode_data.append((episode_data, episode_dir))
    
    # Get dimension information from a sample episode
    sample_episode_data, _ = all_episode_data[0]
    sample_cam1_paths, sample_cam2_paths, _, _, sample_ft, _, _, _ = sample_episode_data
    if sample_cam1_paths is None or len(sample_cam1_paths) == 0:
        raise ValueError("Could not read sample episode data")
    
    # Load the first image to get dimension information
    sample_img = load_image_fast(sample_cam1_paths[0])
    height, width, channels = sample_img.shape
    
    # Check sample FT data dimension
    ft_dim = 6  # default: 6 axes (fx fy fz tx ty tz)
    
    # Initialize dataset - Camera 1 images (lower compression level for speed)
    cam1_dataset = data_group.require_dataset(
        "handeye_cam_1",
        shape=(0, height, width, channels),
        chunks=(batch_size, height, width, channels),  # Align with batch size
        dtype=np.uint8,
        compressor=Blosc(cname='zstd', clevel=compression_level)  # Set compression level
    )
    
    # Camera 2 dataset (if available)
    if sample_cam2_paths and len(sample_cam2_paths) > 0:
        # Load the first image to get dimension information
        sample_cam2_img = load_image_fast(sample_cam2_paths[0])
        cam2_height, cam2_width, cam2_channels = sample_cam2_img.shape
        cam2_dataset = data_group.require_dataset(
            "handeye_cam_2",
            shape=(0, cam2_height, cam2_width, cam2_channels),
            chunks=(batch_size, cam2_height, cam2_width, cam2_channels),  # Align with batch size
            dtype=np.uint8,
            compressor=Blosc(cname='zstd', clevel=compression_level)  # Set compression level
        )
    else:
        cam2_dataset = None
    
    action_dataset = data_group.require_dataset(
        "action",
        shape=(0, 8),  # 7D + 1D gripper_state = 8D
        chunks=(10, 8),
        dtype=np.float32,
        compressor=Blosc(cname='zstd', clevel=5)
    )
    
    # State dataset (position + orientation)
    state_dataset = data_group.require_dataset(
        "state",
        shape=(0, 7),  # position(3) + orientation(4)
        chunks=(10, 7),
        dtype=np.float32,
        compressor=Blosc(cname='zstd', clevel=5)
    )

    # Pose-wrt-start dataset
    pose_wrt_start_dataset = data_group.require_dataset(
        "pose_wrt_start",
        shape=(0, 7),  # relative_position(3) + relative_orientation(4)
        chunks=(10, 7),
        dtype=np.float32,
        compressor=Blosc(cname='zstd', clevel=5)
    )
    
    # F/T dataset
    ft_dataset = data_group.require_dataset(
        "ft_data",
        shape=(0, ft_dim),
        chunks=(100, ft_dim),
        dtype=np.float32,
        compressor=Blosc(cname='zstd', clevel=5)
    )
    
    img_timestamps_dataset = data_group.require_dataset(
        "img_timestamps",
        shape=(0,),
        chunks=(100,),
        dtype=np.float64,
        compressor=Blosc(cname='zstd', clevel=5)
    )
    
    ft_timestamps_dataset = data_group.require_dataset(
        "ft_timestamps",
        shape=(0,),
        chunks=(100,),
        dtype=np.float64,
        compressor=Blosc(cname='zstd', clevel=5)
    )
    
    # Episode boundary arrays
    episode_ends = meta_group.zeros("episode_ends", shape=(0,), dtype=np.int64)
    episode_img_ends = meta_group.zeros("episode_img_ends", shape=(0,), dtype=np.int64)
    episode_ft_ends = meta_group.zeros("episode_ft_ends", shape=(0,), dtype=np.int64)
    
    # Initialize total steps
    total_img_steps = 0
    total_ft_steps = 0
    
    # Process each episode - memory-efficient version
    for i, (episode_data, episode_dir) in enumerate(all_episode_data):
        print(f"Processing episode {i+1}/{len(all_episode_data)}: {os.path.basename(episode_dir)}")
        
        cam1_image_paths, cam2_image_paths, actions, states, ft_data, img_timestamps, ft_timestamps, poses_wrt_start = episode_data
        
        if cam1_image_paths is None:
            print(f"Skipping {episode_dir} due to missing data.")
            continue
        
        print(f"  - Camera 1 images: {len(cam1_image_paths)}, Timestamps: {img_timestamps.shape}")
        if cam2_image_paths:
            print(f"  - Camera 2 images: {len(cam2_image_paths)}")
        
        # Save actions, states, and timestamps (no image loading yet)
        action_dataset.append(actions)
        state_dataset.append(states)
        if poses_wrt_start is not None:
            pose_wrt_start_dataset.append(poses_wrt_start)
        img_timestamps_dataset.append(img_timestamps)
        
        # Camera 1: load and save in parallel batches
        print(f"  - Loading camera 1 images...")
        cam1_batches = load_images_batch_parallel(cam1_image_paths, batch_size=batch_size, max_workers=max_workers)
        for batch in cam1_batches:
            cam1_dataset.append(batch)
        
        # Camera 2: load and save in parallel batches (if available)
        if cam2_image_paths and cam2_dataset is not None:
            print(f"  - Loading camera 2 images...")
            cam2_batches = load_images_batch_parallel(cam2_image_paths, batch_size=batch_size, max_workers=max_workers)
            for batch in cam2_batches:
                cam2_dataset.append(batch)
        
        # Update image episode boundary
        total_img_steps += len(cam1_image_paths)
        episode_img_ends.resize(episode_img_ends.shape[0] + 1)
        episode_img_ends[-1] = total_img_steps
        episode_ends.resize(episode_ends.shape[0] + 1)
        episode_ends[-1] = total_img_steps
        
            # FT 데이터가 있는 경우에만 저장
        if ft_data is not None and ft_timestamps is not None:
            print(f"  - FT data: {ft_data.shape}, Timestamps: {len(ft_timestamps)}")
            
            # Trim/pad FT to exactly 6 columns
            if ft_data.shape[1] >= 6:
                ft_to_save = ft_data[:, :6]  # 첫 6개 열만 사용
            else:
                # Pad with zeros if fewer than 6 columns
                ft_to_save = np.zeros((ft_data.shape[0], 6), dtype=np.float32)
                ft_to_save[:, :ft_data.shape[1]] = ft_data
            
            print(f"  - FT data saved: {ft_to_save.shape}")
            
            ft_dataset.append(ft_to_save)
            ft_timestamps_dataset.append(ft_timestamps)

            # Update F/T episode boundary
            total_ft_steps += len(ft_data)
            episode_ft_ends.resize(episode_ft_ends.shape[0] + 1)
            episode_ft_ends[-1] = total_ft_steps
            
        else:
            print(f"  - No FT data available for this episode")
            # No FT data: extend boundary array with previous value
            episode_ft_ends.resize(episode_ft_ends.shape[0] + 1)
            episode_ft_ends[-1] = total_ft_steps
    
    # Save metadata
    meta_group.attrs['total_img_steps'] = total_img_steps
    meta_group.attrs['total_ft_steps'] = total_ft_steps
    meta_group.attrs['episodes'] = len(episode_dirs)
    
    # Save episode gripping biases if computed
    if episode_biases is not None and any(b is not None for b in episode_biases):
        meta_group.attrs['episode_gripping_biases'] = [b.tolist() if b is not None else None for b in episode_biases]
        print(f"✅ Per-episode gripping biases saved to metadata")
    
    print(f"Saved dataset to {output_path}")
    print(f"Total image steps: {total_img_steps}, Total FT steps: {total_ft_steps}")
    print(f"Episodes: {len(episode_dirs)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert data to Zarr format with FT data and timestamps',
        epilog='⚠️  IMPORTANT: Output path MUST end with .zarr extension to prevent overwriting source data!'
    )
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to the data directory containing episode folders')
    parser.add_argument('--output_path', type=str, required=True,
                      help='Path for the output Zarr file (MUST end with .zarr, e.g., data/output_dataset.zarr)')
    parser.add_argument('--no-ema-normalize', action='store_true',
                      help='Disable EMA normalization for FT data (use raw FT values)')
    # Gripping force compensation arguments
    parser.add_argument('--disable-gripping-compensation', action='store_true',
                      help='Disable gripping force bias compensation (enabled by default)')
    parser.add_argument('--gripping-start-idx', type=int, default=5,
                      help='Start index for gripping bias calculation (default: 5)')
    parser.add_argument('--gripping-end-idx', type=int, default=100,
                      help='End index for gripping bias calculation (default: 100)')
    # Performance tuning arguments
    parser.add_argument('--batch-size', type=int, default=32,
                      help='Batch size for image loading (default: 32)')
    parser.add_argument('--max-workers', type=int, default=8,
                      help='Number of parallel workers for image loading (default: 8)')
    parser.add_argument('--compression-level', type=int, default=3, choices=[1, 2, 3, 4, 5],
                      help='Zarr compression level (1=fast, 5=small, default: 3)')
    parser.add_argument('--regenerate-ft-timestamps', action='store_true',
                      help='Regenerate FT timestamps with regular 5ms intervals (fixes timestamp anomalies)')
    args = parser.parse_args()

    print("="*60)
    print("Starting Zarr conversion process")
    print("="*60)

    # Pre-validate output path before calling save_to_zarr
    if not args.output_path.endswith('.zarr'):
        print(f"❌ ERROR: Output path must end with '.zarr' extension!")
        print(f"   You provided: {args.output_path}")
        print(f"   Correct example: {args.output_path}.zarr")
        print(f"   This prevents accidentally overwriting your source data directory.")
        exit(1)
    
    # Pre-validate that input and output are not the same path
    input_abs = os.path.abspath(args.data_path)
    output_abs = os.path.abspath(args.output_path)
    if output_abs == input_abs:
        print(f"❌ ERROR: Output path cannot be the same as input path!")
        print(f"   Input:  {input_abs}")
        print(f"   Output: {output_abs}")
        print(f"   Please choose a different output path (e.g., {input_abs}.zarr).")
        exit(1)

    # EMA normalize option
    ema_normalize = not args.no_ema_normalize

    # Gripping force compensation option
    compensate_gripping_force = not args.disable_gripping_compensation
    
    print(f"🎛️ EMA Normalize: {'Enabled' if ema_normalize else 'Disabled (Raw FT data)'}")
    print(f"🔧 Gripping Force Compensation: {'Enabled' if compensate_gripping_force else 'Disabled'}")
    if compensate_gripping_force:
        print(f"   - Bias calculation range: indices {args.gripping_start_idx}:{args.gripping_end_idx}")
    print(f"🕐 FT Timestamp Regeneration: {'Enabled' if args.regenerate_ft_timestamps else 'Disabled'}")
    if args.regenerate_ft_timestamps:
        print(f"   - Will regenerate timestamps with regular 5ms intervals (200Hz)")
    print(f"⚡ Optimization settings:")
    print(f"   - Batch size: {args.batch_size}")
    print(f"   - Max workers: {args.max_workers}")
    print(f"   - Compression level: {args.compression_level}")

    # Run conversion
    try:
        save_to_zarr(
            data_root=args.data_path, 
            output_path=args.output_path, 
            ema_normalize=ema_normalize,
            compensate_gripping_force=compensate_gripping_force,
            gripping_start_idx=args.gripping_start_idx,
            gripping_end_idx=args.gripping_end_idx,
            batch_size=args.batch_size,
            max_workers=args.max_workers,
            compression_level=args.compression_level,
            regenerate_ft_timestamps=args.regenerate_ft_timestamps
        )
        print("\n" + "="*60)
        print("✅ Zarr conversion completed successfully!")
        print("="*60)
    except Exception as e:
        print(f"\n❌ ERROR during conversion: {e}")
        exit(1)