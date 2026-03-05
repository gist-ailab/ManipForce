
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import os
import pandas as pd
import numpy as np
import shutil
import argparse
from tqdm import tqdm
from datetime import datetime

def parse_ts_to_float(ts_str):
    """Convert a timestamp string '20260209_152216_409' (ms) to a comparable float."""
    try:
        parts = ts_str.split('_')
        if len(parts) == 3:
            # Pad 3-digit ms to 6-digit microseconds to avoid 1000x precision loss
            ms_part = parts[2].ljust(6, '0')
            ts_str_fixed = f"{parts[0]}_{parts[1]}_{ms_part}"
            dt = datetime.strptime(ts_str_fixed, "%Y%m%d_%H%M%S_%f")
            return dt.timestamp()
        return 0.0
    except Exception as e:
        return 0.0

def align_episode(episode_path, dji_latency=0.0):
    print(f"\n📂 Processing episode: {episode_path} (DJI latency offset: {dji_latency}s)")
    
    images_base = os.path.join(episode_path, "images")
    raw_backup = os.path.join(images_base, "raw_backup")
    
    # 1. Back up raw data and clean destination folders
    if not os.path.exists(raw_backup):
        print(f"Backing up raw data to {raw_backup}...")
        os.makedirs(raw_backup, exist_ok=True)
        for folder in ["handeye", "pose_tracking", "additional_cam"]:
            src_folder = os.path.join(images_base, folder)
            if os.path.exists(src_folder):
                shutil.move(src_folder, os.path.join(raw_backup, folder))
    
    for folder in ["handeye", "pose_tracking", "additional_cam"]:
        os.makedirs(os.path.join(images_base, folder), exist_ok=True)

    # 2. Load frame lists and timestamps per camera
    def get_frame_info(folder_name):
        folder_path = os.path.join(raw_backup, folder_name)
        if not os.path.exists(folder_path):
            return [], []
        files = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
        ts_strs = [os.path.splitext(f)[0] for f in files]
        ts_floats = np.array([parse_ts_to_float(s) for s in ts_strs])
        return ts_strs, ts_floats

    h_strs, h_floats = get_frame_info("handeye")
    p_strs, p_floats = get_frame_info("pose_tracking")
    a_strs, a_floats = get_frame_info("additional_cam")

    if len(h_strs) == 0:
        print("[ERROR] No handeye images found.")
        return

    print(f"Loaded: Handeye({len(h_strs)}), DJI({len(p_strs)}), Additional({len(a_strs)})")
    print(f"Starting nearest-neighbor matching...")

    # 3. Match each handeye frame to the closest DJI/additional frame
    for i in tqdm(range(len(h_strs))):
        ref_ts_str = h_strs[i]
        ref_time = h_floats[i]
        
        # 3-1. Handeye (master)
        src_h = os.path.join(raw_backup, "handeye", f"{ref_ts_str}.jpg")
        dst_h = os.path.join(images_base, "handeye", f"{ref_ts_str}.jpg")
        if os.path.exists(src_h):
            shutil.copy2(src_h, dst_h)
            
        # 3-2. DJI (Pose Tracking) — compensate for hardware encoding latency
        if len(p_floats) > 0:
            # DJI frames arrive later due to hardware delay, so we look ahead by dji_latency
            target_p_time = ref_time + dji_latency
            p_idx = np.argmin(np.abs(p_floats - target_p_time))
            match_p_ts = p_strs[p_idx]
            src_p = os.path.join(raw_backup, "pose_tracking", f"{match_p_ts}.jpg")
            dst_p = os.path.join(images_base, "pose_tracking", f"{ref_ts_str}.jpg")
            shutil.copy2(src_p, dst_p)
            
        # 3-3. Additional cam — nearest-neighbor match
        if len(a_floats) > 0:
            a_idx = np.argmin(np.abs(a_floats - ref_time))
            match_a_ts = a_strs[a_idx]
            src_a = os.path.join(raw_backup, "additional_cam", f"{match_a_ts}.jpg")
            dst_a = os.path.join(images_base, "additional_cam", f"{ref_ts_str}.jpg")
            shutil.copy2(src_a, dst_a)

    print(f"✅ Alignment complete (latency={dji_latency}s)!")
    
    # 4. Remove raw backup to free disk space
    if os.path.exists(raw_backup):
        shutil.rmtree(raw_backup)
        print(f"🗑️ Raw backup deleted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to episode folder")
    parser.add_argument("--episode", type=int, default=None, help="Specific episode number to process")
    parser.add_argument("--dji_latency", type=float, default=0.133, help="DJI hardware/decoding latency in seconds (4 frames = 0.133s)")
    args = parser.parse_args()

    if args.episode is not None:
        align_episode(os.path.join(args.data_path, f"episode_{args.episode}"), args.dji_latency)
    else:
        ep_dirs = [d for d in os.listdir(args.data_path) if d.startswith("episode_")]
        for ep_dir in sorted(ep_dirs, key=lambda x: int(x.split('_')[1])):
            align_episode(os.path.join(args.data_path, ep_dir), args.dji_latency)