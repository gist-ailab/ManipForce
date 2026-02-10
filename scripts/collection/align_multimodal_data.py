
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
    """'20260209_152216_409' (ms)를 비교 가능한 float (timestamp)로 변환"""
    try:
        parts = ts_str.split('_')
        if len(parts) == 3:
            # 3자리 밀리초(409)를 6자리 마이크로초(409000)로 패딩하여 1000배 오차 방지
            ms_part = parts[2].ljust(6, '0')
            ts_str_fixed = f"{parts[0]}_{parts[1]}_{ms_part}"
            dt = datetime.strptime(ts_str_fixed, "%Y%m%d_%H%M%S_%f")
            return dt.timestamp()
        return 0.0
    except Exception as e:
        return 0.0

def align_episode(episode_path, dji_latency=0.0):
    print(f"\n📂 에피소드 처리 중: {episode_path} (DJI Latency Offset: {dji_latency}s)")
    
    images_base = os.path.join(episode_path, "images")
    raw_backup = os.path.join(images_base, "raw_backup")
    
    # 1. 기존 데이터 백업 및 폴더 정리
    if not os.path.exists(raw_backup):
        print(f"기존 Raw 데이터를 {raw_backup}으로 백업합니다...")
        os.makedirs(raw_backup, exist_ok=True)
        for folder in ["handeye", "pose_tracking", "additional_cam"]:
            src_folder = os.path.join(images_base, folder)
            if os.path.exists(src_folder):
                shutil.move(src_folder, os.path.join(raw_backup, folder))
    
    # 정렬된 데이터를 담을 원래 폴더 생성
    for folder in ["handeye", "pose_tracking", "additional_cam"]:
        os.makedirs(os.path.join(images_base, folder), exist_ok=True)

    # 2. 각 카메라별 파일 목록 및 타임스탬프 로드 (생략된 도우미 함수는 위와 동일)
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
        print("[❌ ERROR] Handeye 이미지가 없습니다.")
        return

    print(f"로드 완료: Handeye({len(h_strs)}), DJI({len(p_strs)}), Add({len(a_strs)})")
    print(f"In-place 디렉토리 기반 매칭 시작...")

    # 3. Handeye 기준으로 매칭 및 복사
    for i in tqdm(range(len(h_strs))):
        ref_ts_str = h_strs[i]
        ref_time = h_floats[i]
        
        # 3-1. Handeye (Master)
        src_h = os.path.join(raw_backup, "handeye", f"{ref_ts_str}.jpg")
        dst_h = os.path.join(images_base, "handeye", f"{ref_ts_str}.jpg")
        if os.path.exists(src_h):
            shutil.copy2(src_h, dst_h)
            
        # 3-2. DJI (Pose Tracking) - 최적 매칭 (Latency 보정 적용)
        if len(p_floats) > 0:
            # DJI는 하드웨어 지연으로 인해 '참값'보다 늦게 도착함.
            # 그래서 Handeye(T)에 맞는 DJI는 'T + latency' 시점에 도착한 놈임.
            target_p_time = ref_time + dji_latency
            p_idx = np.argmin(np.abs(p_floats - target_p_time))
            match_p_ts = p_strs[p_idx]
            src_p = os.path.join(raw_backup, "pose_tracking", f"{match_p_ts}.jpg")
            dst_p = os.path.join(images_base, "pose_tracking", f"{ref_ts_str}.jpg")
            shutil.copy2(src_p, dst_p)
            
        # 3-3. Additional Cam - 최적 매칭
        if len(a_floats) > 0:
            a_idx = np.argmin(np.abs(a_floats - ref_time))
            match_a_ts = a_strs[a_idx]
            src_a = os.path.join(raw_backup, "additional_cam", f"{match_a_ts}.jpg")
            dst_a = os.path.join(images_base, "additional_cam", f"{ref_ts_str}.jpg")
            shutil.copy2(src_a, dst_a)

    print(f"✅ 정렬 완료 (latency={dji_latency}s)!")
    
    # 4. 원본 데이터 삭제 (용량 확보)
    if os.path.exists(raw_backup):
        shutil.rmtree(raw_backup)
        print(f"🗑️ 원본 데이터(raw_backup) 삭제 완료.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="에피소드 폴더 경로")
    parser.add_argument("--episode", type=int, default=None, help="특정 에피소드 번호")
    parser.add_argument("--dji_latency", type=float, default=0.133, help="DJI 하드웨어/디코딩 지연 시간 (초), 4프레임=0.133")
    args = parser.parse_args()

    if args.episode is not None:
        align_episode(os.path.join(args.data_path, f"episode_{args.episode}"), args.dji_latency)
    else:
        ep_dirs = [d for d in os.listdir(args.data_path) if d.startswith("episode_")]
        for ep_dir in sorted(ep_dirs, key=lambda x: int(x.split('_')[1])):
            align_episode(os.path.join(args.data_path, ep_dir), args.dji_latency)