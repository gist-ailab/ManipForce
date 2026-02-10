#!/usr/bin/env python3


import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import zarr
import numpy as np
import os
import sys

def check_zarr_structure(dataset_path):
    """zarr 데이터셋 구조와 내용을 직접 확인"""
    
    print(f"🔍 Zarr 데이터셋 분석: {dataset_path}")
    print("="*60)
    
    # zarr 파일 열기
    if dataset_path.endswith('.zarr.zip'):
        store = zarr.ZipStore(dataset_path, mode='r')
        root = zarr.group(store)
    else:
        root = zarr.open(dataset_path, mode='r')
    
    print("📁 최상위 그룹 구조:")
    print(f"  Keys: {list(root.keys())}")
    print()
    
    # data 그룹 확인
    if 'data' in root:
        data_group = root['data']
        print("📊 data 그룹:")
        print(f"  Keys: {list(data_group.keys())}")
        
        for key in data_group.keys():
            dataset = data_group[key]
            print(f"  📈 {key}:")
            print(f"    - 형태: {dataset.shape}")
            print(f"    - 타입: {dataset.dtype}")
            if len(dataset) > 0:
                if 'timestamp' in key.lower():
                    print(f"    - 범위: {dataset[0]} ~ {dataset[-1]}")
                elif key == 'ft_data':
                    print(f"    - 값 범위: {np.min(dataset[:])} ~ {np.max(dataset[:])}")
                    print(f"    - 첫 5개 샘플:")
                    for i in range(min(5, len(dataset))):
                        print(f"      [{i}]: {dataset[i]}")
                    
                    # 0이 아닌 값 확인
                    non_zero_mask = np.any(dataset[:] != 0, axis=1)
                    non_zero_count = np.sum(non_zero_mask)
                    print(f"    - 0이 아닌 샘플: {non_zero_count}/{len(dataset)} ({non_zero_count/len(dataset)*100:.1f}%)")
                    
                elif key in ['action', 'state', 'pose_wrt_start']:  # pose_wrt_start 추가
                    print(f"    - 값 범위: {np.min(dataset[:])} ~ {np.max(dataset[:])}")
                    print(f"    - 첫 3개 샘플:")
                    for i in range(min(3, len(dataset))):
                        print(f"      [{i}]: {dataset[i]}")
                        
                    # pose_wrt_start에 대한 추가 검증
                    if key == 'pose_wrt_start':
                        # 시작점이 원점(0)인지 확인
                        if not np.allclose(dataset[0], np.zeros_like(dataset[0])):
                            print("    ⚠️  Warning: First pose_wrt_start is not zero!")
                        
                        # 상대 위치의 변화량 확인
                        position_changes = np.linalg.norm(dataset[1:, :3] - dataset[:-1, :3], axis=1)
                        print(f"    - 평균 위치 변화량: {np.mean(position_changes):.4f}")
                        print(f"    - 최대 위치 변화량: {np.max(position_changes):.4f}")
            else:
                print(f"    - ⚠️  비어있음!")
        print()
    
    # meta 그룹 확인
    if 'meta' in root:
        meta_group = root['meta']
        print("📋 meta 그룹:")
        print(f"  Keys: {list(meta_group.keys())}")
        
        for key in meta_group.keys():
            if hasattr(meta_group[key], 'shape'):  # 데이터셋인 경우
                dataset = meta_group[key]
                print(f"  📌 {key}: {dataset[:]}")
            else:  # 속성인 경우
                print(f"  📌 {key}: {meta_group[key]}")
        
        # 속성들도 확인
        if hasattr(meta_group, 'attrs'):
            print(f"  📎 Attributes: {dict(meta_group.attrs)}")
        print()
    
    # 에피소드 분석
    if 'data' in root and 'meta' in root:
        data_group = root['data']
        meta_group = root['meta']
        
        print("🎬 에피소드 분석:")
        
        # 에피소드 경계 확인
        episode_ends = None
        episode_ft_ends = None
        episode_img_ends = None
        
        if 'episode_ends' in meta_group:
            episode_ends = meta_group['episode_ends'][:]
            print(f"  📐 기본 에피소드 경계: {episode_ends}")
        
        if 'episode_ft_ends' in meta_group:
            episode_ft_ends = meta_group['episode_ft_ends'][:]
            print(f"  🔧 FT 에피소드 경계: {episode_ft_ends}")
        
        if 'episode_img_ends' in meta_group:
            episode_img_ends = meta_group['episode_img_ends'][:]
            print(f"  📷 이미지 에피소드 경계: {episode_img_ends}")
        
        # 각 에피소드별 데이터 분석
        if episode_ends is not None:
            print(f"\n  📊 에피소드별 데이터 분석:")
            
            for i, end_idx in enumerate(episode_ends):
                start_idx = episode_ends[i-1] if i > 0 else 0
                
                print(f"    에피소드 {i+1}: 인덱스 {start_idx}~{end_idx} ({end_idx-start_idx}개)")
                
                # 각 데이터 타입별 개수 확인
                for key in data_group.keys():
                    if key in ['handeye_cam_1', 'handeye_cam_2', 'action', 'state', 'pose_wrt_start', 'img_timestamps']:  # pose_wrt_start 추가
                        # 이미지 관련 데이터
                        if episode_img_ends is not None:
                            img_start = episode_img_ends[i-1] if i > 0 else 0
                            img_end = episode_img_ends[i]
                            count = img_end - img_start
                        else:
                            count = end_idx - start_idx
                        
                        # pose_wrt_start에 대한 추가 검증
                        if key == 'pose_wrt_start':
                            print(f"      {key}: {count}개")
                            # 에피소드의 첫 프레임이 0인지 확인
                            if count > 0:
                                first_pose = data_group[key][start_idx]
                                if not np.allclose(first_pose, np.zeros_like(first_pose)):
                                    print(f"        ⚠️  Warning: First pose is not zero: {first_pose}")
                                print(f"        첫 프레임: {first_pose}")
                        else:
                            print(f"      {key}: {count}개")
                    
                    elif key in ['ft_data', 'ft_timestamps']:
                        # FT 관련 데이터
                        if episode_ft_ends is not None:
                            ft_start = episode_ft_ends[i-1] if i > 0 else 0
                            ft_end = episode_ft_ends[i]
                            count = ft_end - ft_start
                            
                            if count > 0 and key == 'ft_data':
                                # FT 데이터 샘플 확인
                                ft_sample = data_group[key][ft_start:ft_start+min(3, count)]
                                print(f"      {key}: {count}개")
                                print(f"        샘플: {ft_sample}")
                            else:
                                print(f"      {key}: {count}개")  
                        else:
                            print(f"      {key}: 경계 정보 없음")
                print()
    
    # 타임스탬프 매칭 테스트
    if 'data' in root:
        data_group = root['data']
        
        if 'img_timestamps' in data_group and 'ft_timestamps' in data_group:
            print("⏰ 타임스탬프 매칭 테스트:")
            
            img_ts = data_group['img_timestamps']
            ft_ts = data_group['ft_timestamps']
            
            if len(img_ts) > 0 and len(ft_ts) > 0:
                print(f"  이미지 타임스탬프: {len(img_ts)}개")
                print(f"    범위: {img_ts[0]} ~ {img_ts[-1]}")
                print(f"    첫 5개: {img_ts[:5]}")
                
                print(f"  FT 타임스탬프: {len(ft_ts)}개")  
                print(f"    범위: {ft_ts[0]} ~ {ft_ts[-1]}")
                print(f"    첫 5개: {ft_ts[:5]}")
                
                # 겹치는 범위 확인
                img_min, img_max = img_ts[0], img_ts[-1]
                ft_min, ft_max = ft_ts[0], ft_ts[-1]
                
                overlap_start = max(img_min, ft_min)
                overlap_end = min(img_max, ft_max)
                
                if overlap_start <= overlap_end:
                    print(f"  ✅ 타임스탬프 겹침 범위: {overlap_start} ~ {overlap_end}")
                else:
                    print(f"  ❌ 타임스탬프 겹침 없음!")
                    print(f"    이미지: {img_min} ~ {img_max}")
                    print(f"    FT: {ft_min} ~ {ft_max}")
    
    # 파일 닫기
    if dataset_path.endswith('.zarr.zip'):
        store.close()
    
    print("="*60)
    print("✅ 분석 완료!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        # 기본 경로들 시도
        possible_paths = [
            "data/LanPort_Insertion_0810.zarr"
        ]
        
        dataset_path = None
        for path in possible_paths:
            if os.path.exists(path):
                dataset_path = path
                break
        
        if dataset_path is None:
            print("사용법: python check_zarr_data.py <zarr_dataset_path>")
            print("또는 다음 경로 중 하나에 데이터셋을 배치하세요:")
            for path in possible_paths:
                print(f"  - {path}")
            sys.exit(1)
    
    if not os.path.exists(dataset_path):
        print(f"❌ 파일을 찾을 수 없습니다: {dataset_path}")
        sys.exit(1)
    
    check_zarr_structure(dataset_path)