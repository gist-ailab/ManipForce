"""Observation building: FT window selection, torque scaling, obs dict construction."""
import time
import numpy as np

from utils.eval_display import resize_with_padding


def convert_torque_scale(ft):
    """Robot FT 축을 GUMI 축으로 변환 (실제 측정 기반)"""
    ft = np.asarray(ft).copy()
    scale = 12.0 / 18.0
    if ft.ndim == 1:
        ft[3:6] *= scale
    else:
        ft[:, 3:6] *= scale
    return ft


def get_ft_time_range(prev_timestamps, config):
    """학습과 동일: 마지막 2개 이미지 프레임의 실제 timestamp 사이 FT 선택.
    학습 sampler는 img_timestamps[idx - obs_down_sample_steps]와
    img_timestamps[idx] 사이의 FT를 사용한다.
    """
    img_ts = list(prev_timestamps)
    if len(img_ts) >= 2:
        t0 = img_ts[-2]
        t1 = img_ts[-1]
    elif len(img_ts) == 1:
        ds = int(config['image'].get('down_sample_steps', 1))
        fps = config.get('action', {}).get('data_hz', 30)
        t1 = img_ts[0]
        t0 = t1 - ds / fps
    else:
        ds = int(config['image'].get('down_sample_steps', 1))
        fps = config.get('action', {}).get('data_hz', 30)
        t1 = time.time()
        t0 = t1 - ds / fps
    return t0, t1


def select_ft_window(ft_buf, ts_buf, t0, t1, config):
    """FT 버퍼에서 시간 범위에 맞는 프레임을 선택/패딩/샘플링."""
    target_frames = config['ft_sensor']['obs_horizon']
    img_ts_int = np.array([int(t0 * 1e6), int(t1 * 1e6)], dtype=np.int64)
    ft_ts_int = np.array([int(ts * 1e6) for ts in ts_buf], dtype=np.int64)
    left = max(0, np.searchsorted(ft_ts_int, img_ts_int[0], side='left'))
    right = min(len(ft_ts_int), np.searchsorted(ft_ts_int, img_ts_int[1], side='right'))
    ft_slice = ft_buf[left:right]
    ft_ts_slice = ts_buf[left:right]
    n = len(ft_slice)

    try:
        if n < target_frames:
            if n == 0:
                return (np.zeros((target_frames, 6), dtype=np.float32),
                        np.zeros(target_frames, dtype=np.float64))
            pad_n = target_frames - n
            ft_sel = np.concatenate([ft_slice, np.repeat(ft_slice[-1:], pad_n, axis=0)])
            ts_sel = np.concatenate([ft_ts_slice, np.repeat(ft_ts_slice[-1:], pad_n)])
        elif n > target_frames:
            idxs = np.linspace(0, n - 1, target_frames).round().astype(int)
            ft_sel, ts_sel = ft_slice[idxs], ft_ts_slice[idxs]
        else:
            ft_sel, ts_sel = ft_slice, ft_ts_slice
        assert ft_sel.shape[0] == target_frames
    except (ValueError, TypeError) as e:
        print(f"FT 타임스탬프 매핑 오류: {e}, 기본값 사용")
        ft_sel = np.zeros((target_frames, 6), dtype=np.float32)
        ts_sel = np.zeros(target_frames, dtype=np.float64)

    return ft_sel, ts_sel


def build_obs_np(prev_frames_cam1, prev_frames_cam2, ft_sel, config,
                 ft_timestamps=None, last_executed_action=None):
    """Build observation numpy dict for model inference."""
    target_size = config['image']['target_resolution'][0]

    cam1 = np.stack(prev_frames_cam1, axis=0).astype(np.float32) / 255.0
    cam2 = np.stack(prev_frames_cam2, axis=0).astype(np.float32) / 255.0
    cam1_tchw = np.moveaxis(cam1, -1, 1)
    cam2_tchw = np.moveaxis(cam2, -1, 1)

    def _resize_if_needed(tchw):
        if tchw.shape[2] == target_size and tchw.shape[3] == target_size:
            return tchw
        resized = []
        for t in range(tchw.shape[0]):
            img_hwc = np.moveaxis(tchw[t], 0, -1)
            img_u8 = (img_hwc * 255).astype(np.uint8)
            r = resize_with_padding(img_u8, target_size=target_size).astype(np.float32) / 255.0
            resized.append(np.moveaxis(r, -1, 0))
        return np.stack(resized, axis=0)

    obs_np = {
        'handeye_cam_1': _resize_if_needed(cam1_tchw),
        'handeye_cam_2': _resize_if_needed(cam2_tchw),
        'ft_data': ft_sel,
    }
    if ft_timestamps is not None:
        obs_np['ft_timestamps'] = ft_timestamps.astype(np.float32)
    return obs_np
