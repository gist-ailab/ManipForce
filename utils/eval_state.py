"""EvalState — mutable state bag for the evaluation loop."""

import os
import time
import numpy as np
from collections import deque
from utils.video_writer import VideoWriter


class EvalState:
    """Mutable state bag for the evaluation loop."""

    def __init__(self, cfg):
        # image history (from config)
        self.prev_frames_cam1 = deque(maxlen=cfg['image']['history_length'])
        self.prev_frames_cam2 = deque(maxlen=cfg['image']['history_length'])
        self.prev_timestamps = deque(maxlen=cfg['image']['history_length'])

        # display / Hz
        self.last_disp_t = time.time()
        self.disp_cnt = 0
        self.img_accum = 0.0
        self.ft_accum = 0.0
        self.gui_inference_hz = 0.0
        self.gui_cycle_ms = 0.0
        self.gui_mode_label = ""
        self.gui_timesteps_str = ""

        # policy timing
        self.policy_start_time = None
        self.policy_start_pose = None

        # video
        self.video_recorder = VideoWriter(fps=30.0)
        self.output_base = cfg.get('paths', {}).get('output_dir', 'eval_results')
        self.output_dir = '.'

        # diffusion intermediates
        self.diffusion_intermediates = []
        self.save_intermediates = cfg.get('model', {}).get('save_intermediates', False)
        self.headless = cfg.get('debug', {}).get('headless', False)

        # FT
        self.ft_latest = np.zeros(6, dtype=np.float32)
        self.ft_time_start = None
        self.ft_contact_active = False

        # gear insertion
        self.gear_inserted = False
        self.gear_insert_time = None

        # task state machine
        self.task_state = 'IDLE'
        self.contact_step = None
        self.contact_time = None
        self.insert_step = None
        self.insert_time = None
        self.total_inference_steps = 0
        self.robot_moving = False
        self.robot_move_threshold = 0.001
        self.robot_start_ee_pose = None

        # model timing
        self.model_time_accum = 0.0
        self.model_call_count = 0

        # gripper (from config)
        self.gripper_history = deque(maxlen=cfg['gripper']['history_length'])
        self.current_gripper_state = 'open'
        self.current_gripper_target = 'open'

        # action history
        self.action_history = deque(maxlen=4)

        # interpolation state (from config)
        self.interp_queue = []
        self.interp_idx = 0
        self.interp_start_time = 0.0
        self.interp_interval = 1.0 / cfg.get('action', {}).get('data_hz', 7.5)
        self.interp_last_target = None

        # display: model input FT time range
        self.ft_model_t0 = None
        self.ft_model_t1 = None

        # display: latest model output action
        self.last_raw_action = None
        self.last_transformed_action = None
        self.last_target_pose = None
        self.last_current_ee = None

        # contact assist
        self.ca_active = False
        self.ca_force_norm = 0.0
        self.ca_offset_mm = np.zeros(3)
        self.ca_force_dir_world = np.zeros(3)
        self.ca_force_body = np.zeros(3)
        self.ca_ft_baseline = np.zeros(6, dtype=np.float32)
        self.ca_last_applied_time = 0.0  # 마지막 CA 적용 시각 (GUI 표시 유지용)
        self.ca_apply_count = 0  # CA 적용 총 횟수
        self.ft_model_input = np.zeros(6, dtype=np.float32)
        self.ft_model_input_history = []  # list of (timestamp, ft_6d) for graph overlay

        # Per-trial gripping force bias (학습 zarr와 동일하게 에피소드 초기 FT 평균 제거)
        self.gripping_ft_bias = np.zeros(6, dtype=np.float32)
        # FT[5:100] recalibration: 정책 시작 후 FT 수집하여 학습과 동일한 bias 계산
        self.ft_recalib_done = False
        self.ft_recalib_start_time = None  # warmup 끝나는 시점의 FT timestamp
        self.ft_recalib_buffer = []  # 정책 시작 후 수집된 raw FT 샘플

        # EE pose ring buffer for pose_wrt_start: (timestamp, 7D pose) pairs
        self.ee_pose_buffer = deque(maxlen=60)  # ~2sec at 30Hz

        # obs_down_sample_steps == 'inference' 모드: 이전 inference 프레임 저장
        self.prev_inference_cam1 = None  # (timestamp, rgb_array)
        self.prev_inference_cam2 = None

        # current mode (GUI reads this)
        self.current_mode = 'teleop'

        # observe mode
        self.observe_mode = False
        self.observe_actions = []
        self.observe_fig = None

    def build_output_dir(self, config_path, config):
        """모델 로드 후 호출 — config 이름 + 모드 + 설정으로 하위 폴더 자동 생성."""
        task_name = os.path.splitext(os.path.basename(config_path))[0] if config_path else 'unknown'
        mode_tag = self.gui_mode_label.replace(' ', '_') if self.gui_mode_label else 'unknown'
        chunk = config['action'].get('chunk_steps', 1)
        chunk_tag = f"{chunk}step" if chunk > 1 else "1step"
        sync_tag = "sync" if config.get('control', {}).get('sync_mode', False) else "async"
        action_mode = config['action'].get('action_mode', 'delta')
        sub_dir = f"{mode_tag}_{action_mode}_{chunk_tag}_{sync_tag}"
        self.output_dir = os.path.join(self.output_base, task_name, sub_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def reset_task_state(self):
        """Reset all task-related state (called on teleop transition)."""
        self.policy_start_time = None
        self.ft_time_start = None
        self.ft_contact_active = False
        self.gear_inserted = False
        self.gear_insert_time = None
        self.task_state = 'IDLE'
        self.contact_step = None
        self.contact_time = None
        self.insert_step = None
        self.insert_time = None
        self.diffusion_intermediates = []
        self.model_time_accum = 0.0
        self.model_call_count = 0
        self.ft_model_input_history = []
        self.gripping_ft_bias = np.zeros(6, dtype=np.float32)
        self.ft_recalib_done = False
        self.ft_recalib_start_time = None
        self.ft_recalib_buffer = []
        # reset inference-mode image history
        self.prev_inference_cam1 = None
        self.prev_inference_cam2 = None
        # reset interpolation queue
        self.interp_queue = []
        self.interp_idx = 0
