<div align="center">
<img src="assets/banner.png" width="50%">

Geonhyup Lee, Youngjin Lee, Kangmin Kim, Seongju Lee, Sangjun Noh, Seunghyeok Back, Kyoobin Lee

</div>

---


This is an official implementation for "ManipForce: Force-Guided Policy Learning with Frequency-Aware Representation for Contact-Rich Manipulation", 2026 IEEE International Conference on Robotics and Automation (ICRA 2026).

## 🛠️ Setup
```bash
# 1. Install mamba (if not already installed)
conda install -c conda-forge mamba -n base -y

# 2. Create conda environment
mamba env create -f environment.yml

# 3. Activate environment
conda activate manipforce

# 4. Download pre-trained models
python checkpoints/prepare_dinov2.py
```

## 📡 Data Collection & Processing
```bash
# Step 1: Capture multimodal data
python scripts/collection/capture_multimodal_data_adv.py --data_path data/260210 --add_cam

# Step 2: Synchronize multi-camera images
python scripts/collection/align_multimodal_data.py --data_path data/260210

# Step 3: Estimate wrist marker pose
python scripts/processing/get_wrist_pose_adv.py --data_path data/260210 --visualize --add_cam

# Step 4: Refine pose with filtering and interpolation
python scripts/processing/pose_refinement.py --data_path data/260210

# Step 5: Convert processed data to Zarr format
python scripts/processing/change_to_zarr.py --data_path data/260210 --output_path data/260210.zarr
```


## 🏋️ Training
Our method supports different observation down-sampling steps.
```bash
# For tasks requiring high precision (obs_down_sample_steps=2), e.g., LAN Insertion
python scripts/launch.py --gpu 0 --config manipforce_ods2_256x256 --dataset lanport

# For other tasks (obs_down_sample_steps=3)
python scripts/launch.py --gpu 0 --config manipforce_ods3_256x256 --dataset lanport
```

## 🤖 Evaluation
```bash
python scripts/eval/eval_robot.py --config_path "eval_config/gear_insertion.yaml" 
```

### 📝 Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--gpu` | GPU ID to use for training. | `0` |
| `--config` | Hydra configuration file name (with or without `.yaml`). | `manipforce_ods2_256x256` |
| `--dataset` | Dataset key name defined in `scripts/launch.py` (e.g., `lanport`). | `lanport` |