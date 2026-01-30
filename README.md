<div align="center">
<img src="assets/banner.png" width="75%">

Geonhyup Lee, Yeongjin Lee, Kangmin Kim, Seongju Lee, Sangjun Noh, Seunghyeok Back, Kyoobin Lee

</div>

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

## 🏋️ Training
Our method supports different observation down-sampling steps.
```bash
# For tasks requiring high precision (obs_down_sample_steps=2), e.g., LAN Insertion
python scripts/launch.py --gpu 0 --config manipforce_ods2_256x256 --dataset lanport

# For other tasks (obs_down_sample_steps=3)
python scripts/launch.py --gpu 0 --config manipforce_ods3_256x256 --dataset lanport
```

### 📝 Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--gpu` | GPU ID to use for training. | `0` |
| `--config` | Hydra configuration file name (with or without `.yaml`). | `manipforce_ods2_256x256` |
| `--dataset` | Dataset key name defined in `scripts/launch.py` (e.g., `lanport`). | `lanport` |