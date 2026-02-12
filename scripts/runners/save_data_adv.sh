
# gumi anaconda environment
conda activate gumi
# ------------------------------------------------------------

# Step1 : Capture Multimodal Data
python scripts/collection/capture_multimodal_data_adv.py --data_path data/gear_assem --add_cam
python scripts/collection/capture_multimodal_data_adv.py --data_path data/battery_assem --add_cam

# Step2 : Synchronize Images
python scripts/collection/align_multimodal_data.py --data_path data/gear_assem
python scripts/collection/align_multimodal_data.py --data_path data/battery_assem

# Verify Synchronized Images (Option)
python scripts/collection/verify_sync_interactive.py --episode_path data/gear_assem/episode_1 --synced --add_cam

# Step3 : Estimate Marker Pose
python scripts/processing/get_wrist_pose_adv.py --data_path data/battery_assem --visualize

# Step4 : Filtering and Interpolation
python scripts/processing/pose_refinement.py --data_path data/battery_assem

# Check Data (Option)
python scripts/processing/check_data.py --data_path data/gear_assem/episode_1

# Step5 : Convert to Zarr
python scripts/processing/change_to_zarr.py --data_path data/battery_assem --output_path data/battery_assem.zarr

# ------------------------------------------------------------
# move to server
# rsync -avz --progress /home/ailab-2204/Workspace/gail-umi/data/Gear_Insertion_0827_all.zarr hinton:/SSDb/geonhyup_lee/workspace/gail_umi/

rsync -avz --progress /home/ailab-2204/Workspace/ManipForce/data/battery_assem.zarr ailab@172.27.183.165:/SSD1/Workspace/ManipForce/data/