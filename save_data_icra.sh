
# gumi anaconda environment
conda activate gumi
# ------------------------------------------------------------

# Aidin ft 300
python scripts/collection/capture_multimodal_data.py --data_path data/test_1104

# 캡쳐한 데이터에서 마커 포즈 추정(pose json)
python scripts/processing/get_wrist_pose.py --data_path data/test_1028_2

# filtering 및 보간
python scripts/processing/pose_refinement.py --data_path data/test_1028_2

# zarr 파일로 변환
python scripts/processing/change_to_zarr.py --data_path /media/ailab-2204/research/vc_umi/dataset/data_raw/Box_flipping --output_path /media/ailab-2204/research/vc_umi/dataset/dataset_zarr/Box_flipping.zarr
# python 2-5.change_to_zarr_w_ft_multi_imgs.py --data_path data/building_Tdomino_0906 --output_path data/building_Tdomino_0906.zarr

# ------------------------------------------------------------
# move to server
# rsync -avz --progress /home/ailab-2204/Workspace/gail-umi/data/Gear_Insertion_0827_all.zarr hinton:/SSDb/geonhyup_lee/workspace/gail_umi/
