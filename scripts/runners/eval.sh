########################################################################
############################# Eval Dataset #############################
########################################################################

python scripts/eval/eval_dataset.py \
    --dataset_path /home/ailab-2204/Workspace/gail-umi/data/Gear_Insertion_0827_all.zarr\
    --model_checkpoint_path /home/ailab-2204/Workspace/gail-umi/data/outputs/__v2-256x256-p768-CA-non_frozen_dinov2-B_transformer-dp_500e_gear-half/checkpoints/epoch=0499-val_action_error=0.00005.ckpt \
    --episode_idx 0

# Replay dataset
python scripts/processing/replay_dataset_w_robot.py --server_port 4999 --base_dir /home/ailab-2204/Workspace/gail-umi/data/Gear_Insertion_0827_all/episode_1

########################################################################
############################# Eval Robot ###############################
########################################################################

# ManipForce Baseline
python scripts/eval/eval_robot.py \
--config_path "eval_config/gear_insertion.yaml" 

python scripts/eval/eval_robot.py \
--config_path "eval_config/battery_assemb.yaml" 

# Get weight from server
# rsync -avz --progress hinton:/home/geonhyup_lee/workspace/gail-umi/data/outputs/bi_crossatten_w_resid_2_Gear/checkpoints/latest.ckpt /home/ailab-2204/Workspace/gail-umi/data/outputs

########################################################################
################################ ETC ###################################
########################################################################

# FT 센서 전용 스크립트
# FT 데이터만 ROS로 publish (데이터 저장 없이)
python scripts/hardware_tests/check_franka_panda_ft.py --enable_ros --ft_filter_alpha 0.3
# FT 데이터 수집 + ROS publish + CSV 저장
python scripts/hardware_tests/check_franka_panda_ft.py --enable_ros --save_data --ft_filter_alpha 0.2 --output_file my_ft_data.csv
