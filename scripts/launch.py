import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Run training script with Hydra and custom dataset paths.")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID to use.")
    parser.add_argument('--config', type=str, default='manipforce_ods2_256x256', help="Hydra config name.")
    parser.add_argument('--dataset', type=str, default='lanport', help="Dataset key from the predefined dictionary.")
    args = parser.parse_args()

    # Predefined dataset paths
    dataset_path_dict = {
        'lanport': '../../dset/manipforce_dataset/LAN_Insertion_0830.zarr',
        'gear': '/home/ailab-2204/Workspace/ManipForce/data/gear_assem.zarr',
    }

    if args.dataset not in dataset_path_dict:
        print(f"Error: Dataset '{args.dataset}' not found in dataset_path_dict.")
        return

    gpu_id = args.gpu
    config_name = args.config
    if config_name.endswith('.yaml'):
        config_name = config_name[:-5]
    dataset_name = args.dataset
    dataset_path = dataset_path_dict[dataset_name]
    job_name = f"{config_name}_{dataset_name}"

    # Build the command string
    full_cmd = [
        f"CUDA_VISIBLE_DEVICES={gpu_id}",
        "python scripts/train.py",
        f"--config-name={config_name}",
        f"task.dataset_path={dataset_path}",
        f"logging.name={job_name}",
        f"multi_run.run_dir=outputs/{job_name}",
        f"multi_run.wandb_name_base={job_name}",
        f"hydra.job.override_dirname={job_name}",
        f"hydra.run.dir=outputs/{job_name}",
        f"hydra.sweep.dir=outputs/{job_name}"
    ]

    cmd_string = " ".join(full_cmd)
    print(f"Executing command:\n{cmd_string}\n")
    
    os.system(cmd_string)

if __name__ == "__main__":
    main()