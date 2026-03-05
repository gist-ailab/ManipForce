import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Run training script with Hydra and custom dataset paths.")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID to use.")
    parser.add_argument('--config', type=str, default='manipforce_ods2_256x256', help="Hydra config name.")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset key (e.g. 'gear', 'battery') or a direct path to a .zarr file.")
    args = parser.parse_args()

    # Predefined dataset shortcuts (use relative paths, compatible with any environment)
    # You can also pass a direct path to a .zarr file via --dataset, e.g.:
    #   --dataset data/my_task.zarr
    dataset_path_dict = {
        'gear':    'data/gear_assem.zarr',
        'battery': 'data/battery_assem.zarr',
    }

    # Allow direct path to be passed as --dataset argument
    if os.path.exists(args.dataset):
        dataset_path = args.dataset
        dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
    elif args.dataset in dataset_path_dict:
        dataset_path = dataset_path_dict[args.dataset]
        dataset_name = args.dataset
    else:
        print(f"Error: Dataset '{args.dataset}' is not a known key or a valid path.")
        print(f"Available keys: {list(dataset_path_dict.keys())}")
        return
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