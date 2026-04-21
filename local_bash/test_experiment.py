import subprocess
import os
import argparse
from pathlib import Path

def run_experiment(listFile):
    print(f"🚀 Initializing CLRerNet evaluation...")
    
    # Path handling for output directory mapping
    path_parts = Path(listFile).parts
    if len(path_parts) >= 3:
        run_name = path_parts[-1].replace('.txt', '')
        parent_dir = path_parts[-2]
        grandparent_dir = path_parts[-3]
        show_dir = f"work_dirs/exodo/{grandparent_dir}/{parent_dir}/{run_name}"
    else:
        show_dir = f"work_dirs/exodo/{os.path.basename(listFile).replace('.txt', '')}"

    dataset_root = "/home1/adoyle2025/Datasets/Datasets/Curvelanes"
    
    command = [
        "python", "/home1/adoyle2025/CLRerNet/tools/test.py",
        "/home1/adoyle2025/CLRerNet/configs/clrernet/culane/clrernet_culane_dla34_ema.py",
        "/home1/adoyle2025/CLRerNet/clrernet_culane_dla34_ema.pth",
        "--cfg-options", 
        f'test_dataloader.dataset.data_list="{listFile}"',
        f'test_evaluator.data_list="{listFile}"',
        f'test_dataloader.dataset.data_root="{dataset_root}"',
        f'test_evaluator.data_root="{dataset_root}"',
        'default_hooks.visualization.draw=False'
    ]

    try:
        print(f"📄 Input List: {listFile}")
        print(f"📂 Associated Output Tracking: {show_dir}\n")
        subprocess.run(command, check=True)
        print(f"\n✅ Evaluation complete!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Command failed with error code {e.returncode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("listFile")
    args = parser.parse_args()
    if os.path.exists(args.listFile):
        run_experiment(args.listFile)
