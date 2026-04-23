import subprocess
import os
import argparse
from pathlib import Path
try:
    from mmcv.utils import Config
except ImportError:
    try:
        from mmcv import Config
    except ImportError:
        from mmengine.config import Config


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

    # --- DYNAMIC CONFIG MODIFICATION ---
    base_config_path = "/home1/adoyle2025/CLRerNet-Runtime-Monitor-for-Lane-Detection/configs/clrernet/culane/clrernet_culane_dla34_ema.py"
    temp_config_path = "temp_curvelanes_eval_config.py"

    # Load the base config
    cfg = Config.fromfile(base_config_path)

    # Inject an albumentation Resize before the Crop step so that CurveLanes
    # images (which may differ from CULane's 1640x590) are scaled to the
    # CULane native resolution before cropping.
    # cfg.data.test.pipeline[0] is the 'albumentation' step whose 'pipelines'
    # list is [Compose, Crop, Resize].  Inserting at index 1 (after Compose,
    # before Crop) gives [Compose, Resize_to_CULane, Crop, Resize_to_model].
    resize_to_culane = dict(type='Resize', height=590, width=1640, p=1)
    cfg.data.test.pipeline[0]['pipelines'].insert(1, resize_to_culane)

    # Save the patched configuration
    cfg.dump(temp_config_path)
    # -----------------------------------

    command = [
        "python",
        "/home1/adoyle2025/CLRerNet-Runtime-Monitor-for-Lane-Detection/tools/test.py",
        temp_config_path,
        "/home1/adoyle2025/CLRerNet-Runtime-Monitor-for-Lane-Detection/clrernet_culane_dla34_ema.pth",
        "--cfg-options",
        f"data.test.data_list={listFile}",
        f"data.test.data_root={dataset_root}",
    ]

    try:
        print(f"📄 Input List: {listFile}")
        print(f"📂 Associated Output Tracking: {show_dir}\n")
        subprocess.run(command, check=True)
        print(f"\n✅ Evaluation complete!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Command failed with error code {e.returncode}")
    finally:
        # Clean up the temporary config file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("listFile")
    args = parser.parse_args()
    if os.path.exists(args.listFile):
        run_experiment(args.listFile)
    else:
        print(f"❌ File not found: {args.listFile}")
