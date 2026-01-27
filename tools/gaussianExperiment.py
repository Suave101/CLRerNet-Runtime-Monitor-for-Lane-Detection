import torch
import argparse
import json
import os
from mmengine.runner import Runner
from mmengine.config import Config
from mmengine.dist import get_rank

def parse_args():
    parser = argparse.ArgumentParser(description='Distributed Noise Test')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--sigma', type=float, default=0.0, help='Gaussian noise std')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # --- MANUAL PATH FIX ---
    # Force absolute paths for the dataset to bypass symlink/relative path issues
    ABS_DATA_ROOT = '/home1/adoyle2025/Distribution-Shift-Lane-Perception-Old/datasets/CULane'
    
    # Override test dataloader
    cfg.test_dataloader.dataset.data_root = ABS_DATA_ROOT
    cfg.test_dataloader.dataset.data_list = os.path.join(ABS_DATA_ROOT, 'list/test.txt')
    
    # Override evaluator
    cfg.test_evaluator.data_root = ABS_DATA_ROOT
    cfg.test_evaluator.data_list = os.path.join(ABS_DATA_ROOT, 'list/test.txt')

    # Fix the work_dir KeyError from earlier
    if 'work_dir' not in cfg:
        cfg.work_dir = './work_dirs/gaussian_test'
    # -----------------------

    cfg.launcher = args.launcher
    cfg.load_from = args.checkpoint

    runner = Runner.from_cfg(cfg)
    
    # Inject noise into the model's forward pass
    sigma = args.sigma
    original_forward = runner.model.forward
    
    def noisy_forward(img, **kwargs):
        if sigma > 0:
            # Noise must be on the same device as the image (e.g., cuda:0)
            noise = torch.randn_like(img) * sigma
            img = torch.clamp(img + noise, 0, 1)
        return original_forward(img, **kwargs)
    
    runner.model.forward = noisy_forward

    # Start testing
    metrics = runner.test()

    # Only the master process (Rank 0) should log the results to JSON
    if get_rank() == 0:
        log_file = "noise_results_distributed.json"
        entry = {"sigma": sigma, "metrics": metrics}
        
        # Append to file
        results = []
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                results = json.load(f)
        
        results.append(entry)
        
        with open('noise_results.json', 'w') as f:
            # Use 'default=int' to catch any numpy int64 and convert them
            json.dump(results, f, indent=4, default=lambda x: int(x) if hasattr(x, 'item') else x)
        
        print(f"🏁 Sigma {sigma} finished. Metrics: {metrics}")

if __name__ == '__main__':
    main()
