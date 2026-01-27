import torch
import argparse
import json
import os
import torchvision.utils as vutils
from mmengine.runner import Runner
from mmengine.config import Config
from mmengine.dist import get_rank
from functools import partial
import cv2
import numpy as np  # <--- Added numpy import

def parse_args():
    parser = argparse.ArgumentParser(description='Distributed Blur Test')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--sigma', type=float, default=0.0, help='Gaussian Blur Sigma')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    return parser.parse_args()

def noisy_test_step(self, data, sigma=0.0):
    """
    Injected blur logic with DDP awareness.
    """
    # Fix: Access data_preprocessor via .module if wrapped in DDP
    model_base = self.module if hasattr(self, 'module') else self
    
    # 1. Preprocess (GPU move + Norm) using the base model's preprocessor
    data = model_base.data_preprocessor(data, False)
    inputs = data['inputs']

    # 2. Inject Gaussian Blur (Replaces previous Noise logic)
    if sigma > 0:
        # Move tensor to CPU and convert to Numpy for OpenCV
        # Shape is (Batch, Channel, Height, Width)
        inputs_np = inputs.cpu().numpy()
        blurred_batch = []

        for img in inputs_np:
            # Convert (C, H, W) -> (H, W, C) for OpenCV
            img_hwc = np.transpose(img, (1, 2, 0))

            # Apply Gaussian Blur
            # (0, 0) kernel size tells OpenCV to calculate window size automatically from sigma
            blurred = cv2.GaussianBlur(img_hwc, (0, 0), sigmaX=sigma, sigmaY=sigma)

            # If image is single channel (grayscale), OpenCV might drop the 3rd dim
            if blurred.ndim == 2:
                blurred = blurred[:, :, np.newaxis]

            # Convert back (H, W, C) -> (C, H, W)
            img_chw = np.transpose(blurred, (2, 0, 1))
            blurred_batch.append(img_chw)

        # Stack back into a batch, convert to Tensor, and move back to GPU
        inputs = torch.from_numpy(np.stack(blurred_batch)).to(inputs.device)
        
        # Ensure data stays within valid range (optional but recommended)
        data['inputs'] = torch.clamp(inputs, 0, 1)

    # 3. Verification Saves (Rank 0 only)
    if get_rank() == 0:
        if not hasattr(model_base, 'vis_count'):
            model_base.vis_count = 0
        if model_base.vis_count < 5:
            # Renamed folder to reflect it is blur, not noise
            vis_dir = f"verification_blur_sigma_{sigma}"
            os.makedirs(vis_dir, exist_ok=True)
            img_to_save = data['inputs'][0].detach().cpu()
            save_path = os.path.join(vis_dir, f"blurry_sample_{model_base.vis_count}.png")
            vutils.save_image(img_to_save, save_path)
            print(f"📸 Saved verification image: {save_path}")
            model_base.vis_count += 1

    # 4. Forward through the original self (DDP or Raw)
    return self(mode='predict', **data)

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # Absolute Path Fixes
    ABS_DATA_ROOT = '/home1/adoyle2025/Distribution-Shift-Lane-Perception-Old/datasets/CULane'
    cfg.test_dataloader.dataset.data_root = ABS_DATA_ROOT
    cfg.test_dataloader.dataset.data_list = os.path.join(ABS_DATA_ROOT, 'list/test.txt')
    cfg.test_evaluator.data_root = ABS_DATA_ROOT
    cfg.test_evaluator.data_list = os.path.join(ABS_DATA_ROOT, 'list/test.txt')

    if 'work_dir' not in cfg:
        cfg.work_dir = './work_dirs/gaussian_blur_test' # Updated work dir name

    cfg.launcher = args.launcher
    cfg.load_from = args.checkpoint

    runner = Runner.from_cfg(cfg)

    # --- THE FIX: MANUAL METHOD BINDING ---
    # We define a wrapper and use __get__ to bind it as a proper instance method
    sigma_val = args.sigma
    def test_step_wrapper(self, data):
        return noisy_test_step(self, data, sigma=sigma_val)

    runner.model.test_step = test_step_wrapper.__get__(runner.model, runner.model.__class__)
    # --------------------------------------

    metrics = runner.test()

    if get_rank() == 0:
        # Renamed log file to distinguish from noise tests
        log_file = "blur_results.json"
        entry = {"sigma": args.sigma, "metrics": metrics}
        
        results = []
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                try:
                    results = json.load(f)
                except:
                    results = []
        
        results.append(entry)
        with open(log_file, 'w') as f:
            json.dump(results, f, indent=4, default=lambda x: float(x) if hasattr(x, 'item') else x)
        
        print(f"🏁 Finished Sigma {args.sigma}. Results saved to {log_file}")

if __name__ == '__main__':
    main()
