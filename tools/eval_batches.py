"""
Batch evaluation script using the CLRerNet runner pipeline (Sandbox Approach).

For each set of image paths found in experiment JSON log files, this script:
  1. Dynamically constructs a temporary CULane-structured sandbox directory.
  2. Symlinks the target images into the sandbox with safe relative paths.
  3. Converts Curvelanes JSON labels to CULane .lines.txt format, scaling to 1640x590.
  4. Points the dataloader and evaluator at this perfect isolated sandbox.
  5. Cleans up the sandbox after metric extraction.

Usage:
    python tools/eval_batches.py \
        configs/clrernet/culane/clrernet_culane_dla34_ema.py \
        clrernet_culane_dla34_ema.pth \
        --data-root /path/to/CULane \
        --logs-dir  /path/to/logs/Exodo \
        --out-csv   /path/to/master_evaluation_results.csv
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

# Ensure the repo root is on sys.path when the script is run directly.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from mmengine.config import Config
from mmengine.runner import Runner

import cv2
import numpy as np

# ==============================================================================
# ULTIMATE RUNTIME INTERCEPTOR (MONKEY PATCH V2)
# ==============================================================================
try:
    from libs.datasets.pipelines.compose import Compose
    if not hasattr(Compose, '_patched_for_resize'):
        _orig_compose_call = Compose.__call__
        
        def _patched_compose_call(self, data):
            if 'img' in data and isinstance(data['img'], np.ndarray):
                h, w = data['img'].shape[:2]
                if w != 1640 or h != 590:
                    data['img'] = cv2.resize(data['img'], (1640, 590))
                    data['img_shape'] = (590, 1640)
                    data['ori_shape'] = (590, 1640)
            
            for t in self.transforms:
                data = t(data)
                if data is None:
                    return None
            return data
            
        Compose.__call__ = _patched_compose_call
        Compose._patched_for_resize = True
        print("\n  [SYSTEM] UNCONDITIONAL INTERCEPTOR ACTIVE: All incoming images forced to 1640x590.\n")
except Exception as e:
    print(f"\n  [WARN] Failed to initialize Pipeline Interceptor: {e}\n")
# ==============================================================================


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_sandbox(image_paths: list[str]) -> Path:
    """
    Creates a temporary CULane-structured filesystem isolating the C++ 
    evaluator from string concatenation bugs on absolute paths.
    """
    sandbox = Path(tempfile.mkdtemp(prefix="culane_sandbox_"))
    
    # Replicate exact CULane directory structure
    (sandbox / "images").mkdir(exist_ok=True)
    split_dir = sandbox / "list" / "test_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    
    # --- FIX: Create empty files for all 9 standard CULane categories ---
    # This prevents the CULane metric from throwing FileNotFoundError
    culane_categories = [
        "test0_normal.txt", "test1_crowd.txt", "test2_hlight.txt",
        "test3_shadow.txt", "test4_noline.txt", "test5_arrow.txt",
        "test6_curve.txt", "test7_cross.txt", "test8_night.txt"
    ]
    for cat in culane_categories:
        (split_dir / cat).touch()
    # --------------------------------------------------------------------
    
    list_file_path = sandbox / "test_list.txt"
    split_file_path = split_dir / "test0_normal.txt"
    
    with open(list_file_path, "w") as lf, open(split_file_path, "w") as sf:
        for i, img_path in enumerate(image_paths):
            orig_img = Path(img_path.strip())
            if not orig_img.exists():
                continue
            
            # Use completely safe relative paths (no leading filesystem roots)
            rel_name = f"images/img_{i:04d}.jpg"
            fake_img = sandbox / rel_name
            
            # Symlink the actual image into the sandbox
            try:
                os.symlink(orig_img.absolute(), fake_img)
            except OSError:
                shutil.copy2(orig_img, fake_img)
            
            # Generate the CULane Ground Truth
            fake_txt = fake_img.with_suffix(".lines.txt")
            converted = False
            
            parts = list(orig_img.parts)
            # Detect if it's Curvelanes and convert JSON to TXT
            if "images" in parts and "urvelane" in str(orig_img).lower():
                idx = parts.index("images")
                parts[idx] = "labels"
                json_path = Path(*parts[:-1]) / f"{orig_img.stem}.lines.json"
                
                if json_path.exists():
                    try:
                        img = cv2.imread(str(orig_img))
                        orig_h, orig_w = img.shape[:2] if img is not None else (1440, 2560)
                        scale_x = 1640.0 / orig_w
                        scale_y = 590.0 / orig_h
                        
                        with open(json_path, "r") as jf:
                            data = json.load(jf)
                        
                        with open(fake_txt, "w") as tf:
                            for line in data.get("Lines", []):
                                coords = []
                                for pt in line:
                                    nx = float(pt["x"]) * scale_x
                                    ny = float(pt["y"]) * scale_y
                                    coords.append(f"{nx:.3f}")
                                    coords.append(f"{ny:.3f}")
                                tf.write(" ".join(coords) + "\n")
                        converted = True
                    except Exception as e:
                        print(f"  [WARN] Failed converting JSON for {orig_img.name}: {e}")
            
            # Fallback: copy existing CULane TXT or leave empty
            if not converted:
                orig_txt = orig_img.with_suffix(".lines.txt")
                if orig_txt.exists():
                    try:
                        os.symlink(orig_txt.absolute(), fake_txt)
                    except OSError:
                        shutil.copy2(orig_txt, fake_txt)
                else:
                    fake_txt.touch() # Empty file => 0 lanes
                    
            # Write exactly how CULane metric expects it
            lf.write(f"/{rel_name}\n")
            sf.write(f"/{rel_name}\n")
            
    return sandbox


def _build_runner(
    cfg_path: str,
    checkpoint: str,
    sandbox_root: str,
    list_file: str,
) -> Runner:
    cfg = Config.fromfile(cfg_path)

    # Point everything exclusively to our perfect temporary sandbox
    cfg.test_dataloader.dataset.data_root = sandbox_root
    cfg.test_dataloader.dataset.data_list = list_file
    cfg.test_evaluator.data_root = sandbox_root
    cfg.test_evaluator.data_list = list_file

    cfg.load_from = checkpoint

    if "work_dir" not in cfg or cfg.work_dir is None:
        cfg.work_dir = "./work_dirs/evaluate_batches"

    cfg.default_hooks = cfg.get("default_hooks", {})
    cfg.default_hooks.pop("visualization", None)

    try:
        cfg.model.bbox_head.loss_seg.loss_weight = 0
    except (AttributeError, KeyError):
        pass

    runner = Runner.from_cfg(cfg)
    return runner


def _force_sequential_culane_eval() -> None:
    """
    Force CULane metric computation to run sequentially inside each worker.

    This avoids nested multiprocessing failures when outer image-level
    evaluation already runs in parallel processes.
    """
    try:
        from libs.datasets.metrics import culane_metric as culane_metric_mod
    except Exception:
        return

    if getattr(culane_metric_mod, "_eval_batches_sequential_patched", False):
        return

    original_eval_predictions = culane_metric_mod.eval_predictions

    def _wrapped_eval_predictions(*args, **kwargs):
        kwargs["sequential"] = True
        return original_eval_predictions(*args, **kwargs)

    culane_metric_mod.eval_predictions = _wrapped_eval_predictions
    culane_metric_mod._eval_batches_sequential_patched = True


def _run_batch(
    cfg_path: str,
    checkpoint: str,
    data_root: str,  # Unused directly, sandbox abstracts it
    image_paths: list[str],
) -> dict:
    if not image_paths:
        return {}

    sandbox = _setup_sandbox(image_paths)
    list_file = str(sandbox / "test_list.txt")

    try:
        runner = _build_runner(
            cfg_path, checkpoint, str(sandbox), list_file
        )
        metrics = runner.test()  
        return metrics if metrics else {}
    finally:
        # Obliterate sandbox when done so the server doesn't bloat
        shutil.rmtree(sandbox, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CLRerNet on custom image batches from experiment JSON logs"
    )
    parser.add_argument("config", help="CLRerNet config file path")
    parser.add_argument("checkpoint", help="Checkpoint (.pth) file path")
    parser.add_argument(
        "--data-root",
        required=True,
        help="CULane dataset root directory (unused due to sandbox, kept for cli compat)",
    )
    parser.add_argument(
        "--logs-dir",
        required=True,
        help="Directory containing experiment *.json log files",
    )
    parser.add_argument(
        "--out-csv",
        default="master_evaluation_results.csv",
        help="Path for the output CSV file",
    )
    parser.add_argument(
        "--cache-file",
        default=None,
        help=(
            "Path to persistent per-image metrics cache JSON. "
            "Default: <logs-dir>/image_metrics_cache.json"
        ),
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=4,
        help="Number of GPUs / parallel workers to use for unique image evaluation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    cfg_path = os.path.abspath(args.config)
    checkpoint = os.path.abspath(args.checkpoint)
    data_root = os.path.abspath(args.data_root)
    logs_dir = os.path.abspath(args.logs_dir)
    out_csv = os.path.abspath(args.out_csv)
    cache_path = os.path.abspath(
        args.cache_file or os.path.join(logs_dir, "image_metrics_cache.json")
    )
    if args.num_gpus <= 0:
        raise ValueError("--num-gpus must be a positive integer.")
    num_gpus = args.num_gpus

    json_files = sorted(Path(logs_dir).glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {logs_dir}")
        return

    # Pass 1: collect all unique image paths from all JSON runs.
    print("\n=== Pass 1/2: Collecting unique image paths from all JSON files ===")
    all_runs_first_pass = list(_iter_json_runs(json_files))
    unique_images = {
        _normalize_image_path(p, data_root)
        for run in all_runs_first_pass
        for p in run["image_paths"]
        if _normalize_image_path(p, data_root)
    }
    print(f"Found {len(all_runs_first_pass)} runs and {len(unique_images)} unique images.")

    # Load cache and evaluate only missing image paths.
    metrics_cache = _load_metrics_cache(cache_path)
    pending_images = [p for p in sorted(unique_images) if p not in metrics_cache]
    print(f"Cache entries: {len(metrics_cache)}; pending image evaluations: {len(pending_images)}")

    failed_images: list[tuple[str, str]] = []
    if pending_images:
        print(f"\n=== Evaluating pending unique images in parallel ({num_gpus} workers) ===")
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = []
            for i, image_path in enumerate(pending_images):
                gpu_id = i % num_gpus
                futures.append(
                    executor.submit(
                        _evaluate_image_worker,
                        cfg_path,
                        checkpoint,
                        data_root,
                        image_path,
                        gpu_id,
                    )
                )

            for future in tqdm(as_completed(futures), total=len(futures), desc="unique images"):
                image_path, image_metrics, error = future.result()
                if image_metrics:
                    metrics_cache[image_path] = image_metrics
                elif error:
                    failed_images.append((image_path, error))

        _save_metrics_cache(cache_path, metrics_cache)
        print(f"Updated cache saved to: {cache_path}")

    if failed_images:
        print(f"[WARN] {len(failed_images)} images failed evaluation. First 10:")
        for image_path, err in failed_images[:10]:
            print(f"  - {image_path}: {err}")

    # Pass 2: re-scan JSON files and compute per-run average metrics from cache.
    print("\n=== Pass 2/2: Building CSV rows from cached per-image metrics ===")
    all_rows = []

    for json_path in json_files:
        filename = json_path.stem
        print(f"\n=== Processing log file: {json_path.name} ===")

        with open(json_path, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as exc:
                print(f"  [WARN] Could not parse {json_path.name}: {exc}")
                continue

        for exp in data.get("experiments", []):
            args_fields = exp.get("arguments", {})

            target_keys = [
                ("Sanity Check", "sanity_check"),
                ("Data Shift Test Data", "test_run"),
            ]

            for key, run_type in target_keys:
                if key not in exp.get("data", {}):
                    continue

                if key == "Sanity Check":
                    runs = [
                        {
                            "Image Paths": exp["data"][key]["Image Paths"],
                            "Run": -1,
                            "Seed": "N/A",
                            "Results": exp["data"][key],
                        }
                    ]
                else:
                    runs = exp["data"][key].get("Individual Test Data", [])

                for run in tqdm(runs, desc=f"  {filename} / {run_type}", leave=False):
                    image_paths = run.get("Image Paths", [])
                    res = run.get("Results", {})

                    metrics = _run_batch(cfg_path, checkpoint, data_root, image_paths)

                    row = {
                        **args_fields,
                        "log_file": filename,
                        "run_type": run_type,
                        "run_idx": run.get("Run"),
                        "seed": run.get("Seed"),
                        "images_in_sample": len(image_paths),
                        "f1_score": metrics.get("F1_0.5"),
                        "precision": metrics.get("Precision0.5"),
                        "recall": metrics.get("Recall0.5"),
                        "tp": metrics.get("TP0.5"),
                        "fp": metrics.get("FP0.5"),
                        "fn": metrics.get("FN0.5"),
                        "f1_score_iou0.1": metrics.get("F1_0.1"),
                        "f1_score_iou0.75": metrics.get("F1_0.75"),
                        "mmd_stat": res.get("MMD", {}).get("Stat"),
                        "mmd_p_value": res.get("MMD", {}).get("P-Value"),
                        "mmd_shift_detected": res.get("MMD", {}).get("Shift Detected"),
                    }
                    all_rows.append(row)

    if not all_rows:
        print("\nNo results to write — check that your JSON files have the expected structure.")
        return

    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\n✅ SUCCESS! Metrics saved to: {out_csv}")
    print(df.to_string())


if __name__ == "__main__":
    main()
