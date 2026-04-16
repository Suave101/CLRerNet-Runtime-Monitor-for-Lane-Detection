"""
Batch evaluation script with two-pass JSON processing and persistent image cache.

Workflow:
  1. Scan all JSON files and collect all image paths.
  2. Deduplicate image paths globally.
  3. Evaluate each unique image once (in parallel across GPUs), and cache the
     per-image metrics to disk.
  4. Scan all JSON files again and compute per-run average metrics from the
     cache, then write rows to a CSV.

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_image_path(path: str, data_root: str) -> str:
    """Normalize cache key for an image path."""
    path = path.strip()
    if not path:
        return ""
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(Path(data_root).joinpath(path).resolve())


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _detect_dataset_root(image_paths: list[str], culane_root: str) -> tuple[str, str]:
    """
    Detect whether *image_paths* belong to CULane or Curvelanes and return the
    correct dataset root to use when building the runner.

    The CULane dataset loader always prepends ``data_root`` to every list
    entry, so the root used here must match the prefix of the actual files on
    disk.  Passing the wrong root causes a doubled path like::

        /data/CULane/home/user/Curvelanes/…/img.jpg   ← broken

    Detection rules (evaluated in order):

    1. If every path is an absolute path whose prefix matches *culane_root*
       → ``("culane_root", "CULane")``
    2. If any path contains the substring "urvelane" (matches "Curvelanes"
       or "curvelanes" case-insensitively)
       → ``("/", "Curvelanes")``
    3. Fallback for any other paths that live outside *culane_root*
       → ``("/", "external")``, so that ``Path("/").joinpath(path.lstrip("/"))``
       reconstructs the original absolute path correctly.

    Parameters
    ----------
    image_paths:
        List of image file paths from the experiment JSON.
    culane_root:
        Absolute path to the CULane dataset root (from ``--data-root``).

    Returns
    -------
    (effective_root, dataset_label)
        *effective_root* must be used for both ``test_dataloader.dataset.data_root``
        and ``test_evaluator.data_root`` in the runner config.
        *dataset_label* is a human-readable string for logging only.
    """
    stripped = [p.strip() for p in image_paths if p.strip()]
    if not stripped:
        return culane_root, "CULane"

    culane_root_path = Path(culane_root)
    culane_parts = culane_root_path.parts

    all_culane = all(
        Path(p).is_absolute()
        and Path(p).parts[: len(culane_parts)] == culane_parts
        for p in stripped
    )
    if all_culane:
        return culane_root, "CULane"

    has_curvelanes = any("urvelane" in p for p in stripped)
    if has_curvelanes:
        return "/", "Curvelanes"

    return "/", "external"


def _make_list_file(image_paths: list[str], effective_root: str) -> str:
    """
    Write a temporary CULane-format list file.

    Each line is the image path *relative* to *effective_root*, prefixed with
    '/'.  The file is placed in a temporary directory that persists until the
    caller deletes it (or the process exits).

    *effective_root* must be the value returned by :func:`_detect_dataset_root`
    so that the CULane dataset loader's path construction matches the actual
    location of the files on disk.

    Returns the path to the written file.
    """
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="culane_batch_"
    )
    for img_path in image_paths:
        img_path = img_path.strip()
        if not img_path:
            continue
        # Normalize to a path relative to effective_root, prefixed with '/'.
        try:
            rel = "/" + str(Path(img_path).relative_to(effective_root))
        except ValueError:
            rel = img_path if img_path.startswith("/") else "/" + img_path
        tmp.write(rel + "\n")
    tmp.close()
    return tmp.name


def _build_runner(
    cfg_path: str,
    checkpoint: str,
    effective_root: str,
    list_file: str,
    culane_root: str,
    ) -> Runner:
    """
    Build an mmengine Runner whose test split is limited to *list_file*.
    A fresh Runner is built each time so that the internal mmengine state
    (results cache, evaluator buffers) is clean.

    *effective_root* is the root used when writing *list_file* and is set on
    the dataloader dataset so its path construction matches the list entries.

    *culane_root* is always used for the evaluator's ``data_root`` so that the
    CULane category split files (``list/test_split/``) can be found, regardless
    of whether the images originate from CULane or another dataset.
    """
    cfg = Config.fromfile(cfg_path)

    # Point the dataloader at the custom list using the dataset-specific root.
    cfg.test_dataloader.dataset.data_root = effective_root
    cfg.test_dataloader.dataset.data_list = list_file

    # The evaluator always uses the CULane root so it can locate the category
    # split files (list/test_split/*.txt) which only exist there.
    cfg.test_evaluator.data_root = culane_root
    cfg.test_evaluator.data_list = list_file

    cfg.load_from = checkpoint

    if "work_dir" not in cfg or cfg.work_dir is None:
        cfg.work_dir = "./work_dirs/evaluate_batches"

    # Disable visualization hooks that are not needed here.
    cfg.default_hooks = cfg.get("default_hooks", {})
    cfg.default_hooks.pop("visualization", None)

    # Disable the auxiliary segmentation loss/decoder during evaluation.
    # These components are only used during training; the pretrained checkpoint
    # does not include their weights, which produces spurious "missing keys"
    # warnings when the model is built with loss_weight > 0.
    # KeyError: loss_seg key absent; AttributeError: bbox_head absent or not a ConfigDict.
    try:
        cfg.model.bbox_head.loss_seg.loss_weight = 0
    except (AttributeError, KeyError):
        # Config does not have loss_seg; nothing to disable.
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
    data_root: str,
    image_paths: list[str],
) -> dict:
    """
    Evaluate a single batch of images.

    Detects whether the images are from CULane or Curvelanes by inspecting
    their paths and sets the dataset root accordingly before building the
    runner.  This prevents the CULane dataset loader from doubling the path
    prefix when the images live outside *data_root*.

    Returns a dict with the keys produced by CULaneMetric.compute_metrics,
    e.g. F1_0.5, Precision0.5, Recall0.5, TP0.5, FP0.5, FN0.5 …
    Returns an empty dict if image_paths is empty or if evaluation fails.
    """
    if not image_paths:
        return {}

    _force_sequential_culane_eval()

    effective_root, dataset_label = _detect_dataset_root(image_paths, data_root)
    print(f"  [Dataset] Detected: {dataset_label} (effective_root={effective_root!r})")

    list_file = _make_list_file(image_paths, effective_root)
    try:
        runner = _build_runner(
            cfg_path, checkpoint, effective_root, list_file, culane_root=data_root
        )
        metrics = runner.test()  # returns the dict from compute_metrics
        return metrics if metrics else {}
    finally:
        os.unlink(list_file)


def _load_metrics_cache(cache_path: str) -> dict[str, dict[str, Any]]:
    if not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {
                str(k): v
                for k, v in data.items()
                if isinstance(v, dict)
            }
    except Exception as exc:
        print(f"[WARN] Could not read cache {cache_path}: {exc}")
    return {}


def _save_metrics_cache(cache_path: str, cache: dict[str, dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    tmp = f"{cache_path}.tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    os.replace(tmp, cache_path)


def _iter_json_runs(json_files: list[Path]):
    """Yield run records found in JSON logs."""
    for json_path in json_files:
        filename = json_path.stem
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
                            "Image Paths": exp["data"][key].get("Image Paths", []),
                            "Run": -1,
                            "Seed": "N/A",
                            "Results": exp["data"][key],
                        }
                    ]
                else:
                    runs = exp["data"][key].get("Individual Test Data", [])

                for run in runs:
                    image_paths = run.get("Image Paths", [])
                    if not isinstance(image_paths, list):
                        image_paths = []
                    yield {
                        "log_file": filename,
                        "run_type": run_type,
                        "run_idx": run.get("Run"),
                        "seed": run.get("Seed"),
                        "args_fields": args_fields,
                        "results": run.get("Results", {}),
                        "image_paths": image_paths,
                    }


def _average_run_metrics(
    image_paths: list[str],
    cache: dict[str, dict[str, Any]],
    data_root: str,
) -> dict[str, float | None]:
    per_image_metrics: list[dict[str, Any]] = []
    for image_path in image_paths:
        key = _normalize_image_path(image_path, data_root)
        if not key:
            continue
        m = cache.get(key)
        if isinstance(m, dict):
            per_image_metrics.append(m)

    if not per_image_metrics:
        return {}

    metric_keys = set()
    for m in per_image_metrics:
        metric_keys.update(k for k, v in m.items() if _is_number(v))

    out: dict[str, float | None] = {}
    for key in sorted(metric_keys):
        vals = [float(m[key]) for m in per_image_metrics if _is_number(m.get(key))]
        out[key] = _safe_mean(vals)
    return out


def _evaluate_image_worker(
    cfg_path: str,
    checkpoint: str,
    data_root: str,
    image_path: str,
    gpu_id: int,
) -> tuple[str, dict[str, Any], str | None]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    try:
        metrics = _run_batch(cfg_path, checkpoint, data_root, [image_path])
        return image_path, metrics if isinstance(metrics, dict) else {}, None
    except Exception as exc:
        return image_path, {}, f"{type(exc).__name__}: {exc}"


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
        help="CULane dataset root directory (contains list/, driver_* …)",
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
        args.cache_file if args.cache_file else os.path.join(logs_dir, "image_metrics_cache.json")
    )
    num_gpus = max(1, int(args.num_gpus))

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
    for run in _iter_json_runs(json_files):
        image_paths = run["image_paths"]
        res = run["results"]
        avg_metrics = _average_run_metrics(image_paths, metrics_cache, data_root)

        row = {
            **run["args_fields"],
            "log_file": run["log_file"],
            "run_type": run["run_type"],
            "run_idx": run["run_idx"],
            "seed": run["seed"],
            "images_in_sample": len(image_paths),
            "cached_images_used": sum(
                1
                for p in image_paths
                if _normalize_image_path(p, data_root) in metrics_cache
            ),
            # Averaged IoU-based metrics (primary IoU=0.5).
            "f1_score": avg_metrics.get("F1_0.5"),
            "precision": avg_metrics.get("Precision0.5"),
            "recall": avg_metrics.get("Recall0.5"),
            "accuracy": avg_metrics.get("Accuracy0.5", avg_metrics.get("Acc0.5")),
            "tp": avg_metrics.get("TP0.5"),
            "fp": avg_metrics.get("FP0.5"),
            "fn": avg_metrics.get("FN0.5"),
            "f1_score_iou0.1": avg_metrics.get("F1_0.1"),
            "f1_score_iou0.75": avg_metrics.get("F1_0.75"),
            # Distribution-shift test statistics from the JSON.
            "mmd_stat": res.get("MMD", {}).get("Stat"),
            "mmd_p_value": res.get("MMD", {}).get("P-Value"),
            "mmd_shift_detected": res.get("MMD", {}).get("Shift Detected"),
        }

        # Also persist all averaged numeric metric keys for "etc." analysis.
        for key, value in avg_metrics.items():
            row[f"avg_{key}"] = value
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
