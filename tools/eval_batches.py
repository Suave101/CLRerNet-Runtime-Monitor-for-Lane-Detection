"""
Batch evaluation script using the CLRerNet runner pipeline (Approach 2).

For each set of image paths found in experiment JSON log files, this script:
  1. Detects whether the images are from the CULane or Curvelanes dataset by
     inspecting the image paths.
  2. Writes a temporary CULane-format list file containing only those paths,
     using the detected dataset root so the loader never doubles the prefix.
  3. Overrides the test_dataloader and test_evaluator in the config to use
     that list, then calls runner.test() to obtain official IoU-based
     CULane metrics (F1, precision, recall).
  4. Collects all results into a single CSV.

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
from pathlib import Path

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
    return parser.parse_args()


def main():
    args = parse_args()

    cfg_path = os.path.abspath(args.config)
    checkpoint = os.path.abspath(args.checkpoint)
    data_root = os.path.abspath(args.data_root)
    logs_dir = os.path.abspath(args.logs_dir)
    out_csv = os.path.abspath(args.out_csv)

    json_files = sorted(Path(logs_dir).glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {logs_dir}")
        return

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
                    # Single run, not a list
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

                    # Run official CULane evaluation via runner.test()
                    metrics = _run_batch(cfg_path, checkpoint, data_root, image_paths)

                    # Primary IoU threshold is 0.5 (matches CULane convention).
                    row = {
                        **args_fields,
                        "log_file": filename,
                        "run_type": run_type,
                        "run_idx": run.get("Run"),
                        "seed": run.get("Seed"),
                        "images_in_sample": len(image_paths),
                    # Official CULane metrics at IoU 0.5.
                    # Key names come directly from CULaneMetric.compute_metrics /
                    # eval_predictions: TP/FP/FN/Precision/Recall use no separator
                    # (e.g. "Precision0.5") while F1 uses one ("F1_0.5").  This is
                    # the upstream convention and is intentional here.
                        "f1_score": metrics.get("F1_0.5"),
                        "precision": metrics.get("Precision0.5"),
                        "recall": metrics.get("Recall0.5"),
                        "tp": metrics.get("TP0.5"),
                        "fp": metrics.get("FP0.5"),
                        "fn": metrics.get("FN0.5"),
                        # Optional: also expose IoU 0.1 and 0.75 columns
                        "f1_score_iou0.1": metrics.get("F1_0.1"),
                        "f1_score_iou0.75": metrics.get("F1_0.75"),
                        # Distribution-shift test statistics from the JSON
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
