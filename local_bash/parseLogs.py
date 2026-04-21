import os
import re
import csv
import glob
import json
import argparse

def parse_logs_to_csv(log_dir, output_csv, target_iou="0.5"):
    # --- REGEX PATTERNS ---
    data_list_pattern = re.compile(r"data_list=\s*['\"]([^'\"]+)['\"]")
    path_pattern = re.compile(r"_(\d+)/Run_(\d+)\.txt")
    
    # Matches the specific line format you provided
    iou_header_pattern = re.compile(r"Evaluation results for IoU threshold = ([0-9.]+)")
    metrics_line_pattern = re.compile(
        r"Eval category:\s+test_all\s+,\s+N:\s*(\d+),\s+TP:\s*(\d+),\s+FP:\s*(\d+),\s+FN:\s*(\d+),\s+Precision:\s*([0-9.]+),\s+Recall:\s*([0-9.]+),\s+F1:\s*([0-9.]+)"
    )

    log_files = glob.glob(os.path.join(log_dir, "**", "*.log"), recursive=True)
    print(f"🔍 Found {len(log_files)} log files. Processing...")

    all_rows = []
    proxy_metric_keys = set()
    json_cache = {}

    for log_path in log_files:
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception:
            continue

        # 1. Identity Check
        data_list_match = data_list_pattern.search(content)
        if not data_list_match: continue
        
        data_path = data_list_match.group(1)
        path_match = path_pattern.search(data_path)
        if not path_match: continue
            
        exp_id = int(path_match.group(1))
        run_id = int(path_match.group(2))

        # 2. Extract DL Metrics for Target IoU
        iou_sections = iou_header_pattern.split(content)
        row_data = None

        for i in range(1, len(iou_sections), 2):
            if iou_sections[i].strip() == target_iou:
                match = metrics_line_pattern.search(iou_sections[i+1])
                if match:
                    n, tp, fp = int(match.group(1)), int(match.group(2)), int(match.group(3))
                    row_data = {
                        'block_idx': exp_id - 1,
                        'run_idx': run_id,
                        'N': n,
                        'TP': tp,
                        'FP': fp,
                        'FN': int(match.group(4)),
                        'avg_lanes_detected': round((tp + fp) / n, 4) if n > 0 else 0,
                        'precision': float(match.group(5)),
                        'recall': float(match.group(6)),
                        'f1_score': float(match.group(7))
                    }
                break

        if not row_data:
            continue

        # 3. Extract Proxy Metrics from JSON
        try:
            # Reconstruct JSON path: /path/to/Folder/_7/Run_1.txt -> /path/to/Folder.json
            folder_path = os.path.dirname(os.path.dirname(data_path))
            json_path = folder_path + ".json"

            if os.path.exists(json_path):
                if json_path not in json_cache:
                    with open(json_path, 'r') as jf:
                        json_cache[json_path] = json.load(jf)
                
                json_data = json_cache[json_path]
                exp_list = json_data.get("experiments", [])
                if exp_id <= len(exp_list):
                    runs = exp_list[exp_id-1].get("data", {}).get("Data Shift Test Data", {}).get("Individual Test Data", [])
                    target_run = next((r for r in runs if r.get("Run") == run_id), None)
                    
                    if target_run and "Results" in target_run:
                        for test_name, metrics in target_run["Results"].items():
                            for m_name, m_val in metrics.items():
                                col = f"{test_name.lower()}_{m_name.lower().replace(' ', '_').replace('-', '_')}"
                                row_data[col] = m_val
                                proxy_metric_keys.add(col)
        except Exception:
            pass

        all_rows.append(row_data)

    if not all_rows:
        print("❌ No data found.")
        return

    # Sort results for a clean table
    all_rows.sort(key=lambda x: (x['block_idx'], x['run_idx']))

    # --- 4. WRITE CSV WITH REQUESTED COLUMN ORDER ---
    # Define fixed order for the front columns
    fieldnames = [
        'block_idx', 'run_idx', 'N', 'TP', 'FP', 'FN', 
        'avg_lanes_detected', 'precision', 'recall', 'f1_score'
    ]
    # Append the statistical proxy metrics discovered (alphabetically)
    fieldnames += sorted(list(proxy_metric_keys))

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"✅ SUCCESS: Created {output_csv} with {len(all_rows)} entries.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iou", default="0.5")
    args = parser.parse_args()
    parse_logs_to_csv(args.log_dir, args.out, args.iou)
