import json
import matplotlib.pyplot as plt
import os

def generate_robustness_curve(json_file='noise_results.json', save_path='robustness_curve.png'):
    if not os.path.exists(json_file):
        print(f"❌ Error: {json_file} not found. Wait for the experiment to finish.")
        return

    with open(json_file, 'r') as f:
        data = json.load(f)

    # Sort results by sigma to ensure the line plot is ordered
    data.sort(key=lambda x: x['sigma'])

    sigmas = [d['sigma'] for d in data]
    # Handle the fact that metrics might be nested or named differently
    f1_scores = [d.get('f1', d.get('accuracy', 0)) for d in data]
    
    # If the metrics were stored in the 'raw_metrics' sub-dictionary
    if all(f == 0 for f in f1_scores):
        f1_scores = [d.get('raw_metrics', {}).get('CULane/F1@50', 0) for d in data]

    # Create Plot
    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(sigmas, f1_scores, marker='o', color='#2c3e50', linewidth=2.5, markersize=8, label='$F_1$ Score')
    
    # Styling
    plt.title('CLRerNet Robustness under Gaussian Noise', fontsize=16, fontweight='bold')
    plt.xlabel('Noise Standard Deviation ($\sigma$)', fontsize=13)
    plt.ylabel('$F_1$ Score (CULane)', fontsize=13)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, max(f1_scores) * 1.1)  # Set y-limit with some padding
    
    # Add value labels on top of points
    for i, f1 in enumerate(f1_scores):
        plt.text(sigmas[i], f1 + 0.02, f"{f1:.3f}", ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Robustness graph saved to: {save_path}")

if __name__ == "__main__":
    generate_robustness_curve()