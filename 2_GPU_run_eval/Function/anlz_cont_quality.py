# entropy / repeat 趋势图
# entropy 上下限沿用 Ceval 的 9-21（token entropy），与 extend_metrics.calculate_token_entropy 对齐

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def analyze_entropy_and_repeat(analysis_data, save_fig_path=None):
    if not save_fig_path:
        raise ValueError("❌ save_fig_path is required")

    plt.figure(figsize=(16, 12))

    # Entropy
    plt.subplot(1, 2, 1)
    entropy_data = analysis_data.entropy.sort_values()
    plt.plot(entropy_data.values, "o", markersize=2, alpha=0.7, color="blue")
    step = max(1, len(analysis_data) // 10)
    plt.xticks(np.arange(0, len(analysis_data) + 1, step))
    plt.xlabel("Sample Index")
    plt.ylim(9, 21)
    plt.ylabel("Entropy")
    plt.title("Token Entropy (sorted ascending)")
    plt.grid(True, alpha=0.3)

    # Repeat
    plt.subplot(1, 2, 2)
    repeat_data = analysis_data.repeat.sort_values(ascending=False)
    plt.plot(repeat_data.values, "o", markersize=2, alpha=0.7, color="green")
    plt.xticks(np.arange(0, len(analysis_data) + 1, step))
    plt.xlabel("Sample Index")
    plt.ylim(-0.05, 1.05)
    plt.ylabel("Repeat Rate")
    plt.title("Repetition Rate (sorted descending)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
    plt.savefig(save_fig_path)
    print(f"[FIGURE SAVED] {save_fig_path}")
    plt.close()
