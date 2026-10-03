import json
import matplotlib.pyplot as plt
import os

MODEL_DIR = "outputs"
PLOT_DIR = "plots"

os.makedirs(PLOT_DIR, exist_ok=True)

def plot_comparison():
    with open(os.path.join(MODEL_DIR, "baseline_results.json")) as f:
        results = json.load(f)

    models = [r["model"] for r in results]
    rmse   = [r["RMSE"] for r in results]
    mae    = [r["MAE"] for r in results]
    r2     = [r["R2"] for r in results]

    colors = ["red" if "GAT" in m else "blue" for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, vals, title in zip(
        axes,
        [rmse, mae, r2],
        ["RMSE ↓", "MAE ↓", "R² ↑"]
    ):
        bars = ax.bar(models, vals, color=colors)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{v:.4f}",
                    ha="center")

        ax.set_title(title)
        ax.tick_params(axis='x', rotation=15)

    plt.suptitle("UCMS-CCRP GAT vs Baselines (Test 2023–2025)")
    plt.tight_layout()

    path = os.path.join(PLOT_DIR, "baseline_comparison_fixed.png")
    plt.savefig(path, dpi=150)
    plt.close()

    print(f"✅ Plot saved → {path}")


if __name__ == "__main__":
    plot_comparison()