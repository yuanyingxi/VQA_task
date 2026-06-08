"""
Usage:
    python -m scripts.plot_training                                        # figures/vgg_lstm_concat/
    python -m scripts.plot_training --model_name new_model                 # figures/new_model/
    python -m scripts.plot_training --output my_figures                    # custom output dir
"""
import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# ─── Config ────────────────────────────────────────────────────────────
OUTPUT_DEFAULT = "figures"
FIGURE_SIZE = (14, 5.5)
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
DPI = 200

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def parse_training_log(log_path: str) -> list[dict]:
    """
    Parse a training log file and return per-epoch metrics.

    Expected log line format:
        Epoch  X | Train Loss X.XXXX | Train Acc X.XXXX | Val Loss X.XXXX | Val Acc X.XXXX | Val VQA X.XXXX | LR X.XXXXXX
    """
    epochs = []
    pattern = re.compile(
        r"Epoch\s+(\d+)\s+\|"
        r"\s+Train Loss\s+([\d.]+)\s+\|"
        r"\s+Train Acc\s+([\d.]+)\s+\|"
        r"\s+Val Loss\s+([\d.]+)\s+\|"
        r"\s+Val Acc\s+([\d.]+)\s+\|"
        r"\s+Val VQA\s+([\d.]+)"
    )
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epochs.append({
                    "epoch": int(m.group(1)),
                    "train_loss": float(m.group(2)),
                    "train_acc": float(m.group(3)),
                    "val_loss": float(m.group(4)),
                    "val_acc": float(m.group(5)),
                    "val_vqa": float(m.group(6)),
                })
    return epochs


def extract_experiment_label(log_path: str) -> str:
    """
    Extract a short experiment label from the log filename.
    Format: train_YYYYMMDD_HHMMSS.log -> YYYY-MM-DD HH:MM
    """
    basename = os.path.basename(log_path)
    m = re.search(r"train_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", basename)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}:{m.group(5)}"
    return basename.replace(".log", "").replace("train_", "")


def plot_curves(exp_data: dict, output_dir: str, model_name: str = "vgg_lstm_concat"):
    """
    Generate side-by-side loss and VQA accuracy plots.

    Args:
        exp_data: dict of {experiment_label: [epoch_dict, ...]}
        output_dir: directory to save figures
        model_name: used in the figure title
    """
    if not exp_data:
        print("  No training data found. Nothing to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE)

    for idx, (name, data) in enumerate(exp_data.items()):
        if not data:
            continue
        epochs = [d["epoch"] for d in data]
        c = COLORS[idx % len(COLORS)]

        # -- Loss (left) ------------------------------------------------
        ax1.plot(epochs, [d["train_loss"] for d in data],
                 linestyle="--", color=c, linewidth=1.5, alpha=0.7,
                 label=f"{name} (Train)")
        ax1.plot(epochs, [d["val_loss"] for d in data],
                 linestyle="-", color=c, linewidth=2,
                 label=f"{name} (Val)")

        # -- VQA Accuracy (right) ---------------------------------------
        ax2.plot(epochs, [d["train_acc"] for d in data],
                 linestyle="--", color=c, linewidth=1.5, alpha=0.7,
                 label=f"{name} (Train Acc)")
        ax2.plot(epochs, [d["val_vqa"] for d in data],
                 linestyle="-", color=c, linewidth=2,
                 label=f"{name} (Val VQA)")

    # -- Style Loss plot ------------------------------------------------
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Loss Curves", fontsize=14, fontweight="bold")
    if ax1.get_legend_handles_labels()[0]:
        ax1.legend(framealpha=0.9, edgecolor="#ccc")
    ax1.set_xlim(left=0.5)

    # -- Style Accuracy plot --------------------------------------------
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("VQA Accuracy Curves", fontsize=14, fontweight="bold")
    if ax2.get_legend_handles_labels()[0]:
        ax2.legend(framealpha=0.9, edgecolor="#ccc")
    ax2.set_xlim(left=0.5)
    # Compute y-axis lower bound
    all_vqa = [d["val_vqa"] for d_list in exp_data.values() for d in d_list]
    if all_vqa:
        ax2.set_ylim(bottom=max(0, min(all_vqa) - 0.05))

    # -- Overall title --------------------------------------------------
    fig.suptitle(f"VizWiz-VQA \u00b7 {model_name}",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    # -- Save -----------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    for fmt in ["png", "svg"]:
        out_path = os.path.join(output_dir, f"training_curves.{fmt}")
        plt.savefig(out_path, dpi=DPI, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        size_kb = os.path.getsize(out_path) / 1024
        print(f"  Saved: {out_path} ({size_kb:.0f} KB)")

    plt.close()
    print(f"  -> Plotted {len(exp_data)} experiment(s)")


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from log files")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Log directory (default: logs/<MODEL_NAME>)")
    parser.add_argument("--model_name", type=str, default="vgg_lstm_concat",
                        help="Model name (used to locate log dir and figure title)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: figures/<model_name>)")
    args = parser.parse_args()

    # Determine log directory
    project_root = Path(__file__).resolve().parent.parent
    if args.log_dir is None:
        log_dir = project_root / "logs" / args.model_name
    else:
        log_dir = Path(args.log_dir)

    # Default output: figures/<model_name>
    if args.output is None:
        args.output = str(project_root / OUTPUT_DEFAULT / args.model_name)

    if not log_dir.exists():
        print(f"  Log directory not found: {log_dir}")
        sys.exit(1)

    # Find all train_*.log files
    log_files = sorted(log_dir.glob("train_*.log"))
    if not log_files:
        print(f"  No train_*.log files found in {log_dir}")
        sys.exit(1)

    print(f"  Found {len(log_files)} log file(s) in {log_dir}")

    # Parse each log file
    exp_data = {}
    for log_file in log_files:
        label = extract_experiment_label(str(log_file))
        epochs = parse_training_log(str(log_file))
        if epochs:
            exp_data[label] = epochs
            print(f"  {log_file.name}: {len(epochs)} epochs")
        else:
            print(f"  {log_file.name}: no epoch data found (not a training log?)")

    # Plot
    plot_curves(exp_data, args.output, args.model_name)


if __name__ == "__main__":
    main()
