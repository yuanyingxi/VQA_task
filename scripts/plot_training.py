"""
Usage:
    python -m scripts.plot_training                                    # default: logs/vgg_lstm_concat
    python -m scripts.plot_training --log_dir logs/vgg_lstm_concat     # specific dir
    python -m scripts.plot_training --log_dir logs/other_model --output my_figures
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
import matplotlib.ticker as mticker

# ─── RC Params (clean research style) ──────────────────────────────
# Inspired by typical NeurIPS / ICML / CVPR paper figures:
# thin lines, white background, no top/right spines, readable fonts
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         12,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "legend.fontsize":   10,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#333333",
    "axes.linewidth":    0.8,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "legend.frameon":    True,
    "legend.framealpha": 0.85,
    "legend.edgecolor":  "#cccccc",
    "legend.fancybox":   False,
    "lines.linewidth":   1.2,
    "lines.markersize":  5,
})

# ─── Config ─────────────────────────────────────────────────────────
OUTPUT_DEFAULT = "figures"
FIGURE_SIZE = (12, 4.5)  # slightly wider + shorter for side-by-side
DPI = 250

# ColorBrewer Set1 (colorblind-friendly, distinct)
COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]

# Marker cycle (for multi-experiment clarity)
MARKERS = ["o", "s", "D", "^", "v"]


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


def _set_spines(ax):
    """Remove top and right spines; thin remaining ones."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_linewidth(0.8)


def plot_curves(exp_data: dict, output_dir: str, model_name: str = "vgg_lstm_concat"):
    """
    Generate side-by-side loss and VQA accuracy plots (research-paper style).

    Args:
        exp_data: dict of {experiment_label: [epoch_dict, ...]}
        output_dir: directory to save figures
        model_name: used in the figure title
    """
    if not exp_data:
        print("  No training data found. Nothing to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE)

    single = len(exp_data) == 1  # use clean labels if only one experiment

    for idx, (name, data) in enumerate(exp_data.items()):
        if not data:
            continue
        epochs = [d["epoch"] for d in data]
        c = COLORS[idx % len(COLORS)]
        m = MARKERS[idx % len(MARKERS)]

        # Skip markers if too many epochs (cluttered)
        use_marker = len(epochs) <= 15

        # Labels: clean "Train / Val" when single experiment
        lbl_train_loss = "Train" if single else f"{name} (Train)"
        lbl_val_loss   = "Val"   if single else f"{name} (Val)"
        lbl_train_acc  = "Train" if single else f"{name} (Train Acc)"
        lbl_val_vqa    = "Val"   if single else f"{name} (Val VQA)"

        # -- Loss (left) ------------------------------------------------
        ax1.plot(epochs, [d["train_loss"] for d in data],
                 linestyle="--", color=c, linewidth=1.0, alpha=0.6,
                 marker=m if use_marker else None, markersize=4,
                 label=lbl_train_loss)
        ax1.plot(epochs, [d["val_loss"] for d in data],
                 linestyle="-", color=c, linewidth=1.4,
                 marker=m if use_marker else None, markersize=4,
                 label=lbl_val_loss)

        # -- VQA Accuracy (right) ---------------------------------------
        ax2.plot(epochs, [d["train_acc"] for d in data],
                 linestyle="--", color=c, linewidth=1.0, alpha=0.6,
                 marker=m if use_marker else None, markersize=4,
                 label=lbl_train_acc)
        ax2.plot(epochs, [d["val_vqa"] for d in data],
                 linestyle="-", color=c, linewidth=1.4,
                 marker=m if use_marker else None, markersize=4,
                 label=lbl_val_vqa)

    # ====================== Loss plot ==================================
    _set_spines(ax1)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss", fontweight="normal", pad=10)
    ax1.set_xlim(left=0.5)
    handles1, labels1 = ax1.get_legend_handles_labels()
    if handles1:
        ax1.legend(handles1, labels1, framealpha=0.8, edgecolor="#dddddd",
                   loc="upper right")

    # ====================== Accuracy plot ==============================
    _set_spines(ax2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("VQA Accuracy", fontweight="normal", pad=10)
    ax2.set_xlim(left=0.5)

    # Set y-axis to percentage
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))

    # Lower y-axis bound with some padding
    all_vqa = [d["val_vqa"] for d_list in exp_data.values() for d in d_list]
    if all_vqa:
        bottom = max(0.0, min(all_vqa) - 0.04)
    else:
        bottom = 0.0
    ax2.set_ylim(bottom=bottom)

    handles2, labels2 = ax2.get_legend_handles_labels()
    if handles2:
        ax2.legend(handles2, labels2, framealpha=0.8, edgecolor="#dddddd",
                   loc="lower right")

    # ====================== Overall ====================================
    # Subtle suptitle (not bold, smaller)
    fig.suptitle(f"VizWiz-VQA  ·  {model_name}",
                 fontsize=13, fontweight="normal", y=1.02, color="#333333")
    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    for fmt in ["png", "svg"]:
        out_path = os.path.join(output_dir, f"training_curves.{fmt}")
        plt.savefig(out_path, dpi=DPI, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        size_kb = os.path.getsize(out_path) / 1024
        print(f"  Saved: {out_path} ({size_kb:.0f} KB)")

    plt.close()
    n_exp = len(exp_data)
    print(f"  -> Plotted {n_exp} experiment(s)")


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from log files")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Log directory (default: logs/<MODEL_NAME>)")
    parser.add_argument("--model_name", type=str, default="vgg_lstm_concat",
                        help="Model name (used to locate log dir and figure title)")
    parser.add_argument("--output", type=str, default=OUTPUT_DEFAULT,
                        help=f"Output directory (default: {OUTPUT_DEFAULT})")
    args = parser.parse_args()

    # Determine log directory
    project_root = Path(__file__).resolve().parent.parent
    if args.log_dir is None:
        log_dir = project_root / "logs" / args.model_name
    else:
        log_dir = Path(args.log_dir)

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
