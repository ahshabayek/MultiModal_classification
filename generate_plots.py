#!/usr/bin/env python3
"""
Generate publication-quality plots from MLflow metrics for ViLBERT experiments.

Usage:
    python generate_plots.py

Output:
    - plots/model_comparison.pdf - Bar chart comparing all models
    - plots/training_curves.pdf - Loss and AUROC curves for key models
    - plots/freeze_comparison.pdf - Freezing experiment results
    - plots/hyperparameter_tuning.pdf - Batch size and label smoothing effects
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Use LaTeX-friendly settings
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.figsize": (6, 4),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

MLRUNS_DIR = Path("/home/ashabayek/Documents/Git/MultiModal_classification/mlruns/0")
OUTPUT_DIR = Path("/home/ashabayek/Documents/Git/MultiModal_classification/plots")


def read_metric(run_dir: Path, metric_name: str) -> List[Tuple[int, float]]:
    """Read metric values from MLflow run directory."""
    metric_file = run_dir / "metrics" / metric_name
    if not metric_file.exists():
        return []

    values = []
    with open(metric_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                timestamp, value, step = int(parts[0]), float(parts[1]), int(parts[2])
                values.append((step, value))
    return sorted(values, key=lambda x: x[0])


def get_run_name(run_dir: Path) -> str:
    """Get the MLflow run name."""
    name_file = run_dir / "tags" / "mlflow.runName"
    if name_file.exists():
        return name_file.read_text().strip()
    return run_dir.name


def get_final_metric(run_dir: Path, metric_name: str) -> float:
    """Get the final value of a metric."""
    values = read_metric(run_dir, metric_name)
    if values:
        return values[-1][1]
    return 0.0


def find_runs_by_auroc() -> Dict[str, Tuple[Path, float]]:
    """Find all runs and their test AUROC values."""
    runs = {}
    for run_dir in MLRUNS_DIR.iterdir():
        if run_dir.is_dir() and (run_dir / "metrics").exists():
            name = get_run_name(run_dir)
            auroc = get_final_metric(run_dir, "test_auroc")
            if auroc > 0:
                runs[name] = (run_dir, auroc)
    return runs


# Manual mapping of run names to model names (based on our experiments)
RUN_TO_MODEL = {
    # Best models
    "bright-robin-708": "LMDB (batch=16)",
    "luxuriant-toad-276": "LMDB (batch=16, v2)",
    "languid-finch-679": "LMDB (freeze=6)",
    # ROI experiments
    "enchanting-goose-4": "ROI (baseline)",
    "salty-quail-655": "ROI (batch=16)",
    "amazing-ant-350": "ROI (freeze=6)",
    # DINOv2 experiments
    "likeable-colt-592": "DINOv2 (baseline)",
    "dapper-worm-718": "DINOv2 (freeze=6)",
    # DINOv2-ML experiments
    "valuable-kit-666": "DINOv2-ML (ε=0.1)",
    "crawling-mule-799": "DINOv2-ML (batch=16)",
    "amazing-squirrel-372": "DINOv2-ML (freeze=6)",
    # Facebook baseline
    "abrasive-snipe-343": "Facebook Baseline",
}


def plot_model_comparison():
    """Create bar chart comparing all model configurations."""
    # Data from our experiments (final results)
    models = [
        ("LMDB\n(batch=16)", 0.7580, "tab:blue"),
        ("ROI", 0.7197, "tab:orange"),
        ("DINOv2-ML\n(ε=0.1)", 0.7171, "tab:green"),
        ("DINOv2", 0.7069, "tab:red"),
        ("Facebook\nBaseline", 0.7045, "tab:gray"),
        ("Grid\n(ResNet-152)", 0.6658, "tab:purple"),
        ("FRCNN\n(COCO)", 0.6334, "tab:brown"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    names = [m[0] for m in models]
    aurocs = [m[1] for m in models]
    colors = [m[2] for m in models]

    bars = ax.bar(
        range(len(models)), aurocs, color=colors, edgecolor="black", linewidth=0.5
    )

    # Add value labels on bars
    for bar, auroc in zip(bars, aurocs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{auroc:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Add baseline reference line
    ax.axhline(
        y=0.7045,
        color="red",
        linestyle="--",
        linewidth=1,
        label="Facebook Baseline (0.7045)",
    )

    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(names)
    ax.set_ylabel("Test AUROC")
    ax.set_title("ViLBERT Model Comparison on Hateful Memes")
    ax.set_ylim(0.60, 0.80)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.pdf")
    plt.savefig(OUTPUT_DIR / "model_comparison.png")
    print(f"Saved: {OUTPUT_DIR / 'model_comparison.pdf'}")
    plt.close()


def plot_training_curves():
    """Plot training curves for key models."""
    runs = find_runs_by_auroc()

    # Key runs to plot - all models from the results table
    # Use luxuriant-toad-276 for LMDB as it has better validation curves
    key_runs = {
        "luxuriant-toad-276": ("LMDB", "tab:blue"),
        "enchanting-goose-4": ("ROI Pool", "tab:orange"),
        "valuable-kit-666": ("DINOv2-ML", "tab:green"),
        "likeable-colt-592": ("DINOv2", "tab:red"),
        "nervous-shad-419": ("ResNet Grid", "tab:purple"),
        "fun-chimp-290": ("FRCNN R50", "tab:brown"),
        "selective-stoat-327": ("FRCNN R152", "tab:pink"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Plot training loss
    ax1 = axes[0]
    for run_name, (label, color) in key_runs.items():
        if run_name in runs:
            run_dir = runs[run_name][0]
            train_loss = read_metric(run_dir, "train_loss")
            if train_loss:
                epochs, losses = zip(*train_loss)
                ax1.plot(epochs, losses, label=label, color=color, linewidth=1.5)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss Over Epochs")
    ax1.legend()
    ax1.set_xlim(0, None)

    # Plot validation AUROC
    ax2 = axes[1]
    for run_name, (label, color) in key_runs.items():
        if run_name in runs:
            run_dir = runs[run_name][0]
            val_auroc = read_metric(run_dir, "val_auroc")
            if val_auroc:
                epochs, aurocs = zip(*val_auroc)
                ax2.plot(
                    epochs,
                    aurocs,
                    label=label,
                    color=color,
                    linewidth=1.5,
                    marker="o",
                    markersize=3,
                )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation AUROC")
    ax2.set_title("Validation AUROC Over Epochs")
    ax2.legend()
    ax2.set_xlim(0, None)
    ax2.set_ylim(0.55, 0.80)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.pdf")
    plt.savefig(OUTPUT_DIR / "training_curves.png")
    print(f"Saved: {OUTPUT_DIR / 'training_curves.pdf'}")
    plt.close()


def plot_training_loss_separate():
    """Plot training and validation loss for top 2 models (LMDB and ROI)."""
    runs = find_runs_by_auroc()

    # Only top 2 models with both train and val loss
    key_runs = {
        "luxuriant-toad-276": ("LMDB", "tab:blue"),
        "enchanting-goose-4": ("ROI Pool", "tab:orange"),
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for run_name, (label, color) in key_runs.items():
        if run_name in runs:
            run_dir = runs[run_name][0]
            train_loss = read_metric(run_dir, "train_loss")
            val_loss = read_metric(run_dir, "val_loss")
            if train_loss:
                epochs, losses = zip(*train_loss)
                ax.plot(
                    epochs,
                    losses,
                    label=f"{label} (train)",
                    color=color,
                    linewidth=1.5,
                    linestyle="-",
                )
            if val_loss:
                epochs, losses = zip(*val_loss)
                ax.plot(
                    epochs,
                    losses,
                    label=f"{label} (val)",
                    color=color,
                    linewidth=1.5,
                    linestyle="--",
                )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss (Top 2 Models)")
    ax.legend(loc="upper right")
    ax.set_xlim(0, None)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_loss.pdf")
    plt.savefig(OUTPUT_DIR / "training_loss.png")
    print(f"Saved: {OUTPUT_DIR / 'training_loss.pdf'}")
    plt.close()


def plot_validation_auroc_separate():
    """Plot validation AUROC as a separate figure with all models."""
    runs = find_runs_by_auroc()

    # Use luxuriant-toad-276 for LMDB as it has better validation curves
    key_runs = {
        "luxuriant-toad-276": ("LMDB", "tab:blue"),
        "enchanting-goose-4": ("ROI Pool", "tab:orange"),
        "valuable-kit-666": ("DINOv2-ML", "tab:green"),
        "likeable-colt-592": ("DINOv2", "tab:red"),
        "nervous-shad-419": ("ResNet Grid", "tab:purple"),
        "fun-chimp-290": ("FRCNN R50", "tab:brown"),
        "selective-stoat-327": ("FRCNN R152", "tab:pink"),
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for run_name, (label, color) in key_runs.items():
        if run_name in runs:
            run_dir = runs[run_name][0]
            val_auroc = read_metric(run_dir, "val_auroc")
            if val_auroc:
                epochs, aurocs = zip(*val_auroc)
                ax.plot(
                    epochs,
                    aurocs,
                    label=label,
                    color=color,
                    linewidth=1.5,
                    marker="o",
                    markersize=3,
                )

    # Add Facebook baseline reference line
    ax.axhline(
        y=0.7045,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="Facebook Baseline",
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation AUROC")
    ax.set_title("Validation AUROC Progress Across Feature Extractors")
    ax.legend(loc="lower right")
    ax.set_xlim(0, None)
    ax.set_ylim(0.55, 0.80)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "validation_auroc.pdf")
    plt.savefig(OUTPUT_DIR / "validation_auroc.png")
    print(f"Saved: {OUTPUT_DIR / 'validation_auroc.pdf'}")
    plt.close()


def plot_freeze_comparison():
    """Plot freezing experiment comparison."""
    # Data from freeze experiments
    models = ["LMDB", "ROI", "DINOv2", "DINOv2-ML"]
    baseline = [0.7580, 0.7197, 0.7069, 0.7171]
    frozen = [0.7577, 0.7020, 0.6940, 0.6905]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    bars1 = ax.bar(
        x - width / 2,
        baseline,
        width,
        label="Baseline (freeze=0)",
        color="tab:blue",
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        frozen,
        width,
        label="Frozen (freeze=6)",
        color="tab:orange",
        edgecolor="black",
        linewidth=0.5,
    )

    # Add value labels
    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{bar.get_height():.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{bar.get_height():.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Add change annotations
    changes = [-0.03, -1.77, -1.29, -2.66]
    for i, change in enumerate(changes):
        color = "green" if change > -0.5 else "red"
        ax.annotate(
            f"{change:+.2f}%",
            xy=(x[i] + width / 2, frozen[i] - 0.015),
            ha="center",
            fontsize=9,
            color=color,
            fontweight="bold",
        )

    ax.set_ylabel("Test AUROC")
    ax.set_title("Effect of Freezing First 6 BERT Layers")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(loc="upper right")
    ax.set_ylim(0.65, 0.80)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "freeze_comparison.pdf")
    plt.savefig(OUTPUT_DIR / "freeze_comparison.png")
    print(f"Saved: {OUTPUT_DIR / 'freeze_comparison.pdf'}")
    plt.close()


def plot_hyperparameter_tuning():
    """Plot hyperparameter tuning results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Batch size comparison
    ax1 = axes[0]
    models = ["LMDB", "ROI", "DINOv2-ML"]
    batch32 = [0.7433, 0.7197, 0.7141]
    batch16 = [0.7580, 0.7180, 0.7141]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax1.bar(
        x - width / 2,
        batch32,
        width,
        label="Batch=32",
        color="tab:blue",
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax1.bar(
        x + width / 2,
        batch16,
        width,
        label="Batch=16",
        color="tab:orange",
        edgecolor="black",
        linewidth=0.5,
    )

    for bar in bars1:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{bar.get_height():.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar in bars2:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{bar.get_height():.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax1.set_ylabel("Test AUROC")
    ax1.set_title("Effect of Batch Size")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.set_ylim(0.68, 0.78)

    # Label smoothing (DINOv2-ML only)
    ax2 = axes[1]
    smoothing = ["0.0", "0.1", "0.2"]
    aurocs = [0.7056, 0.7171, 0.7141]
    colors = ["tab:blue", "tab:green", "tab:orange"]

    bars = ax2.bar(smoothing, aurocs, color=colors, edgecolor="black", linewidth=0.5)

    for bar, auroc in zip(bars, aurocs):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{auroc:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Highlight best
    bars[1].set_edgecolor("red")
    bars[1].set_linewidth(2)

    ax2.set_xlabel("Label Smoothing (ε)")
    ax2.set_ylabel("Test AUROC")
    ax2.set_title("Label Smoothing Effect (DINOv2-ML)")
    ax2.set_ylim(0.68, 0.74)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hyperparameter_tuning.pdf")
    plt.savefig(OUTPUT_DIR / "hyperparameter_tuning.png")
    print(f"Saved: {OUTPUT_DIR / 'hyperparameter_tuning.pdf'}")
    plt.close()


def plot_feature_extractor_breakdown():
    """Create a detailed breakdown of feature extractors."""
    # Grouped by feature extraction approach
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = {
        "Precomputed\n(LMDB)": [("ResNeXt-152 VG", 0.7580, "tab:blue")],
        "On-the-fly\nROI Pooling": [
            ("ResNet-152 ROI", 0.7197, "tab:orange"),
            ("FRCNN ResNet-152", 0.6334, "tab:red"),
            ("FRCNN ResNet-50", 0.6472, "tab:pink"),
        ],
        "On-the-fly\nGrid Features": [
            ("DINOv2-ML", 0.7171, "tab:green"),
            ("DINOv2", 0.7069, "tab:olive"),
            ("ResNet-152 Grid", 0.6658, "tab:cyan"),
        ],
        "Detection\nBased": [
            ("VG RPN", 0.6417, "tab:purple"),
            ("VG Grid", 0.6367, "tab:brown"),
        ],
    }

    # Flatten for plotting
    all_models = []
    all_aurocs = []
    all_colors = []
    group_positions = []
    group_labels = []

    pos = 0
    for group_name, models in categories.items():
        group_start = pos
        for model_name, auroc, color in models:
            all_models.append(model_name)
            all_aurocs.append(auroc)
            all_colors.append(color)
            pos += 1
        group_positions.append((group_start + pos - 1) / 2)
        group_labels.append(group_name)
        pos += 0.5  # Gap between groups

    # Plot
    x_pos = []
    current = 0
    for i, group in enumerate(categories.values()):
        for _ in group:
            x_pos.append(current)
            current += 1
        current += 0.5

    bars = ax.bar(x_pos, all_aurocs, color=all_colors, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, auroc in zip(bars, all_aurocs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{auroc:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=45,
        )

    # Baseline line
    ax.axhline(
        y=0.7045,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Facebook Baseline (0.7045)",
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_models, rotation=45, ha="right")
    ax.set_ylabel("Test AUROC")
    ax.set_title("Feature Extractor Comparison by Category")
    ax.set_ylim(0.60, 0.80)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_extractor_breakdown.pdf")
    plt.savefig(OUTPUT_DIR / "feature_extractor_breakdown.png")
    print(f"Saved: {OUTPUT_DIR / 'feature_extractor_breakdown.pdf'}")
    plt.close()


def main():
    """Generate all plots."""
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating plots from MLflow metrics...")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Generate each plot
    plot_model_comparison()
    plot_training_curves()
    plot_training_loss_separate()
    plot_validation_auroc_separate()
    plot_freeze_comparison()
    plot_hyperparameter_tuning()
    plot_feature_extractor_breakdown()

    print()
    print("All plots generated successfully!")
    print(f"Files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
