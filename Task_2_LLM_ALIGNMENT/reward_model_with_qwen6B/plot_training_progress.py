"""
Visualize training progress for reward model training
Reads TensorBoard logs and creates plots for loss, accuracy, and other metrics
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_logs(log_dir="./reward_model_output"):
    """Load training metrics from TensorBoard logs"""
    print(f"Loading logs from {log_dir}...")

    # Check if log_dir exists
    if not os.path.exists(log_dir):
        print(f"Error: Log directory {log_dir} not found")
        return None

    # Check for nested runs directory structure
    runs_dir = os.path.join(log_dir, "runs")
    if os.path.exists(runs_dir):
        # Find the latest run directory
        run_dirs = [d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
        if run_dirs:
            run_dirs.sort()  # Sort to get latest
            actual_log_dir = os.path.join(runs_dir, run_dirs[-1])
            print(f"Found runs directory, using: {actual_log_dir}")
            log_dir = actual_log_dir

    try:
        # Initialize EventAccumulator
        ea = EventAccumulator(log_dir)
        ea.Reload()

        # Get available tags
        scalar_tags = ea.Tags()["scalars"]
        print(f"Available scalar tags: {scalar_tags}")

        if not scalar_tags:
            print("Warning: No scalar metrics found")
            return None

        return ea
    except Exception as e:
        print(f"Error loading logs: {e}")
        return None


def extract_metrics(ea):
    """Extract metrics from EventAccumulator"""
    metrics = {}
    scalar_tags = ea.Tags()["scalars"]

    for tag in scalar_tags:
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        metrics[tag] = {"steps": steps, "values": values}

    return metrics


def plot_loss_and_accuracy(metrics, save_path="./reward_model_output"):
    """Plot loss and accuracy over training steps"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Reward Model Training Progress", fontsize=16, fontweight="bold")

    # Helper function to find metric (with or without train/ prefix)
    def get_metric(metrics, name):
        if name in metrics:
            return metrics[name]
        elif f"train/{name}" in metrics:
            return metrics[f"train/{name}"]
        return None

    # Plot 1: Loss
    loss_metric = get_metric(metrics, "loss")
    if loss_metric:
        ax = axes[0, 0]
        steps = loss_metric["steps"]
        values = loss_metric["values"]
        ax.plot(steps, values, linewidth=2, color="red", label="Training Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Loss over Steps")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Plot 2: Accuracy
    acc_metric = get_metric(metrics, "accuracy")
    if acc_metric:
        ax = axes[0, 1]
        steps = acc_metric["steps"]
        values = acc_metric["values"]
        ax.plot(steps, values, linewidth=2, color="green", label="Accuracy")
        ax.set_xlabel("Step")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy over Steps")
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        ax.legend()

    # Plot 3: Margin (difference between chosen and rejected rewards)
    margin_metric = get_metric(metrics, "margin")
    if margin_metric:
        ax = axes[1, 0]
        steps = margin_metric["steps"]
        values = margin_metric["values"]
        ax.plot(steps, values, linewidth=2, color="blue", label="Margin")
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax.set_xlabel("Step")
        ax.set_ylabel("Margin (chosen - rejected)")
        ax.set_title("Reward Margin over Steps")
        ax.grid(True, alpha=0.3)
        ax.legend()

    # Plot 4: Mean Reward
    mean_reward_metric = get_metric(metrics, "mean_reward")
    if mean_reward_metric:
        ax = axes[1, 1]
        steps = mean_reward_metric["steps"]
        values = mean_reward_metric["values"]
        ax.plot(steps, values, linewidth=2, color="purple", label="Mean Reward")
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean Reward")
        ax.set_title("Mean Reward over Steps")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(save_path, "training_progress.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    plt.show()


def plot_reward_statistics(metrics, save_path="./reward_model_output"):
    """Plot min, mean, max reward statistics"""

    # Helper function to find metric (with or without train/ prefix)
    def get_metric(metrics, name):
        if name in metrics:
            return metrics[name]
        elif f"train/{name}" in metrics:
            return metrics[f"train/{name}"]
        return None

    min_reward = get_metric(metrics, "min_reward")
    max_reward = get_metric(metrics, "max_reward")
    mean_reward = get_metric(metrics, "mean_reward")

    if not min_reward and not max_reward:
        print("No reward statistics found in logs")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract steps from any metric
    steps = None
    if min_reward:
        steps = min_reward["steps"]
    elif max_reward:
        steps = max_reward["steps"]
    elif mean_reward:
        steps = mean_reward["steps"]

    if steps is None:
        print("Could not find steps")
        return

    # Plot min, mean, max
    if min_reward and max_reward:
        ax.fill_between(
            steps,
            min_reward["values"],
            max_reward["values"],
            alpha=0.3,
            label="Min-Max Range",
            color="lightblue",
        )

    if mean_reward:
        ax.plot(
            mean_reward["steps"],
            mean_reward["values"],
            linewidth=2.5,
            label="Mean Reward",
            color="darkblue",
        )

    if min_reward:
        ax.plot(
            min_reward["steps"],
            min_reward["values"],
            linewidth=1.5,
            label="Min Reward",
            color="red",
            linestyle="--",
        )

    if max_reward:
        ax.plot(
            max_reward["steps"],
            max_reward["values"],
            linewidth=1.5,
            label="Max Reward",
            color="green",
            linestyle="--",
        )

    ax.axhline(y=0, color="k", linestyle=":", alpha=0.5)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Reward Value", fontsize=12)
    ax.set_title("Reward Statistics over Training", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(save_path, "reward_statistics.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    plt.show()


def save_metrics_summary(metrics, save_path="./reward_model_output"):
    """Save metrics summary to CSV"""
    import csv

    # Find common steps across all metrics
    all_steps = set()
    for metric_name, metric_data in metrics.items():
        all_steps.update(metric_data["steps"])

    all_steps = sorted(list(all_steps))

    # Create summary
    summary = {"step": all_steps}
    for metric_name, metric_data in metrics.items():
        # Create a mapping of steps to values
        step_to_value = dict(zip(metric_data["steps"], metric_data["values"]))
        summary[metric_name] = [step_to_value.get(step, None) for step in all_steps]

    # Save to CSV
    output_file = os.path.join(save_path, "training_metrics.csv")
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(summary.keys())
        for i in range(len(all_steps)):
            row = [summary[key][i] for key in summary.keys()]
            writer.writerow(row)

    print(f"Saved metrics summary to {output_file}")


def print_metrics_summary(metrics):
    """Print summary statistics of training metrics"""
    print("\n" + "=" * 80)
    print("TRAINING METRICS SUMMARY")
    print("=" * 80)

    for metric_name, metric_data in metrics.items():
        values = np.array(metric_data["values"])
        steps = metric_data["steps"]

        print(f"\n{metric_name}:")
        print(f"  Steps: {len(steps)} (from {steps[0]} to {steps[-1]})")
        print(f"  Min:   {np.min(values):.6f}")
        print(f"  Max:   {np.max(values):.6f}")
        print(f"  Mean:  {np.mean(values):.6f}")
        print(f"  Std:   {np.std(values):.6f}")
        if len(values) > 1:
            print(f"  Final: {values[-1]:.6f} (change: {values[-1] - values[0]:+.6f})")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Plot training progress")
    parser.add_argument(
        "--log-dir",
        default="./reward_model_output",
        help="Path to training output directory with TensorBoard logs",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots (requires GUI)",
    )
    args = parser.parse_args()

    # Load logs
    ea = load_tensorboard_logs(args.log_dir)
    if ea is None:
        return

    # Extract metrics
    metrics = extract_metrics(ea)

    if not metrics:
        print("No metrics found in logs")
        return

    # Print summary
    print_metrics_summary(metrics)

    # Create plots
    print("\nGenerating visualizations...")
    plot_loss_and_accuracy(metrics, args.log_dir)
    plot_reward_statistics(metrics, args.log_dir)

    # Save metrics summary
    save_metrics_summary(metrics, args.log_dir)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {args.log_dir}")
    print("  - training_progress.png (loss, accuracy, margin, mean_reward)")
    print("  - reward_statistics.png (min/mean/max reward)")
    print("  - training_metrics.csv (all metrics)")


if __name__ == "__main__":
    main()
