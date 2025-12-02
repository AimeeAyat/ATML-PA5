"""
Visualize PPO training progress
Reads TensorBoard logs and creates plots for loss, rewards, KL divergence, and other metrics
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_logs(log_dir="./ppo_training_output/logs"):
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


def plot_loss_and_rewards(metrics, save_path="./ppo_training_output"):
    """Plot training loss, learning rate, rewards, KL divergence, and entropy"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle("PPO Training Progress", fontsize=16, fontweight="bold")

    # Helper function to find metric (with or without prefix)
    def get_metric(metrics, name):
        if name in metrics:
            return metrics[name]
        elif f"train/{name}" in metrics:
            return metrics[f"train/{name}"]
        return None

    # Plot 1: Value Loss
    value_loss = get_metric(metrics, "train/loss/value_avg")
    if value_loss and value_loss["steps"]:
        ax = axes[0, 0]
        steps = value_loss["steps"]
        values = value_loss["values"]
        ax.plot(steps, values, linewidth=2, color="red", label="Value Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Value Loss over Steps")
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax = axes[0, 0]
        ax.text(0.5, 0.5, "Value Loss metric not found", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Plot 2: Learning Rate
    lr_metric = get_metric(metrics, "train/lr")
    if lr_metric and lr_metric["steps"]:
        ax = axes[0, 1]
        steps = lr_metric["steps"]
        values = lr_metric["values"]
        ax.plot(steps, values, linewidth=2, color="green", label="Learning Rate")
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax = axes[0, 1]
        ax.text(0.5, 0.5, "Learning rate metric not found", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Plot 3: KL Divergence (PPO-specific metric)
    kl_metric = get_metric(metrics, "train/objective/kl")
    if kl_metric and kl_metric["steps"]:
        ax = axes[1, 0]
        steps = kl_metric["steps"]
        values = kl_metric["values"]
        ax.plot(steps, values, linewidth=2, color="purple", label="KL Divergence")
        ax.set_xlabel("Step")
        ax.set_ylabel("KL Divergence")
        ax.set_title("KL Divergence from Reference Policy")
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax = axes[1, 0]
        ax.text(0.5, 0.5, "KL divergence metric not found", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Plot 4: RLHF Reward & Scores
    rlhf_reward = get_metric(metrics, "train/objective/rlhf_reward")
    scores = get_metric(metrics, "train/objective/scores")
    if rlhf_reward and rlhf_reward["steps"]:
        ax = axes[1, 1]
        steps = rlhf_reward["steps"]
        values = rlhf_reward["values"]
        ax.plot(steps, values, linewidth=2, color="blue", label="RLHF Reward")
        if scores and scores["steps"]:
            ax.plot(scores["steps"], scores["values"], linewidth=2, color="hotpink", label="Scores", linestyle="--")
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.set_title("RLHF Reward & Scores per Batch")
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax = axes[1, 1]
        ax.text(0.5, 0.5, "Reward metrics not found", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Plot 5: Policy Entropy
    entropy_metric = get_metric(metrics, "train/policy/entropy_avg")
    if entropy_metric and entropy_metric["steps"]:
        ax = axes[2, 0]
        steps = entropy_metric["steps"]
        values = entropy_metric["values"]
        ax.plot(steps, values, linewidth=2, color="orange", label="Policy Entropy")
        ax.set_xlabel("Step")
        ax.set_ylabel("Entropy")
        ax.set_title("Policy Entropy over Steps")
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax = axes[2, 0]
        ax.text(0.5, 0.5, "Entropy metric not found", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # Hide the last subplot if not needed
    axes[2, 1].axis('off')

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(save_path, "training_progress.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    plt.show()


def save_metrics_summary(metrics, save_path="./ppo_training_output"):
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

    parser = argparse.ArgumentParser(description="Plot PPO training progress")
    parser.add_argument(
        "--log-dir",
        default="./ppo_training_output/logs",
        help="Path to training logs directory",
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
        print("Could not load TensorBoard logs. Make sure training has started.")
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
    output_dir = os.path.dirname(args.log_dir) if args.log_dir.endswith("logs") else args.log_dir
    plot_loss_and_rewards(metrics, output_dir)

    # Save metrics summary
    save_metrics_summary(metrics, output_dir)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}")
    print("  - training_progress.png (loss, learning rate, KL divergence, mean reward)")
    print("  - training_metrics.csv (all metrics)")


if __name__ == "__main__":
    main()
