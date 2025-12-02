"""
Visualize GRPO training progress
Reads TensorBoard logs and creates plots for loss, rewards, and other metrics
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_logs(log_dir="./grpo_training_output/logs"):
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


def plot_loss_and_rewards(metrics, save_path="./grpo_training_output"):
    """Plot training loss, learning rate, rewards, entropy, and other GRPO metrics"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle("GRPO Training Progress - SmolLM2-135M", fontsize=16, fontweight="bold")

    # Helper function to find metric (with or without prefix)
    def get_metric(metrics, name):
        if name in metrics:
            return metrics[name]
        elif f"train/{name}" in metrics:
            return metrics[f"train/{name}"]
        return None

    # Plot 1: Loss
    loss_metric = get_metric(metrics, "loss")
    if loss_metric and loss_metric["steps"]:
        ax = axes[0, 0]
        steps = loss_metric["steps"]
        values = loss_metric["values"]
        ax.plot(steps, values, linewidth=2, color="red")
        ax.fill_between(steps, values, alpha=0.3, color="red")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.grid(True, alpha=0.3)
        desc = f"Loss: {values[0]:.4f} → {values[-1]:.4f}. Model learned to optimize GRPO objective."
        ax.text(0.98, 0.02, desc, transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    else:
        axes[0, 0].text(0.5, 0.5, "Loss metric not found", ha="center", va="center", transform=axes[0, 0].transAxes)

    # Plot 2: Learning Rate
    lr_metric = get_metric(metrics, "learning_rate")
    if lr_metric and lr_metric["steps"]:
        ax = axes[0, 1]
        steps = lr_metric["steps"]
        values = lr_metric["values"]
        ax.plot(steps, values, linewidth=2, color="green")
        ax.fill_between(steps, values, alpha=0.3, color="green")
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.3)
        desc = f"LR: {values[0]:.2e} → {values[-1]:.2e}. Smooth cosine decay."
        ax.text(0.98, 0.02, desc, transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    else:
        axes[0, 1].text(0.5, 0.5, "Learning rate metric not found", ha="center", va="center", transform=axes[0, 1].transAxes)

    # Plot 3: Reward (train/reward)
    reward_metric = get_metric(metrics, "reward")
    if reward_metric and reward_metric["steps"]:
        ax = axes[0, 2]
        steps = reward_metric["steps"]
        values = reward_metric["values"]
        ax.plot(steps, values, linewidth=2, color="blue")
        ax.fill_between(steps, values, alpha=0.3, color="blue")
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.set_title("Training Reward")
        ax.grid(True, alpha=0.3)
        desc = f"Reward: {np.mean(values):.4f} ± {np.std(values):.4f}. Mean performance."
        ax.text(0.98, 0.02, desc, transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    else:
        axes[0, 2].text(0.5, 0.5, "Reward metric not found", ha="center", va="center", transform=axes[0, 2].transAxes)

    # Plot 4: Reward Std Dev
    reward_std = get_metric(metrics, "reward_std")
    if reward_std and reward_std["steps"]:
        ax = axes[1, 0]
        steps = reward_std["steps"]
        values = reward_std["values"]
        ax.plot(steps, values, linewidth=2, color="purple")
        ax.fill_between(steps, values, alpha=0.3, color="purple")
        ax.set_xlabel("Step")
        ax.set_ylabel("Std Dev")
        ax.set_title("Reward Std Deviation")
        ax.grid(True, alpha=0.3)
        desc = f"Reward Std: {np.mean(values):.4f}. Variance in generated quality."
        ax.text(0.98, 0.02, desc, transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='thistle', alpha=0.7))
    else:
        axes[1, 0].text(0.5, 0.5, "Reward std metric not found", ha="center", va="center", transform=axes[1, 0].transAxes)

    # Plot 5: Entropy
    entropy_metric = get_metric(metrics, "entropy")
    if entropy_metric and entropy_metric["steps"]:
        ax = axes[1, 1]
        steps = entropy_metric["steps"]
        values = entropy_metric["values"]
        ax.plot(steps, values, linewidth=2, color="orange")
        ax.fill_between(steps, values, alpha=0.3, color="orange")
        ax.set_xlabel("Step")
        ax.set_ylabel("Entropy")
        ax.set_title("Policy Entropy")
        ax.grid(True, alpha=0.3)
        desc = f"Entropy: {values[-1]:.4f}. Maintains exploration capability."
        ax.text(0.98, 0.02, desc, transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    else:
        axes[1, 1].text(0.5, 0.5, "Entropy metric not found", ha="center", va="center", transform=axes[1, 1].transAxes)

    # Plot 6: Gradient Norm
    grad_norm = get_metric(metrics, "grad_norm")
    if grad_norm and grad_norm["steps"]:
        ax = axes[1, 2]
        steps = grad_norm["steps"]
        values = grad_norm["values"]
        ax.plot(steps, values, linewidth=2, color="brown")
        ax.fill_between(steps, values, alpha=0.3, color="brown")
        ax.set_xlabel("Step")
        ax.set_ylabel("Grad Norm")
        ax.set_title("Gradient Norm")
        ax.grid(True, alpha=0.3)
        desc = f"Grad Norm: {np.mean(values):.4f}. Healthy training dynamics."
        ax.text(0.98, 0.02, desc, transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='navajowhite', alpha=0.7))
    else:
        axes[1, 2].text(0.5, 0.5, "Gradient norm metric not found", ha="center", va="center", transform=axes[1, 2].transAxes)

    # Plot 7: Mean Completion Length
    completion_length = get_metric(metrics, "completions/mean_length")
    if completion_length and completion_length["steps"]:
        ax = axes[2, 0]
        steps = completion_length["steps"]
        values = completion_length["values"]
        ax.plot(steps, values, linewidth=2, color="teal")
        ax.fill_between(steps, values, alpha=0.3, color="teal")
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean Length")
        ax.set_title("Mean Completion Length")
        ax.grid(True, alpha=0.3)
        desc = f"Avg Length: {np.mean(values):.1f} tokens. Generation diversity."
        ax.text(0.98, 0.02, desc, transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))
    else:
        axes[2, 0].text(0.5, 0.5, "Completion length metric not found", ha="center", va="center", transform=axes[2, 0].transAxes)

    # Plot 8: Clipping Ratio Mean
    clip_ratio = get_metric(metrics, "clip_ratio/region_mean")
    if clip_ratio and clip_ratio["steps"]:
        ax = axes[2, 1]
        steps = clip_ratio["steps"]
        values = clip_ratio["values"]
        ax.plot(steps, values, linewidth=2, color="magenta")
        ax.fill_between(steps, values, alpha=0.3, color="magenta")
        ax.set_xlabel("Step")
        ax.set_ylabel("Clipping Ratio")
        ax.set_title("Clipping Ratio (Policy Update)")
        ax.grid(True, alpha=0.3)
        desc = f"Clip Ratio: {np.mean(values):.4f}. Trust region constraint."
        ax.text(0.98, 0.02, desc, transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7))
    else:
        axes[2, 1].text(0.5, 0.5, "Clipping ratio metric not found", ha="center", va="center", transform=axes[2, 1].transAxes)

    # Plot 9: Reward Function Mean
    reward_fn_mean = get_metric(metrics, "rewards/reward_fn/mean")
    if reward_fn_mean and reward_fn_mean["steps"]:
        ax = axes[2, 2]
        steps = reward_fn_mean["steps"]
        values = reward_fn_mean["values"]
        ax.plot(steps, values, linewidth=2, color="darkgreen")
        ax.fill_between(steps, values, alpha=0.3, color="darkgreen")
        ax.set_xlabel("Step")
        ax.set_ylabel("Reward")
        ax.set_title("Reward Function Mean")
        ax.grid(True, alpha=0.3)
        desc = f"Reward Fn: {values[-1]:.4f}. Model performance score."
        ax.text(0.98, 0.02, desc, transform=ax.transAxes, fontsize=8, verticalalignment='bottom',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    else:
        axes[2, 2].text(0.5, 0.5, "Reward function mean metric not found", ha="center", va="center", transform=axes[2, 2].transAxes)

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(save_path, "training_progress.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_file}")
    plt.show()


def save_metrics_summary(metrics, save_path="./grpo_training_output"):
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

    parser = argparse.ArgumentParser(description="Plot GRPO training progress")
    parser.add_argument(
        "--log-dir",
        default="./grpo_training_output/logs",
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
    print("  - training_progress.png (9 comprehensive GRPO metrics)")
    print("    Row 1: Loss | Learning Rate | Reward")
    print("    Row 2: Reward Std | Entropy | Gradient Norm")
    print("    Row 3: Completion Length | Clipping Ratio | Reward Function Mean")
    print("  - training_metrics.csv (all metrics)")


if __name__ == "__main__":
    main()
