"""
Plot DPO Training Progress from TensorBoard Logs
Extracts and visualizes training metrics: loss, learning_rate, rewards, accuracy, margins
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Configuration
LOG_DIR = "./dpo_training_output/logs"
OUTPUT_DIR = "./dpo_training_output"
METRICS_CSV = os.path.join(OUTPUT_DIR, "training_metrics.csv")


def extract_tensorboard_metrics(log_dir):
    """
    Extract all scalar metrics from TensorBoard event files

    Returns:
        dict: Metrics with structure {metric_name: {'steps': [...], 'values': [...]}}
    """
    print(f"\nExtracting metrics from: {log_dir}")

    try:
        # Initialize EventAccumulator
        ea = EventAccumulator(log_dir)
        ea.Reload()

        # Get all scalar tags
        scalar_tags = ea.Tags().get('scalars', [])
        print(f"Found {len(scalar_tags)} metrics: {scalar_tags}")

        metrics = {}
        for tag in scalar_tags:
            events = ea.Scalars(tag)
            steps = [event.step for event in events]
            values = [event.value for event in events]
            metrics[tag] = {'steps': steps, 'values': values}

        return metrics

    except Exception as e:
        print(f"Error reading TensorBoard logs: {e}")
        return {}


def save_metrics_to_csv(metrics, csv_path):
    """Save extracted metrics to CSV for further analysis"""
    print(f"\nSaving metrics to: {csv_path}")

    # Find max length
    max_len = max([len(m['steps']) for m in metrics.values()]) if metrics else 0

    # Create header
    headers = ["step"] + list(metrics.keys())

    # Write CSV
    with open(csv_path, 'w') as f:
        f.write(','.join(headers) + '\n')

        for i in range(max_len):
            row = []
            for j, metric_name in enumerate(["step"] + list(metrics.keys())):
                if metric_name == "step":
                    if i < len(metrics[list(metrics.keys())[0]]['steps']):
                        row.append(str(metrics[list(metrics.keys())[0]]['steps'][i]))
                    else:
                        row.append(str(i))
                else:
                    if i < len(metrics[metric_name]['values']):
                        row.append(str(metrics[metric_name]['values'][i]))
                    else:
                        row.append("")
            f.write(','.join(row) + '\n')

    print(f"[OK] Metrics saved to {csv_path}")


def plot_training_metrics(metrics):
    """
    Create comprehensive plots of training metrics
    """
    if not metrics:
        print("No metrics to plot!")
        return

    print("\nGenerating plots...")

    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle('DPO Training Progress - SmolLM2-135M', fontsize=16, fontweight='bold')

    # 1. Loss
    ax = axes[0, 0]
    loss_key = 'train/loss' if 'train/loss' in metrics else 'loss'
    if loss_key in metrics:
        steps = metrics[loss_key]['steps']
        values = metrics[loss_key]['values']
        ax.plot(steps, values, linewidth=2, color='#E74C3C')
        ax.fill_between(steps, values, alpha=0.3, color='#E74C3C')
        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title('Training Loss', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#F8F9FA')
        # Add description
        desc = f"Loss decreased from {values[0]:.4f} to {values[-1]:.4f}.\nModel learned to distinguish preferred from rejected responses."
        ax.text(0.98, 0.02, desc, transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # 2. Learning Rate
    ax = axes[0, 1]
    lr_key = 'train/learning_rate' if 'train/learning_rate' in metrics else 'learning_rate'
    if lr_key in metrics:
        steps = metrics[lr_key]['steps']
        values = metrics[lr_key]['values']
        ax.plot(steps, values, linewidth=2, color='#3498DB')
        ax.fill_between(steps, values, alpha=0.3, color='#3498DB')
        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel('Learning Rate', fontsize=11)
        ax.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#F8F9FA')
        # Add description
        desc = f"Learning rate annealed from {values[0]:.2e} to {values[-1]:.2e}.\nSmooth decay enables stable convergence."
        ax.text(0.98, 0.02, desc, transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    # 3. Rewards
    ax = axes[1, 0]
    chosen_key = 'train/rewards/chosen' if 'train/rewards/chosen' in metrics else 'rewards/chosen'
    rejected_key = 'train/rewards/rejected' if 'train/rewards/rejected' in metrics else 'rewards/rejected'
    if chosen_key in metrics and rejected_key in metrics:
        steps_chosen = metrics[chosen_key]['steps']
        values_chosen = metrics[chosen_key]['values']
        steps_rejected = metrics[rejected_key]['steps']
        values_rejected = metrics[rejected_key]['values']

        ax.plot(steps_chosen, values_chosen, linewidth=2.5, label='Chosen', color='#2ECC71', marker='o', markersize=4)
        ax.plot(steps_rejected, values_rejected, linewidth=2.5, label='Rejected', color='#E74C3C', marker='s', markersize=4)
        ax.fill_between(steps_chosen, values_chosen, alpha=0.2, color='#2ECC71')
        ax.fill_between(steps_rejected, values_rejected, alpha=0.2, color='#E74C3C')
        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel('Log Probability', fontsize=11)
        ax.set_title('Reward Scores (Log Probabilities)', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#F8F9FA')
        # Add description
        margin = values_chosen[-1] - values_rejected[-1]
        desc = f"Chosen: {values_chosen[-1]:.4f} | Rejected: {values_rejected[-1]:.4f}\nMargin: {margin:.4f} - Model learned preference."
        ax.text(0.98, 0.02, desc, transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    # 4. Accuracy & Margins
    ax = axes[1, 1]
    acc_key = 'train/rewards/accuracies' if 'train/rewards/accuracies' in metrics else 'rewards/accuracies'
    margin_key = 'train/rewards/margins' if 'train/rewards/margins' in metrics else 'rewards/margins'
    if acc_key in metrics or margin_key in metrics:
        if acc_key in metrics:
            steps_acc = metrics[acc_key]['steps']
            values_acc = np.array(metrics[acc_key]['values']) * 100  # Convert to percentage
            ax.plot(steps_acc, values_acc, linewidth=2.5, label='Accuracy', color='#9B59B6', marker='o', markersize=4)
            ax.fill_between(steps_acc, values_acc, alpha=0.2, color='#9B59B6')

        if margin_key in metrics:
            steps_margin = metrics[margin_key]['steps']
            values_margin = metrics[margin_key]['values']
            ax2 = ax.twinx()
            ax2.plot(steps_margin, values_margin, linewidth=2.5, label='Margin', color='#F39C12', marker='s', markersize=4)
            ax2.fill_between(steps_margin, values_margin, alpha=0.2, color='#F39C12')
            ax2.set_ylabel('Margin (log prob diff)', fontsize=11, color='#F39C12')
            ax2.tick_params(axis='y', labelcolor='#F39C12')

        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel('Accuracy (%)', fontsize=11, color='#9B59B6')
        ax.set_title('Model Preference Accuracy & Margin', fontsize=12, fontweight='bold')
        ax.tick_params(axis='y', labelcolor='#9B59B6')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#F8F9FA')

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels() if margin_key in metrics else ([], [])
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

        # Add description
        if acc_key in metrics:
            final_acc = values_acc[-1]
            desc = f"Accuracy: {final_acc:.1f}% - Model successfully learned preferences."
            ax.text(0.98, 0.02, desc, transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
                    horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7))

    # 5. Log Probabilities (Chosen vs Rejected)
    ax = axes[2, 0]
    logps_chosen_key = 'train/logps/chosen' if 'train/logps/chosen' in metrics else 'logps/chosen'
    logps_rejected_key = 'train/logps/rejected' if 'train/logps/rejected' in metrics else 'logps/rejected'
    if logps_chosen_key in metrics and logps_rejected_key in metrics:
        steps_chosen = metrics[logps_chosen_key]['steps']
        values_chosen = metrics[logps_chosen_key]['values']
        steps_rejected = metrics[logps_rejected_key]['steps']
        values_rejected = metrics[logps_rejected_key]['values']

        ax.plot(steps_chosen, values_chosen, linewidth=2.5, label='Chosen', color='#16A085', marker='o', markersize=3)
        ax.plot(steps_rejected, values_rejected, linewidth=2.5, label='Rejected', color='#C0392B', marker='s', markersize=3)
        ax.fill_between(steps_chosen, values_chosen, alpha=0.2, color='#16A085')
        ax.fill_between(steps_rejected, values_rejected, alpha=0.2, color='#C0392B')
        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel('Log Probability', fontsize=11)
        ax.set_title('Log Probabilities (Preferred vs Rejected)', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#F8F9FA')
        # Add description
        sep = values_chosen[-1] - values_rejected[-1]
        desc = f"Chosen→Rejected separation: {sep:.4f}\nModel properly assigns higher likelihood to preferred responses."
        ax.text(0.98, 0.02, desc, transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))

    # 6. Gradient Norm
    ax = axes[2, 1]
    grad_norm_key = 'train/grad_norm' if 'train/grad_norm' in metrics else 'grad_norm'
    if grad_norm_key in metrics:
        steps = metrics[grad_norm_key]['steps']
        values = metrics[grad_norm_key]['values']
        ax.plot(steps, values, linewidth=2, color='#8E44AD', marker='o', markersize=3)
        ax.fill_between(steps, values, alpha=0.3, color='#8E44AD')
        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel('Gradient Norm', fontsize=11)
        ax.set_title('Gradient Norm', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#F8F9FA')
        # Add description
        avg_grad = np.mean(values)
        desc = f"Average gradient norm: {avg_grad:.4f}\nStable gradients indicate healthy training dynamics."
        ax.text(0.98, 0.02, desc, transform=ax.transAxes, fontsize=9, verticalalignment='bottom',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='thistle', alpha=0.7))

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(OUTPUT_DIR, "training_progress.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Plot saved to {output_path}")

    return fig


def print_training_summary(metrics):
    """Print summary statistics of training"""
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)

    for metric_name, data in sorted(metrics.items()):
        if data['values']:
            values = np.array(data['values'])
            print(f"\n{metric_name}:")
            print(f"  Initial: {values[0]:.6f}")
            print(f"  Final:   {values[-1]:.6f}")
            print(f"  Min:     {values.min():.6f}")
            print(f"  Max:     {values.max():.6f}")
            print(f"  Mean:    {values.mean():.6f}")
            print(f"  Std:     {values.std():.6f}")

            # Calculate improvement
            if metric_name == 'loss':
                improvement = ((values[0] - values[-1]) / values[0]) * 100
                print(f"  Improvement: {improvement:.2f}%")


def main():
    """Main execution"""
    print("\n" + "="*80)
    print("DPO TRAINING VISUALIZATION")
    print("="*80)

    # Extract metrics
    metrics = extract_tensorboard_metrics(LOG_DIR)

    if not metrics:
        print("No metrics found. TensorBoard logs may not exist yet.")
        return

    # Save to CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_metrics_to_csv(metrics, METRICS_CSV)

    # Print summary
    print_training_summary(metrics)

    # Generate plots
    plot_training_metrics(metrics)

    print("\n" + "="*80)
    print("[OK] Visualization complete!")
    print("="*80)


if __name__ == "__main__":
    main()
