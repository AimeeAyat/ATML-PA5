"""
Plotting and visualization functions for analysis results.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import os
from config import PLOTS_DIR, TEMPERATURE_VALUES


class ResultsPlotter:
    """Generate visualization plots for evaluation results."""

    def __init__(self, results_dir: str = PLOTS_DIR):
        """
        Initialize plotter.

        Args:
            results_dir: Directory to save plots
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def plot_temperature_ablation_distinct(self, results: Dict):
        """
        Plot Distinct-N scores across temperature values.

        Args:
            results: Results dictionary from temperature ablation
        """
        temp_results = results.get("temperature_ablation", {})
        if not temp_results:
            print("No temperature ablation results found")
            return

        strategies = list(temp_results.keys())
# Check if per-temperature timing data exists        has_timing_data = False        for strategy in strategies:            if "temperatures" in temp_results[strategy]:                temps = list(temp_results[strategy]["temperatures"].keys())                if temps and "time_seconds" in temp_results[strategy]["temperatures"][temps[0]]:                    has_timing_data = True                    break                if not has_timing_data:            return
        n_values = [1, 2, 3]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("Distinct-N Scores Across Temperature", fontsize=16, fontweight="bold")

        for ax_idx, n in enumerate(n_values):
            ax = axes[ax_idx]

            for strategy in strategies:
                temps = sorted(
                    temp_results[strategy]["temperatures"].keys(), key=lambda x: float(x)
                )
                distinct_scores = [
                    temp_results[strategy]["temperatures"][temp]["distinct_scores"][
                        f"Distinct-{n}"
                    ]
                    for temp in temps
                ]

                ax.plot(temps, distinct_scores, marker="o", label=strategy, linewidth=2)

            ax.set_xlabel("Temperature", fontsize=11)
            ax.set_ylabel(f"Distinct-{n}", fontsize=11)
            ax.set_title(f"Distinct-{n}", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        filepath = os.path.join(self.results_dir, "temperature_ablation_distinct.png")
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {filepath}")
        plt.close()

    def plot_temperature_ablation_reward(self, results: Dict):
        """
        Plot reward scores across temperature values.

        Args:
            results: Results dictionary from temperature ablation
        """
        temp_results = results.get("temperature_ablation", {})
        if not temp_results:
            print("No temperature ablation results found")
            return

        strategies = list(temp_results.keys())
# Check if per-temperature timing data exists        has_timing_data = False        for strategy in strategies:            if "temperatures" in temp_results[strategy]:                temps = list(temp_results[strategy]["temperatures"].keys())                if temps and "time_seconds" in temp_results[strategy]["temperatures"][temps[0]]:                    has_timing_data = True                    break                if not has_timing_data:            return

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle("Reward Model Scores Across Temperature", fontsize=16, fontweight="bold")

        for strategy in strategies:
            temps = sorted(temp_results[strategy]["temperatures"].keys(), key=lambda x: float(x))
            reward_scores = [
                temp_results[strategy]["temperatures"][temp]["avg_reward"] for temp in temps
            ]

            ax.plot(temps, reward_scores, marker="o", label=strategy, linewidth=2)

        ax.set_xlabel("Temperature", fontsize=12)
        ax.set_ylabel("Average Reward Score", fontsize=12)
        ax.set_title("Quality vs. Temperature", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        filepath = os.path.join(self.results_dir, "temperature_ablation_reward.png")
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {filepath}")
        plt.close()

    def plot_across_prompt_diversity(self, results: Dict):
        """
        Plot across-prompt diversity comparison.

        Args:
            results: Results dictionary with diversity metrics
        """
        diversity_results = results.get("across_prompt_diversity", {})
        if not diversity_results:
            print("No across-prompt diversity results found")
            return

        strategies = list(diversity_results.keys())
        distinct_values = [1, 2, 3]

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle("Across-Prompt Diversity Comparison", fontsize=16, fontweight="bold")

        x = np.arange(len(strategies))
        width = 0.25

        for idx, n in enumerate(distinct_values):
            scores = [
                diversity_results[strategy]["distinct_scores"][f"Distinct-{n}"]
                for strategy in strategies
            ]
            ax.bar(x + idx * width, scores, width, label=f"Distinct-{n}")

        ax.set_xlabel("Decoding Strategy", fontsize=12)
        ax.set_ylabel("Distinct-N Score", fontsize=12)
        ax.set_title("Lexical Diversity (One Sample per Prompt)", fontsize=13, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(strategies, rotation=15, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        filepath = os.path.join(self.results_dir, "across_prompt_diversity.png")
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {filepath}")
        plt.close()

    def plot_within_prompt_diversity(self, results: Dict):
        """
        Plot within-prompt diversity comparison.

        Args:
            results: Results dictionary with diversity metrics
        """
        diversity_results = results.get("within_prompt_diversity", {})
        if not diversity_results:
            print("No within-prompt diversity results found")
            return

        strategies = list(diversity_results.keys())
        distinct_values = [1, 2, 3]

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle("Within-Prompt Diversity Comparison", fontsize=16, fontweight="bold")

        x = np.arange(len(strategies))
        width = 0.25

        for idx, n in enumerate(distinct_values):
            scores = [
                diversity_results[strategy]["distinct_scores"][f"Distinct-{n}"]
                for strategy in strategies
            ]
            ax.bar(x + idx * width, scores, width, label=f"Distinct-{n}")

        ax.set_xlabel("Decoding Strategy", fontsize=12)
        ax.set_ylabel("Distinct-N Score", fontsize=12)
        ax.set_title("Inter-Sample Variation (Multiple Samples per Prompt)", fontsize=13, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(strategies, rotation=15, ha="right")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        filepath = os.path.join(self.results_dir, "within_prompt_diversity.png")
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {filepath}")
        plt.close()

    def plot_diversity_quality_tradeoff(self, results: Dict):
        """
        Plot diversity vs. quality tradeoff across temperature.

        Args:
            results: Results dictionary from temperature ablation
        """
        temp_results = results.get("temperature_ablation", {})
        if not temp_results:
            print("No temperature ablation results found")
            return

        strategies = list(temp_results.keys())
# Check if per-temperature timing data exists        has_timing_data = False        for strategy in strategies:            if "temperatures" in temp_results[strategy]:                temps = list(temp_results[strategy]["temperatures"].keys())                if temps and "time_seconds" in temp_results[strategy]["temperatures"][temps[0]]:                    has_timing_data = True                    break                if not has_timing_data:            return

        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle("Diversity-Quality Tradeoff (Distinct-1 vs. Reward)", fontsize=16, fontweight="bold")

        for strategy in strategies:
            temps = sorted(temp_results[strategy]["temperatures"].keys(), key=lambda x: float(x))

            distinct_scores = [
                temp_results[strategy]["temperatures"][temp]["distinct_scores"]["Distinct-1"]
                for temp in temps
            ]
            reward_scores = [
                temp_results[strategy]["temperatures"][temp]["avg_reward"] for temp in temps
            ]

            # Plot with color gradient based on temperature
            scatter = ax.scatter(
                distinct_scores,
                reward_scores,
                s=100,
                alpha=0.6,
                label=strategy,
            )

            # Add temperature annotations
            for i, temp in enumerate(temps):
                ax.annotate(
                    f"T={temp}",
                    (distinct_scores[i], reward_scores[i]),
                    fontsize=8,
                    alpha=0.7,
                )

        ax.set_xlabel("Distinct-1 (Diversity)", fontsize=12)
        ax.set_ylabel("Reward Score (Quality)", fontsize=12)
        ax.set_title("Exploring the Diversity-Quality Tradeoff", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        filepath = os.path.join(self.results_dir, "diversity_quality_tradeoff.png")
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {filepath}")
        plt.close()

    def plot_strategy_timing_comparison(self, results: Dict):
        """
        Plot total time taken by each strategy.

        Args:
            results: Results dictionary from evaluation
        """
        temp_results = results.get("temperature_ablation", {})
        if not temp_results:
            print("No timing data found")
            return

        strategies = list(temp_results.keys())
# Check if per-temperature timing data exists        has_timing_data = False        for strategy in strategies:            if "temperatures" in temp_results[strategy]:                temps = list(temp_results[strategy]["temperatures"].keys())                if temps and "time_seconds" in temp_results[strategy]["temperatures"][temps[0]]:                    has_timing_data = True                    break                if not has_timing_data:            return
        times = [temp_results[s]["total_time_seconds"] for s in strategies]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(strategies, times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}s',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel("Time (seconds)", fontsize=12)
        ax.set_title("Total Time per Strategy (All Temperatures)", fontsize=13, fontweight="bold")
        ax.set_ylim(0, max(times) * 1.15)
        ax.grid(True, alpha=0.3, axis="y")

        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        filepath = os.path.join(self.results_dir, "strategy_timing_comparison.png")
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
    def plot_timing_per_temperature(self, results: Dict):
        """
        Plot time taken at each temperature level (if data available).

        Args:
            results: Results dictionary from temperature ablation
        """
        temp_results = results.get("temperature_ablation", {})
        if not temp_results:
            return

        strategies = list(temp_results.keys())
        
        # Check if per-temperature timing data exists
        has_timing_data = False
        for strategy in strategies:
            if "temperatures" in temp_results[strategy]:
                for temp_key in temp_results[strategy]["temperatures"].keys():
                    if "time_seconds" in temp_results[strategy]["temperatures"][temp_key]:
                        has_timing_data = True
                        break
            if has_timing_data:
                break
        
        if not has_timing_data:
            # Skip plot if timing data not available
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        for strategy in strategies:
            temps = sorted(temp_results[strategy]["temperatures"].keys(), key=lambda x: float(x))
            times = [temp_results[strategy]["temperatures"][t].get("time_seconds", 0) for t in temps]
            ax.plot(temps, times, marker='o', label=strategy, linewidth=2)

        ax.set_xlabel("Temperature", fontsize=12)
        ax.set_ylabel("Time per Temperature (seconds)", fontsize=12)
        ax.set_title("Inference Time vs Temperature", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.tight_layout()
        filepath = os.path.join(self.results_dir, "timing_per_temperature.png")
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {filepath}")
        plt.close()


    def plot_across_prompt_timing(self, results: Dict):
        """
        Plot time taken for across-prompt diversity test.

        Args:
            results: Results dictionary with diversity timing
        """
        diversity_results = results.get("across_prompt_diversity", {})
        if not diversity_results:
            return

        strategies = list(diversity_results.keys())
        times = [diversity_results[s]["total_time_seconds"] for s in strategies]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(strategies, times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}s',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel("Time (seconds)", fontsize=12)
        ax.set_title("Across-Prompt Diversity - Time Comparison", fontsize=13, fontweight="bold")
        ax.set_ylim(0, max(times) * 1.15)
        ax.grid(True, alpha=0.3, axis="y")

        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        filepath = os.path.join(self.results_dir, "timing_across_prompt.png")
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {filepath}")
        plt.close()

    def plot_within_prompt_timing(self, results: Dict):
        """
        Plot time taken for within-prompt diversity test.

        Args:
            results: Results dictionary with diversity timing
        """
        diversity_results = results.get("within_prompt_diversity", {})
        if not diversity_results:
            return

        strategies = list(diversity_results.keys())
        times = [diversity_results[s]["total_time_seconds"] for s in strategies]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(strategies, times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}s',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_ylabel("Time (seconds)", fontsize=12)
        ax.set_title("Within-Prompt Diversity - Time Comparison", fontsize=13, fontweight="bold")
        ax.set_ylim(0, max(times) * 1.15)
        ax.grid(True, alpha=0.3, axis="y")

        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        filepath = os.path.join(self.results_dir, "timing_within_prompt.png")
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {filepath}")
        plt.close()

    def plot_all_comparisons(self, results: Dict):
        """
        Generate all comparison plots.

        Args:
            results: Complete results dictionary
        """
        print("\nGenerating plots...")
        self.plot_temperature_ablation_distinct(results)
        self.plot_temperature_ablation_reward(results)
        self.plot_across_prompt_diversity(results)
        self.plot_within_prompt_diversity(results)
        self.plot_diversity_quality_tradeoff(results)
        self.plot_strategy_timing_comparison(results)
        self.plot_timing_per_temperature(results)
        self.plot_across_prompt_timing(results)
        self.plot_within_prompt_timing(results)
        print("All plots saved successfully!")
