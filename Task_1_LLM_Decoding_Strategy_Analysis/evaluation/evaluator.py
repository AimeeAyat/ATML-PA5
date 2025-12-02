"""
Comprehensive evaluation pipeline for decoding strategies.
"""

import torch
import json
import os
import time
import random
from typing import Dict, List, Tuple
from tqdm import tqdm

from decoding_strategies import (
    GreedySearchDecoder,
    BeamSearchDecoder,
    TopKSamplingDecoder,
    TopPSamplingDecoder,
)
from metrics import DistinctNMetric, RewardModelScorer
from config import (
    TEMPERATURE_VALUES,
    DISTINCT_N_VALUES,
    NUM_PROMPTS,
    SAMPLES_PER_PROMPT,
    NUM_SAMPLES_PER_TEMP,
    RESULTS_DIR,
    JSON_RESULTS_DIR,
    PLOTS_DIR,
    REWARD_MODEL_NAME,
)


class DecodingStrategyEvaluator:
    """Comprehensive evaluator for decoding strategies."""

    def __init__(self, model, tokenizer, beam_width=5, top_k=50, top_p=0.95, reward_model_name=None):
        """
        Initialize evaluator.

        Args:
            model: Language model
            tokenizer: Tokenizer
            beam_width: Beam width for beam search
            top_k: Top-K value for top-k sampling
            top_p: Top-P value for top-p sampling
            reward_model_name: Reward model name (None to skip)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.top_k = top_k
        self.top_p = top_p
        self.max_retries = 3  # Retry up to 3 times if generation is empty

        # Initialize decoders
        self.decoders = {
            "Greedy": GreedySearchDecoder(model, tokenizer),
            "Beam Search": BeamSearchDecoder(model, tokenizer, beam_width=beam_width),
            "Top-K": TopKSamplingDecoder(model, tokenizer, top_k=top_k),
            "Top-P": TopPSamplingDecoder(model, tokenizer, top_p=top_p),
        }

        # Initialize reward model (only if specified)
        if reward_model_name:
            self.reward_scorer = RewardModelScorer(reward_model_name)
            if not self.reward_scorer.is_dummy_scorer():
                print(f"[SUCCESS] Reward model initialized with real scores")
            else:
                print(f"[WARNING] Reward model failed to load. Using dummy scorer (constant 0.5 scores)")
        else:
            print("[INFO] No reward model specified. Using dummy scorer (constant 0.5 scores)")
            self.reward_scorer = RewardModelScorer(None)  # Dummy scorer

        # Initialize results storage
        self.results = {}

        # Create necessary directories
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(JSON_RESULTS_DIR, exist_ok=True)
        os.makedirs(PLOTS_DIR, exist_ok=True)

    def _generate_with_retry(self, decoder, prompt: str, max_length: int, temperature: float) -> Tuple[str, torch.Tensor]:
        """
        Generate text with retry on empty output.

        Args:
            decoder: Decoder instance
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Temperature value

        Returns:
            Tuple of (generated_text, log_probabilities)
        """
        for attempt in range(self.max_retries):
            # Randomize seed for each retry to get different random behavior
            retry_seed = random.randint(0, 2**31 - 1)
            torch.manual_seed(retry_seed)
            random.seed(retry_seed)

            generated_text, log_probs = decoder.decode(
                prompt, max_length=max_length, temperature=temperature
            )

            # If non-empty, return immediately
            if generated_text.strip():
                return generated_text, log_probs

        # If all retries failed, return the last (empty) result
        return generated_text, log_probs

    def evaluate_temperature_ablation(
        self, prompts: List[str], max_length: int = 200
    ) -> Dict:
        """
        Evaluate decoding strategies across temperature range.

        Args:
            prompts: List of prompts
            max_length: Maximum generation length

        Returns:
            Dictionary with results
        """
        results = {
            "temperature_ablation": {
                strategy: {"temperatures": {}, "aggregated_metrics": {}}
                for strategy in self.decoders.keys()
            }
        }

        print("\n=== Temperature Ablation ===")
        for strategy_name, decoder in self.decoders.items():
            print(f"\nEvaluating {strategy_name}...")
            strategy_start_time = time.time()

            for temp in TEMPERATURE_VALUES:
                print(f"  Temperature: {temp}")
                generations = []
                reward_scores = []

                for prompt in tqdm(prompts[:NUM_PROMPTS], desc=f"  Generating", leave=False):
                    for _ in range(NUM_SAMPLES_PER_TEMP):
                        try:
                            # Use retry logic to avoid empty generations
                            generated_text, _ = self._generate_with_retry(
                                decoder, prompt, max_length=max_length, temperature=temp
                            )
                            generations.append(generated_text)

                            # Score with reward model
                            reward_score = self.reward_scorer.score_response_only(generated_text)
                            reward_scores.append(reward_score)
                        except Exception as e:
                            print(f"    Error in generation: {e}")
                            continue

                # Compute metrics
                distinct_scores = DistinctNMetric.compute_all_distinct_n(
                    generations, DISTINCT_N_VALUES
                )
                avg_reward = sum(reward_scores) / len(reward_scores) if reward_scores else 0.0

                results["temperature_ablation"][strategy_name]["temperatures"][temp] = {
                    "distinct_scores": distinct_scores,
                    "avg_reward": avg_reward,
                    "num_samples": len(generations),
                }

            # Aggregate metrics across temperatures
            temps = list(results["temperature_ablation"][strategy_name]["temperatures"].keys())
            for metric_name in ["Distinct-1", "Distinct-2", "Distinct-3"]:
                values = [
                    results["temperature_ablation"][strategy_name]["temperatures"][t][
                        "distinct_scores"
                    ][metric_name]
                    for t in temps
                ]
                results["temperature_ablation"][strategy_name]["aggregated_metrics"][
                    f"{metric_name}_avg"
                ] = sum(values) / len(values)

            # Track timing
            strategy_time = time.time() - strategy_start_time
            results["temperature_ablation"][strategy_name]["total_time_seconds"] = strategy_time
            print(f"  Total time for {strategy_name}: {strategy_time:.2f}s")

        return results

    def evaluate_across_prompt_diversity(self, prompts: List[str], max_length: int = 200) -> Dict:
        """
        Evaluate diversity across different prompts (one sample per prompt).

        Args:
            prompts: List of prompts
            max_length: Maximum generation length

        Returns:
            Dictionary with results
        """
        results = {"across_prompt_diversity": {}}
        temp = 0.8  # Fixed temperature

        print("\n=== Across-Prompt Diversity (T=0.8) ===")
        for strategy_name, decoder in self.decoders.items():
            print(f"Evaluating {strategy_name}...")
            strategy_start_time = time.time()
            generations = []

            for prompt in tqdm(prompts[:NUM_PROMPTS], desc=f"  Generating", leave=False):
                try:
                    # Use retry logic to avoid empty generations
                    generated_text, _ = self._generate_with_retry(
                        decoder, prompt, max_length=max_length, temperature=temp
                    )
                    generations.append(generated_text)
                except Exception as e:
                    print(f"  Error: {e}")
                    continue

            # Compute Distinct-N across all generations
            distinct_scores = DistinctNMetric.compute_all_distinct_n(
                generations, DISTINCT_N_VALUES
            )

            strategy_time = time.time() - strategy_start_time
            results["across_prompt_diversity"][strategy_name] = {
                "distinct_scores": distinct_scores,
                "num_samples": len(generations),
                "total_time_seconds": strategy_time,
            }
            print(f"  Total time for {strategy_name}: {strategy_time:.2f}s")

        return results

    def evaluate_within_prompt_diversity(self, prompt: str, max_length: int = 200) -> Dict:
        """
        Evaluate diversity within a single prompt (multiple samples per prompt).

        Args:
            prompt: Single prompt
            max_length: Maximum generation length

        Returns:
            Dictionary with results
        """
        results = {"within_prompt_diversity": {}}
        temp = 0.8  # Fixed temperature

        print(f"\n=== Within-Prompt Diversity (T=0.8) ===")
        print(f"Prompt: {prompt[:100]}...")

        for strategy_name, decoder in self.decoders.items():
            print(f"Evaluating {strategy_name}...")
            strategy_start_time = time.time()
            generations = []

            for _ in tqdm(range(SAMPLES_PER_PROMPT), desc=f"  Sampling", leave=False):
                try:
                    # Use retry logic to avoid empty generations
                    generated_text, _ = self._generate_with_retry(
                        decoder, prompt, max_length=max_length, temperature=temp
                    )
                    generations.append(generated_text)
                except Exception as e:
                    print(f"  Error: {e}")
                    continue

            # Compute Distinct-N for samples from same prompt
            distinct_scores = DistinctNMetric.compute_all_distinct_n(
                generations, DISTINCT_N_VALUES
            )

            strategy_time = time.time() - strategy_start_time
            results["within_prompt_diversity"][strategy_name] = {
                "distinct_scores": distinct_scores,
                "num_samples": len(generations),
                "total_time_seconds": strategy_time,
            }
            print(f"  Total time for {strategy_name}: {strategy_time:.2f}s")

        return results

    def save_results(self, all_results: Dict, filename: str = "evaluation_results.json"):
        """
        Save results to JSON file.

        Args:
            all_results: Dictionary with all results
            filename: Output filename
        """
        filepath = os.path.join(JSON_RESULTS_DIR, filename)
        with open(filepath, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"\nResults saved to {filepath}")

    def save_sample_generations(self, samples: Dict, filename: str = "sample_generations.json"):
        """
        Save sample prompts and generations to JSON.

        Args:
            samples: Dictionary with sample generations
            filename: Output filename
        """
        filepath = os.path.join(JSON_RESULTS_DIR, filename)
        with open(filepath, "w") as f:
            json.dump(samples, f, indent=4)
        print(f"Sample generations saved to {filepath}")

    def run_full_evaluation(
        self, prompts: List[str], single_prompt: str, max_length: int = 200
    ) -> Dict:
        """
        Run full evaluation pipeline.

        Args:
            prompts: List of prompts for diversity tests
            single_prompt: Single prompt for within-prompt diversity
            max_length: Maximum generation length

        Returns:
            Dictionary with all results
        """
        all_results = {}

        # Temperature ablation
        all_results.update(self.evaluate_temperature_ablation(prompts, max_length))

        # Diversity tests
        all_results.update(self.evaluate_across_prompt_diversity(prompts, max_length))
        all_results.update(self.evaluate_within_prompt_diversity(single_prompt, max_length))

        # Collect sample generations for inspection across all temperatures
        print("\n=== Collecting Sample Generations Across All Temperatures ===")
        samples = {"samples": []}
        sample_prompts = prompts[:2]  # First 2 prompts (to keep file manageable)

        for prompt in sample_prompts:
            prompt_samples = {
                "prompt": prompt,
                "strategies": {}
            }

            for strategy_name, decoder in self.decoders.items():
                prompt_samples["strategies"][strategy_name] = {
                    "temperatures": {}
                }

                for temp in TEMPERATURE_VALUES:
                    try:
                        # Use retry logic to avoid empty generations
                        generated_text, _ = self._generate_with_retry(
                            decoder, prompt, max_length=max_length, temperature=temp
                        )
                        reward = self.reward_scorer.score_response_only(generated_text)

                        prompt_samples["strategies"][strategy_name]["temperatures"][temp] = {
                            "generation": generated_text,
                            "reward": reward,
                            "length": len(generated_text.split())
                        }
                    except Exception as e:
                        prompt_samples["strategies"][strategy_name]["temperatures"][temp] = {
                            "generation": "",
                            "reward": 0.0,
                            "length": 0,
                            "error": str(e)
                        }

            samples["samples"].append(prompt_samples)

        # Save results and samples
        self.save_results(all_results)
        self.save_sample_generations(samples)

        return all_results
