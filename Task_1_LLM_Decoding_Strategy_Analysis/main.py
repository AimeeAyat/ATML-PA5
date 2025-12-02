"""
Main orchestration script for LLM decoding strategies analysis.
"""

import torch
import argparse
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import InstructionDataset
from evaluation import DecodingStrategyEvaluator
from utils import ResultsPlotter
from config import (
    MODEL_NAME,
    DEVICE,
    BEAM_WIDTH,
    TOP_K,
    TOP_P,
    MAX_LENGTH,
    RANDOM_SEED,
    REWARD_MODEL_NAME,
)


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def load_model_and_tokenizer(model_name: str, device: str):
    """
    Load model and tokenizer.

    Args:
        model_name: Model name on HuggingFace Hub
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer)
    """
    logger = setup_logging()
    logger.info(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )

    if device == "cpu":
        model = model.to(device)

    model.eval()

    logger.info(f"Model loaded successfully on {device}")
    return model, tokenizer


def main(args):
    """
    Main execution function.

    Args:
        args: Command line arguments
    """
    logger = setup_logging()

    # Set random seed
    torch.manual_seed(RANDOM_SEED)

    logger.info("=" * 60)
    logger.info("LLM Decoding Strategies Analysis")
    logger.info("=" * 60)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME, DEVICE)

    # Load dataset
    logger.info("Loading instruction dataset...")
    dataset_loader = InstructionDataset(MODEL_NAME)
    prompts = dataset_loader.load_dataset()
    logger.info(f"Loaded {len(prompts)} prompts")

    # Initialize evaluator
    logger.info("Initializing evaluator...")
    evaluator = DecodingStrategyEvaluator(
        model,
        tokenizer,
        beam_width=BEAM_WIDTH,
        top_k=TOP_K,
        top_p=TOP_P,
        reward_model_name=REWARD_MODEL_NAME,
    )

    # Run evaluation
    logger.info("Starting evaluation pipeline...")
    single_prompt = prompts[0] if prompts else "Write a poem about nature."

    results = evaluator.run_full_evaluation(
        prompts=prompts,
        single_prompt=single_prompt,
        max_length=MAX_LENGTH,
    )

    # Generate plots
    logger.info("Generating visualization plots...")
    plotter = ResultsPlotter()
    plotter.plot_all_comparisons(results)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Complete!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: results/json_results/evaluation_results.json")
    logger.info(f"Plots saved to: results/plots/")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze LLM decoding strategies"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="Model name on HuggingFace Hub",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=BEAM_WIDTH,
        help="Beam width for beam search",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help="Top-K value",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=TOP_P,
        help="Top-P value",
    )

    args = parser.parse_args()
    main(args)
