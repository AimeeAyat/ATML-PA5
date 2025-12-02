"""
Data loading and preparation for instruction-following dataset.
"""

from typing import List, Dict
import datasets
from transformers import AutoTokenizer
import random
from config import RANDOM_SEED, INSTRUCTION_DATASET, DATASET_SIZE, MODEL_NAME


class InstructionDataset:
    """Load and manage instruction-following dataset."""

    def __init__(self, model_name: str = MODEL_NAME, dataset_name: str = INSTRUCTION_DATASET):
        """
        Initialize the dataset loader.

        Args:
            model_name: Name of the model for tokenizer
            dataset_name: Name of the instruction dataset
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dataset_name = dataset_name
        self.prompts = []

    def load_dataset(self, split: str = "train", num_samples: int = DATASET_SIZE) -> List[str]:
        """
        Load instruction dataset and extract prompts.

        Args:
            split: Dataset split to use
            num_samples: Number of samples to load

        Returns:
            List of instruction prompts
        """
        # Load dataset
        if self.dataset_name == "tatsu-lab/alpaca":
            dataset = datasets.load_dataset(self.dataset_name, split=split)
        else:
            dataset = datasets.load_dataset(self.dataset_name, split=split)

        # Sample and extract prompts
        random.seed(RANDOM_SEED)
        if len(dataset) > num_samples:
            indices = random.sample(range(len(dataset)), num_samples)
            dataset = dataset.select(indices)

        # Extract instruction text - handling different dataset formats
        self.prompts = []
        for example in dataset:
            if "instruction" in example:
                # Alpaca format: instruction + input (if exists)
                instruction = example.get("instruction", "")
                input_text = example.get("input", "")
                if input_text:
                    prompt = f"{instruction}\n{input_text}"
                else:
                    prompt = instruction
            elif "prompt" in example:
                prompt = example["prompt"]
            elif "text" in example:
                # Take first part of text as prompt
                prompt = example["text"].split("\n")[0]
            else:
                continue

            # Filter prompts that are too short
            if len(prompt.strip()) > 10:
                self.prompts.append(prompt.strip())

        return self.prompts

    def get_prompts(self, num_prompts: int = None) -> List[str]:
        """
        Get prompts from loaded dataset.

        Args:
            num_prompts: Number of prompts to return. If None, returns all.

        Returns:
            List of prompts
        """
        if not self.prompts:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        if num_prompts is None:
            return self.prompts

        return self.prompts[:num_prompts]

    def get_random_prompt(self) -> str:
        """Get a random prompt from the dataset."""
        if not self.prompts:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        return random.choice(self.prompts)
