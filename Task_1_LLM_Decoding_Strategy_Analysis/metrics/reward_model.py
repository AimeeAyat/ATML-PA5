"""
Reward model scoring for evaluating generation quality.
"""

import torch
from typing import List
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class RewardModelScorer:
    """Score generation quality using a pretrained reward model."""

    def __init__(self, model_name=None):
        """
        Initialize the reward model scorer.

        Args:
            model_name: Name of the reward model on HuggingFace Hub. If None, uses dummy scorer.
        """
        self.model_name = model_name
        self.device = torch.device("cpu")  # Force CPU (CUDA sm_120 not fully supported)
        self.model = None
        self.tokenizer = None

        # Load model and tokenizer only if model_name is specified
        if model_name is None:
            # Use dummy scorer (returns constant scores)
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
            self.model.eval()
            print(f"[RewardModel] Successfully loaded: {model_name} on {self.device}")
        except Exception as e:
            print(f"[ERROR] Failed to load reward model: {model_name}")
            print(f"[ERROR] {e}")
            print("[WARNING] Using dummy scorer (constant 0.5 scores) - install model manually if needed")
            self.model = None
            self.tokenizer = None

    def score(self, prompt: str, response: str) -> float:
        """
        Score a response given a prompt.

        Args:
            prompt: Input prompt
            response: Generated response

        Returns:
            Reward score (typically in range [-5, 5] or [0, 1] depending on model)
        """
        if self.model is None:
            # Return dummy score if model not loaded
            return 0.5

        # Format as prompt + response
        text = f"{prompt}\n{response}"

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length",
        ).to(self.device)

        # Score
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Return raw logits (unbounded, authentic model output)
            score = outputs.logits.squeeze().item()
            # # Apply sigmoid to normalize logits to [0, 1] range
            # score = torch.sigmoid(outputs.logits.squeeze()).item()

        return score

    def score_batch(self, prompts: List[str], responses: List[str]) -> List[float]:
        """
        Score multiple prompt-response pairs.

        Args:
            prompts: List of prompts
            responses: List of responses

        Returns:
            List of scores
        """
        scores = []
        for prompt, response in zip(prompts, responses):
            score = self.score(prompt, response)
            scores.append(score)
        return scores

    def score_response_only(self, response: str) -> float:
        """
        Score a response without prompt context.

        Args:
            response: Generated response

        Returns:
            Reward score (real scores if model loaded, 0.5 dummy if not)
        """
        if self.model is None:
            return 0.5

        inputs = self.tokenizer(
            response,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Return raw logits (unbounded, authentic model output)
            score = outputs.logits.squeeze().item()
            # # Apply sigmoid to normalize logits to [0, 1] range
            # score = torch.sigmoid(outputs.logits.squeeze()).item()

        return score

    def is_dummy_scorer(self) -> bool:
        """Check if using dummy scorer (model not loaded)."""
        return self.model is None
