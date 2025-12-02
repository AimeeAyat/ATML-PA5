"""
Beam Search decoding strategy implementation.
"""

import torch
import heapq
from typing import Tuple, List
from transformers import PreTrainedModel, PreTrainedTokenizer


class Beam:
    """Single beam for beam search."""

    def __init__(self, token_ids: torch.Tensor, log_prob: float):
        """
        Initialize a beam.

        Args:
            token_ids: Sequence of token IDs
            log_prob: Cumulative log probability
        """
        self.token_ids = token_ids
        self.log_prob = log_prob

    def __lt__(self, other):
        """Compare beams by log probability (for heap)."""
        return self.log_prob > other.log_prob

    def __repr__(self):
        return f"Beam(log_prob={self.log_prob:.4f})"


class BeamSearchDecoder:
    """Beam search decoder - maintains B most promising sequences."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, beam_width: int = 5):
        """
        Initialize beam search decoder.

        Args:
            model: Language model
            tokenizer: Tokenizer for the model
            beam_width: Number of beams to maintain
        """
        self.model = model
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.device = next(model.parameters()).device

    def decode(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 1.0,
    ) -> Tuple[str, torch.Tensor]:
        """
        Decode using beam search.

        Args:
            prompt: Input prompt
            max_length: Maximum length of generated sequence
            temperature: Temperature for scaling logits

        Returns:
            Tuple of (generated_text, log_probabilities)
        """
        # Format prompt with chat template if available
        if self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, return_tensors="pt", add_generation_prompt=True
            ).to(self.device)
        else:
            # Fallback to plain tokenization
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        input_length = input_ids.shape[1]
        eos_token_id = self.tokenizer.eos_token_id

        # Initialize beams
        beams = [Beam(input_ids.clone(), 0.0)]
        completed_beams = []

        # Generate tokens iteratively (ensure at least 50 tokens minimum)
        min_new_tokens = 50
        num_tokens_to_generate = max(min_new_tokens, max_length - input_length)
        for step in range(num_tokens_to_generate):
            candidates = []

            for beam in beams:
                if beam.token_ids.shape[1] > input_length:
                    # Check if last token is EOS
                    if beam.token_ids[0, -1].item() == eos_token_id:
                        completed_beams.append(beam)
                        continue

                with torch.no_grad():
                    # Forward pass
                    outputs = self.model(beam.token_ids)
                    logits = outputs.logits[:, -1, :]  # Last token logits

                # Apply temperature
                logits = logits / temperature

                # Get log probabilities
                log_probs = torch.log_softmax(logits, dim=-1)

                # Get top-k candidates
                top_log_probs, top_token_ids = torch.topk(log_probs, self.beam_width, dim=-1)

                # Create candidates
                for i in range(self.beam_width):
                    new_token_id = top_token_ids[0, i]
                    new_log_prob = beam.log_prob + top_log_probs[0, i].item()
                    new_token_ids = torch.cat([beam.token_ids, new_token_id.unsqueeze(0).unsqueeze(0)], dim=1)

                    candidates.append(Beam(new_token_ids, new_log_prob))

            # Select top-B candidates
            candidates.sort()
            beams = candidates[: self.beam_width]

            # Check if all beams are completed
            if len(beams) == 0:
                break

        # Combine completed and active beams
        all_beams = completed_beams + beams
        all_beams.sort()

        # Return best beam
        best_beam = all_beams[0]
        generated_tokens = best_beam.token_ids[0, input_length:]

        # If only EOS token was generated, that's effectively empty - return empty string
        if len(generated_tokens) == 0:
            generated_text = ""
        else:
            # Try skip_special_tokens=True first
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            # If that resulted in empty or only whitespace, try without skipping
            if not generated_text.strip():
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
                if self.tokenizer.eos_token:
                    generated_text = generated_text.replace(self.tokenizer.eos_token, "").strip()

        return generated_text, torch.tensor([best_beam.log_prob])
