"""
Greedy Search decoding strategy implementation.
"""

import torch
from typing import Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer


class GreedySearchDecoder:
    """Greedy search decoder - selects highest probability token at each step."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
        """
        Initialize greedy search decoder.

        Args:
            model: Language model
            tokenizer: Tokenizer for the model
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def decode(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 1.0,
    ) -> Tuple[str, torch.Tensor]:
        """
        Decode using greedy search.

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

        # Get EOS token ID
        eos_token_id = self.tokenizer.eos_token_id

        log_probs = []
        current_ids = input_ids.clone()

        # Generate tokens sequentially (ensure at least 50 tokens minimum)
        min_new_tokens = 50
        num_tokens_to_generate = max(min_new_tokens, max_length - input_length)
        for _ in range(num_tokens_to_generate):
            with torch.no_grad():
                # Forward pass
                outputs = self.model(current_ids)
                logits = outputs.logits[:, -1, :]  # Get logits for last token

            # Apply temperature
            logits = logits / temperature

            # Get probabilities
            probs = torch.softmax(logits, dim=-1)

            # Select token with highest probability (greedy)
            next_token_id = torch.argmax(probs, dim=-1, keepdim=True)

            # Store log probability
            log_prob = torch.log(probs.gather(-1, next_token_id))
            log_probs.append(log_prob.item())

            # Append to sequence
            current_ids = torch.cat([current_ids, next_token_id], dim=1)

            # Check for EOS token
            if next_token_id.item() == eos_token_id:
                break

        # Decode to text (don't skip special tokens, we'll handle EOS manually)
        generated_tokens = current_ids[0, input_length:]

        # If only EOS token was generated or empty, that's a problem - try with skip_special_tokens=True
        if len(generated_tokens) == 0:
            generated_text = ""
        else:
            # Decode with skip_special_tokens=True to remove EOS automatically
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # If that also resulted in empty or only whitespace, try without skipping
            if not generated_text.strip():
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=False)
                if self.tokenizer.eos_token:
                    generated_text = generated_text.replace(self.tokenizer.eos_token, "").strip()

        return generated_text, torch.tensor(log_probs)
