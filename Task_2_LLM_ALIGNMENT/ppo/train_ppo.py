"""
PPO Training Script
Trains SmolLM2-135M LLM using Proximal Policy Optimization
with the trained reward model as the reward signal
"""

import os
import torch
import argparse
from config import PPO_CONFIG, LORA_CONFIG, LLM_MODEL_ID, DATASET_NAME, DATASET_SPLIT, REWARD_MODEL_ID
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
from trl import PPOTrainer
from peft import LoraConfig, get_peft_model
import torch.nn as nn


class ValueModelWrapper(nn.Module):
    """Wrapper that adds a score() method to a CausalLM for PPO compatibility

    This wrapper is trainable - the base model is frozen and only the value head is trained.
    This is memory-efficient and prevents catastrophic forgetting of the language model.
    """
    def __init__(self, base_model, freeze_base=True):
        super().__init__()
        self.base_model = base_model
        self.config = base_model.config

        # Set base_model_prefix for PPOTrainer compatibility
        self.base_model_prefix = "base_model"

        # Freeze base model parameters if requested (for value estimation only)
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Add a trainable scoring head: hidden_size -> 1 (scalar value)
        hidden_size = base_model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)

        # Initialize value head with small weights for stability
        nn.init.xavier_uniform_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass through base model

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            **kwargs: Other arguments to pass to base model

        Returns:
            Model output with hidden states
        """
        # Ensure we get hidden states for scoring
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs
        )

    def score(self, hidden_states):
        """Score method expected by TRL's PPOTrainer

        Estimates the value (expected reward) for a given state.

        Args:
            hidden_states: Last hidden state from the model
                Shape: (batch_size, seq_len, hidden_size) or (batch_size, hidden_size)

        Returns:
            Value predictions
            Shape: (batch_size, seq_len) - expanded across sequence length for indexing
        """
        # Take the last token's hidden state for value estimation
        # This represents the final state after processing the entire sequence
        if hidden_states.dim() == 3:
            # (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size)
            batch_size, seq_len, _ = hidden_states.shape
            last_hidden = hidden_states[:, -1, :]
        else:
            # Already (batch_size, hidden_size)
            batch_size = hidden_states.shape[0]
            seq_len = 1
            last_hidden = hidden_states

        # Apply value head: (batch_size, hidden_size) -> (batch_size, 1)
        value = self.value_head(last_hidden)  # shape: (batch_size, 1)

        # Expand value across sequence positions for PPOTrainer.get_reward() indexing
        # PPOTrainer indexes by sequence_length position, so we need (batch_size, seq_len)
        # We repeat the same value for all positions (common in simple value models)
        value_per_token = value.expand(batch_size, seq_len)  # (batch_size, seq_len)

        return value_per_token

    def get_trainable_params(self):
        """Get only trainable parameters (value head)"""
        return [p for p in self.parameters() if p.requires_grad]

    def get_base_model_output(self, *args, **kwargs):
        """Get base model output (for compatibility if needed)"""
        return self.base_model(*args, **kwargs)


def setup_model_and_tokenizer(use_lora=False):
    """
    Setup the LLM tokenizer and load the base model

    Args:
        use_lora: Whether to apply LoRA for parameter-efficient training

    Returns:
        tokenizer, model (if use_lora=False) or peft_config (if use_lora=True)
    """
    print(f"\nLoading tokenizer from {LLM_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_ID,
        padding_side="left",  # Left-pad for generation
    )

    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizer loaded successfully!")
    print(f"  - Vocab size: {len(tokenizer)}")
    print(f"  - Padding token: {tokenizer.pad_token}")

    if use_lora:
        print(f"\nApplying LoRA configuration...")
        peft_config = LoraConfig(**LORA_CONFIG)
        print(f"  - LoRA rank (r): {LORA_CONFIG['r']}")
        print(f"  - LoRA alpha: {LORA_CONFIG['lora_alpha']}")
        print(f"  - Target modules: {LORA_CONFIG['target_modules']}")
        return tokenizer, peft_config
    else:
        print(f"\nLoading LLM: {LLM_MODEL_ID}...")
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            torch_dtype=torch.float32,
            device_map="auto",
        )
        print(f"LLM loaded successfully!")
        print(f"  - Model size: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
        return tokenizer, model


def load_reward_model():
    """
    Load the trained reward model

    Returns:
        reward_model: The trained reward model for scoring completions
    """
    print(f"\nLoading reward model from {REWARD_MODEL_ID}...")

    # Check if path is local or HuggingFace ID
    if os.path.exists(REWARD_MODEL_ID):
        print(f"Loading from local path: {REWARD_MODEL_ID}")
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            REWARD_MODEL_ID,
            torch_dtype=torch.float32,
            device_map="auto",
            num_labels=1,  # Reward model outputs single scalar
        )
    else:
        print(f"Loading from HuggingFace Hub: {REWARD_MODEL_ID}")
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            REWARD_MODEL_ID,
            torch_dtype=torch.float32,
            device_map="auto",
            num_labels=1,
        )

    reward_model.eval()
    print(f"Reward model loaded successfully!")
    return reward_model


def preprocess_function(examples, tokenizer):
    """Tokenize prompts for PPO"""
    return tokenizer(
        examples["prompt"],
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )


def load_dataset_orca(tokenizer):
    """
    Load ORCA DPO dataset, extract prompts, and split into train/eval

    Returns:
        train_dataset, eval_dataset: Tokenized train and eval datasets with 'input_ids' and 'attention_mask'
    """
    print(f"\nLoading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    print(f"Dataset loaded successfully!")
    print(f"  - Total samples: {len(dataset)}")
    print(f"  - Columns: {dataset.column_names}")

    # ORCA DPO dataset has specific column structure
    # Inspect first sample
    if len(dataset) > 0:
        print(f"\nFirst sample columns: {list(dataset[0].keys())}")

    # Extract prompts (assuming 'prompt' or 'instruction' column)
    # ORCA DPO Pairs typically has: system, question, chosen, rejected
    if 'prompt' in dataset.column_names:
        prompt_col = 'prompt'
    elif 'question' in dataset.column_names:
        prompt_col = 'question'
    elif 'instruction' in dataset.column_names:
        prompt_col = 'instruction'
    else:
        # Use first column as fallback
        prompt_col = dataset.column_names[0]
        print(f"Warning: Could not find 'prompt' column, using '{prompt_col}'")

    # Select only prompt column and rename if needed
    dataset = dataset.select_columns([prompt_col])
    dataset = dataset.rename_columns({prompt_col: "prompt"})

    print(f"Extracted {len(dataset)} prompts")
    print(f"Sample prompt: {dataset[0]['prompt'][:100]}...")

    # Optional: limit samples for faster testing
    if DATASET_NAME == "Intel/orca_dpo_pairs":
        # ORCA DPO has ~80k samples, using subset for faster iteration
        dataset = dataset.select(range(min(10000, len(dataset))))
        print(f"Using first {len(dataset)} samples for training/eval")

    # Split into train/eval (90/10 split)
    print(f"\nSplitting dataset into train/eval (90/10)...")
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    print(f"  - Train samples: {len(train_dataset)}")
    print(f"  - Eval samples: {len(eval_dataset)}")

    # Tokenize train dataset
    print(f"\nTokenizing train dataset...")
    train_dataset = train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=["prompt"],
    )

    # Tokenize eval dataset
    print(f"Tokenizing eval dataset...")
    eval_dataset = eval_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=["prompt"],
    )

    print(f"Tokenization complete!")
    return train_dataset, eval_dataset


def create_reward_function(reward_model, tokenizer):
    """
    Create a reward function that uses the trained reward model

    Args:
        reward_model: The trained reward model
        tokenizer: Tokenizer for processing completions

    Returns:
        reward_fn: Function that computes rewards for completions
    """
    def reward_fn(prompts, completions, **kwargs):
        """
        Compute rewards for completions using the reward model

        Args:
            prompts: List of prompt strings
            completions: List of completion strings
            **kwargs: Additional arguments (ignored)

        Returns:
            rewards: List of reward scores (floats)
        """
        # Combine prompts and completions
        full_texts = [p + c for p, c in zip(prompts, completions)]

        # Tokenize
        inputs = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )

        # Move to device
        inputs = {k: v.to(reward_model.device) for k, v in inputs.items()}

        # Get rewards
        with torch.no_grad():
            outputs = reward_model(**inputs)
            rewards = outputs.logits.squeeze(-1).cpu().tolist()

        return rewards

    return reward_fn


def train_ppo(use_lora=False):
    """
    Train the LLM using PPO

    Args:
        use_lora: Whether to use LoRA for efficient training
    """
    print("=" * 80)
    print("PPO TRAINING - SmolLM2-135M with Trained Reward Model")
    print("=" * 80)

    # 1. Setup model and tokenizer
    print("\n[1/4] Setting up model and tokenizer...")
    tokenizer, model_or_peft = setup_model_and_tokenizer(use_lora=use_lora)

    if use_lora:
        # Load base model and apply LoRA
        peft_config = model_or_peft
        model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_ID,
            torch_dtype=torch.float32,
            device_map="auto",
        )
        model = get_peft_model(model, peft_config)
    else:
        model = model_or_peft
        peft_config = None

    # Create reference model (frozen copy) - always needed for PPO
    ref_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        torch_dtype=torch.float32,
        device_map="auto",
    )

    # Value model with wrapper for PPO compatibility
    # Wrap base model to provide required score() method
    base_value_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_ID,
        torch_dtype=torch.float32,
        device_map="auto",
    )
    value_model = ValueModelWrapper(base_value_model, freeze_base=True)

    # 2. Load reward model
    print("\n[2/4] Loading reward model...")
    reward_model = load_reward_model()

    # 3. Load dataset
    print("\n[3/4] Loading and preparing dataset...")
    train_dataset, eval_dataset = load_dataset_orca(tokenizer)

    # 4. Initialize and train
    print("\n[4/4] Initializing PPOTrainer...")

    trainer = PPOTrainer(
        model=model,
        ref_model=ref_model,
        reward_model=reward_model,
        value_model=value_model,
        args=PPO_CONFIG,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Use separate eval dataset for validation
        processing_class=tokenizer,
        peft_config=peft_config if use_lora else None,
    )

    print("\nTraining configuration:")
    print(f"  - Model: {LLM_MODEL_ID}")
    print(f"  - Reward Model: {REWARD_MODEL_ID}")
    print(f"  - Dataset: {DATASET_NAME}")
    print(f"  - Batch size: {PPO_CONFIG.per_device_train_batch_size}")
    print(f"  - PPO epochs per update: {PPO_CONFIG.num_ppo_epochs}")
    print(f"  - Mini batch size: {PPO_CONFIG.mini_batch_size}")
    print(f"  - Gradient accumulation: {PPO_CONFIG.gradient_accumulation_steps}")
    print(f"  - Learning rate: {PPO_CONFIG.learning_rate}")
    print(f"  - Training epochs: {PPO_CONFIG.num_train_epochs}")
    print(f"  - LoRA: {'Yes' if use_lora else 'No'}")
    print(f"  - Output dir: {PPO_CONFIG.output_dir}")

    print("\n" + "=" * 80)
    print("Starting PPO training...")
    print("=" * 80 + "\n")

    # Train
    trainer.train()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)

    # Save the trained model
    print(f"\nSaving trained model to: {PPO_CONFIG.output_dir}")
    trainer.save_model(PPO_CONFIG.output_dir)

    # Save tokenizer
    tokenizer.save_pretrained(PPO_CONFIG.output_dir)
    print(f"Model and tokenizer saved to: {PPO_CONFIG.output_dir}")
    print(f"TensorBoard logs: {PPO_CONFIG.logging_dir}")

    return trainer, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LLM using PPO")
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Use LoRA for parameter-efficient training"
    )
    args = parser.parse_args()

    trainer, tokenizer = train_ppo(use_lora=args.use_lora)
