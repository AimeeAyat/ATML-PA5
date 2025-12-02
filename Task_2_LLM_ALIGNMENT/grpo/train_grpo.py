"""
GRPO Training Script
Trains SmolLM2-135M LLM using Group Relative Policy Optimization
with the trained reward model as the reward signal
"""

import os
import torch
import argparse
import warnings
try:
    import wandb
except ImportError:
    wandb = None
from config import GRPO_CONFIG, LORA_CONFIG, LLM_MODEL_ID, DATASET_NAME, DATASET_SPLIT, REWARD_MODEL_ID
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
from trl import GRPOTrainer
from peft import LoraConfig, get_peft_model


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


def load_dataset_orca():
    """
    Load ORCA DPO dataset and extract prompts

    Returns:
        dataset: Dataset with 'prompt' column for GRPO
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
        print(f"Using first {len(dataset)} samples for training")

    return dataset


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


def train_grpo(use_lora=False):
    """
    Train the LLM using GRPO

    Args:
        use_lora: Whether to use LoRA for efficient training
    """
    print("=" * 80)
    print("GRPO TRAINING - SmolLM2-135M with Trained Reward Model")
    print("=" * 80)

    # 1. Setup model and tokenizer
    print("\n[1/5] Setting up model and tokenizer...")
    if use_lora:
        tokenizer, peft_config = setup_model_and_tokenizer(use_lora=True)
        model = None  # GRPOTrainer will load it
    else:
        tokenizer, model = setup_model_and_tokenizer(use_lora=False)
        peft_config = None

    # 2. Load reward model
    print("\n[2/5] Loading reward model...")
    reward_model = load_reward_model()

    # 3. Load dataset
    print("\n[3/5] Loading and preparing dataset...")
    train_dataset = load_dataset_orca()

    # 4. Create reward function
    print("\n[4/5] Creating reward function...")
    reward_fn = create_reward_function(reward_model, tokenizer)

    # 5. Initialize and train
    print("\n[5/5] Initializing GRPOTrainer...")

    trainer = GRPOTrainer(
        model=model if not use_lora else LLM_MODEL_ID,
        args=GRPO_CONFIG,
        train_dataset=train_dataset,
        reward_funcs=[reward_fn],  # List of reward functions (can have multiple)
        processing_class=tokenizer,
        peft_config=peft_config if use_lora else None,
    )

    print("\nTraining configuration:")
    print(f"  - Model: {LLM_MODEL_ID}")
    print(f"  - Reward Model: {REWARD_MODEL_ID}")
    print(f"  - Dataset: {DATASET_NAME}")
    print(f"  - Batch size: {GRPO_CONFIG.per_device_train_batch_size}")
    print(f"  - Completions per prompt (G): {GRPO_CONFIG.num_generations}")
    print(f"  - Gradient updates per generation (mu): {GRPO_CONFIG.num_iterations}")
    print(f"  - Learning rate: {GRPO_CONFIG.learning_rate}")
    print(f"  - Max completion length: {GRPO_CONFIG.max_completion_length}")
    print(f"  - LoRA: {'Yes' if use_lora else 'No'}")
    print(f"  - Output dir: {GRPO_CONFIG.output_dir}")

    print("\n" + "=" * 80)
    print("Starting GRPO training...")
    print("=" * 80 + "\n")

    # Train
    trainer.train()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)

    # Save the trained model
    print(f"\nSaving trained model to: {GRPO_CONFIG.output_dir}")
    trainer.save_model(GRPO_CONFIG.output_dir)

    # Save tokenizer
    tokenizer.save_pretrained(GRPO_CONFIG.output_dir)
    print(f"Model and tokenizer saved to: {GRPO_CONFIG.output_dir}")
    print(f"TensorBoard logs: {GRPO_CONFIG.logging_dir}")

    return trainer, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LLM using GRPO")
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Use LoRA for parameter-efficient training"
    )
    args = parser.parse_args()

    trainer, tokenizer = train_grpo(use_lora=args.use_lora)
