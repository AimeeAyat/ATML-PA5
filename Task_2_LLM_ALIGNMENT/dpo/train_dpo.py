"""
DPO Training Script
Trains SmolLM2-135M LLM using Direct Preference Optimization
with the ORCA DPO Pairs dataset (chosen vs rejected responses)
"""

import os
import torch
import argparse
from config import DPO_CONFIG, LORA_CONFIG, LLM_MODEL_ID, DATASET_NAME, DATASET_SPLIT
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import DPOTrainer
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
        padding_side="right",  # Right-pad for DPO (different from generation)
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


def load_dataset_orca(tokenizer):
    """
    Load ORCA DPO dataset with chosen/rejected response pairs

    Returns:
        dataset: Dataset with 'prompt', 'chosen', 'rejected' columns
    """
    print(f"\nLoading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    print(f"Dataset loaded successfully!")
    print(f"  - Total samples: {len(dataset)}")
    print(f"  - Columns: {dataset.column_names}")

    # ORCA DPO dataset has specific column structure
    if len(dataset) > 0:
        print(f"\nFirst sample columns: {list(dataset[0].keys())}")

    # ORCA DPO Pairs typically has: system, question, chosen, rejected
    # We need to format as: prompt, chosen, rejected

    # Identify the prompt column
    if 'prompt' in dataset.column_names:
        prompt_col = 'prompt'
    elif 'question' in dataset.column_names:
        prompt_col = 'question'
    elif 'instruction' in dataset.column_names:
        prompt_col = 'instruction'
    else:
        raise ValueError(f"Could not find prompt column. Available: {dataset.column_names}")

    # Check for response columns
    if 'chosen' not in dataset.column_names or 'rejected' not in dataset.column_names:
        raise ValueError(f"Dataset must have 'chosen' and 'rejected' columns. Available: {dataset.column_names}")

    # Select and rename columns to standard DPO format
    dataset = dataset.select_columns([prompt_col, 'chosen', 'rejected'])
    if prompt_col != 'prompt':
        dataset = dataset.rename_columns({prompt_col: 'prompt'})

    print(f"Dataset formatted with columns: {dataset.column_names}")
    print(f"Sample prompt: {dataset[0]['prompt'][:100]}...")
    print(f"Sample chosen: {dataset[0]['chosen'][:100]}...")
    print(f"Sample rejected: {dataset[0]['rejected'][:100]}...")

    # Optional: limit samples for faster testing
    if DATASET_NAME == "Intel/orca_dpo_pairs":
        # ORCA DPO has ~80k samples, using subset for faster iteration
        dataset = dataset.select(range(min(10000, len(dataset))))
        print(f"Using first {len(dataset)} samples for training")

    return dataset


def train_dpo(use_lora=False):
    """
    Train the LLM using DPO

    Args:
        use_lora: Whether to use LoRA for efficient training
    """
    print("=" * 80)
    print("DPO TRAINING - SmolLM2-135M with ORCA Dataset")
    print("=" * 80)

    # 1. Setup model and tokenizer
    print("\n[1/3] Setting up model and tokenizer...")
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
        print(f"Applied LoRA to model")
    else:
        model = model_or_peft
        peft_config = None

    # 2. Load dataset
    print("\n[2/3] Loading and preparing dataset...")
    train_dataset = load_dataset_orca(tokenizer)

    # 3. Initialize and train
    print("\n[3/3] Initializing DPOTrainer...")

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # DPO will auto-create reference model from base model
        args=DPO_CONFIG,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config if use_lora else None,
    )

    print("\nTraining configuration:")
    print(f"  - Model: {LLM_MODEL_ID}")
    print(f"  - Dataset: {DATASET_NAME}")
    print(f"  - Train samples: {len(train_dataset)}")
    print(f"  - Batch size: {DPO_CONFIG.per_device_train_batch_size}")
    print(f"  - Beta (preference strength): {DPO_CONFIG.beta}")
    print(f"  - Learning rate: {DPO_CONFIG.learning_rate}")
    print(f"  - Training epochs: {DPO_CONFIG.num_train_epochs}")
    print(f"  - LoRA: {'Yes' if use_lora else 'No'}")
    print(f"  - Output dir: {DPO_CONFIG.output_dir}")

    print("\n" + "=" * 80)
    print("Starting DPO training...")
    print("=" * 80 + "\n")

    # Train
    trainer.train()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)

    # Save the trained model
    print(f"\nSaving trained model to: {DPO_CONFIG.output_dir}")
    trainer.save_model(DPO_CONFIG.output_dir)

    # Save tokenizer
    tokenizer.save_pretrained(DPO_CONFIG.output_dir)
    print(f"Model and tokenizer saved to: {DPO_CONFIG.output_dir}")
    print(f"TensorBoard logs: {DPO_CONFIG.logging_dir}")

    return trainer, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LLM using DPO")
    parser.add_argument(
        "--use-lora",
        action="store_true",
        help="Use LoRA for parameter-efficient training"
    )
    args = parser.parse_args()

    trainer, tokenizer = train_dpo(use_lora=args.use_lora)
