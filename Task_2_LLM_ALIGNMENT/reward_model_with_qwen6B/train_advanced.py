"""
Advanced Reward Model Training with Configuration
Uses TRL RewardTrainer with Qwen3-0.6B model and ultrafeedback_binarized dataset
"""

from trl import RewardTrainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from config import TRAINING_ARGS, MODEL_ID, DATASET_NAME, LORA_CONFIG, DATASET_SPLIT
import torch

def setup_model_and_tokenizer(use_lora=False):
    """Load model and tokenizer"""
    print(f"Loading model: {MODEL_ID}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA if requested
    peft_config = None
    if use_lora:
        print("Applying LoRA configuration...")
        peft_config = LoraConfig(
            r=LORA_CONFIG["r"],
            lora_alpha=LORA_CONFIG["lora_alpha"],
            lora_dropout=LORA_CONFIG["lora_dropout"],
            bias=LORA_CONFIG["bias"],
            task_type=TaskType.SEQ_CLS,
            target_modules=["q_proj", "v_proj"],  # Common for transformer models
        )

    # Note: RewardTrainer will load the model using TRAINING_ARGS.model_init_kwargs
    # so we don't load it here
    return tokenizer, peft_config


def load_dataset_split():
    """Load the ultrafeedback_binarized dataset and split into train/eval"""
    print(f"\nLoading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

    print(f"Dataset loaded successfully!")
    print(f"  - Total samples: {len(dataset)}")
    print(f"  - Columns: {dataset.column_names}")

    # Split into 90% train, 10% eval
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split["train"]
    eval_dataset = train_test_split["test"]

    print(f"\nDataset split:")
    print(f"  - Train samples: {len(train_dataset)}")
    print(f"  - Eval samples: {len(eval_dataset)}")

    return train_dataset, eval_dataset


def train_reward_model(use_lora=False):
    """Train the reward model"""

    print("=" * 80)
    print("REWARD MODEL TRAINING")
    print("=" * 80)

    # Setup tokenizer and LoRA config
    tokenizer, peft_config = setup_model_and_tokenizer(use_lora=use_lora)
    train_dataset, eval_dataset = load_dataset_split()

    # Initialize trainer (training only, keep last checkpoint)
    print("\nInitializing RewardTrainer...")
    trainer = RewardTrainer(
        model=MODEL_ID,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        args=TRAINING_ARGS,
        peft_config=peft_config,
    )

    # Train
    print("\nStarting training...")
    print(f"Training arguments:")
    print(f"  - Model: {MODEL_ID}")
    print(f"  - Learning rate: {TRAINING_ARGS.learning_rate}")
    print(f"  - Batch size: {TRAINING_ARGS.per_device_train_batch_size}")
    print(f"  - Epochs: {TRAINING_ARGS.num_train_epochs}")
    print(f"  - Output dir: {TRAINING_ARGS.output_dir}")
    print(f"  - Save strategy: Keep only the most recent checkpoint, auto-delete others")

    trainer.train()

    print("\nTraining completed successfully!")
    print(f"Final model saved to: {TRAINING_ARGS.output_dir}")
    print(f"Old checkpoints automatically deleted.")

    return trainer, tokenizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train reward model")
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA for parameter-efficient training")
    args = parser.parse_args()

    trainer, tokenizer = train_reward_model(use_lora=args.use_lora)
