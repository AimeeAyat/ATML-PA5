"""
Configuration for PPO Training on ORCA Dataset
Model: SmolLM2-135M-SFT-Only (LLM to fine-tune)
Dataset: ORCA DPO Pairs (prompts)
Reward Model: SmolLM2-135M trained on ORCA DPO
"""

from trl import PPOConfig
import torch

# Model and Dataset Configuration
LLM_MODEL_ID = "HuggingFaceTB/smollm2-135M-SFT-Only"
DATASET_NAME = "Intel/orca_dpo_pairs"
DATASET_SPLIT = "train"

# Reward Model Configuration (trained earlier)
import os
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REWARD_MODEL_ID = os.path.join(_base_dir, "reward_model_on_orca_dataset", "reward_model_orca_output", "checkpoint-2160")

# PPO Training Configuration (Optimized for RTX 5090)
PPO_CONFIG = PPOConfig(
    # Output and checkpointing
    output_dir="./ppo_training_output",
    logging_dir="./ppo_training_output/logs",

    # Training hyperparameters
    num_train_epochs=3,
    learning_rate=5e-6,  # Conservative learning rate for stability
    lr_scheduler_type="cosine",

    # Batch settings
    per_device_train_batch_size=4,  # Prompts per batch (same as GRPO for memory)

    # PPO-specific parameters
    mini_batch_size=1,  # Number of mini-batches for PPO update
    gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps
    num_ppo_epochs=4,  # Number of epochs for PPO update

    # Optimization
    optim="adamw_torch",
    weight_decay=0.01,
    gradient_checkpointing=True,  # Memory optimization
    bf16=False,  # Use fp32 for compatibility

    # Checkpointing and logging
    save_strategy="epoch",  # Save at end of each epoch
    save_total_limit=1,  # Keep only the latest (final) checkpoint
    logging_steps=10,  # Log metrics every 10 steps
    logging_strategy="steps",  # Log during training for TensorBoard

    # Evaluation
    eval_strategy="no",  # No separate evaluation dataset

    # Reporting
    report_to=["tensorboard"],
    push_to_hub=False,

    # Reproducibility
    seed=42,
)

# LoRA Configuration (for parameter-efficient fine-tuning)
LORA_CONFIG = {
    "r": 16,  # LoRA rank
    "lora_alpha": 32,  # LoRA scaling
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",  # Language modeling task
    "target_modules": ["q_proj", "v_proj"],  # Attention projections
}

# Dataset processing parameters
DATASET_CONFIG = {
    "max_samples": None,  # None = use all samples
    "test_size": 0.1,  # 10% for evaluation (optional)
    "random_seed": 42,
}
