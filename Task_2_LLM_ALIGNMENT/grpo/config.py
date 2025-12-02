"""
Configuration for GRPO Training on ORCA Dataset
Model: SmolLM2-135M-SFT-Only (LLM to fine-tune)
Dataset: ORCA DPO Pairs (prompts)
Reward Model: SmolLM2-135M trained on ORCA DPO
"""

from trl import GRPOConfig
import torch

# Model and Dataset Configuration
LLM_MODEL_ID = "HuggingFaceTB/smollm2-135M-SFT-Only"
DATASET_NAME = "Intel/orca_dpo_pairs"
DATASET_SPLIT = "train"

# Reward Model Configuration (trained earlier)
import os
_base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REWARD_MODEL_ID = os.path.join(_base_dir, "reward_model_on_orca_dataset", "reward_model_orca_output", "checkpoint-2160")

# GRPO Training Configuration (Optimized for RTX 5090)
GRPO_CONFIG = GRPOConfig(
    # Output and checkpointing
    output_dir="./grpo_training_output",
    logging_dir="./grpo_training_output/logs",

    # Training hyperparameters
    num_train_epochs=3,
    learning_rate=5e-6,  # Conservative learning rate for 135M model
    lr_scheduler_type="cosine",
    warmup_steps=100,

    # Batch and generation settings
    per_device_train_batch_size=8,  # Prompts per batch (increased for efficiency)
    per_device_eval_batch_size=8,
    num_generations=2,  # G: completions per prompt (REDUCED from 4 - was too slow)
    num_iterations=2,  # μ: gradient updates per generation cycle (increased to compensate)

    # Generation parameters
    max_prompt_length=512,  # Max prompt tokens
    max_completion_length=256,  # Max generated tokens per completion
    temperature=0.7,  # Sampling temperature
    top_p=0.9,  # Nucleus sampling

    # Optimization
    optim="adamw_torch",
    weight_decay=0.01,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,  # Memory optimization
    bf16=False,  # Use fp32 for compatibility
    fp16=False,  # Disable mixed precision

    # Policy control
    beta=0.0,  # KL divergence weight (disabled - not needed)
    epsilon=0.2,  # Clipping range

    # Checkpointing and logging
    save_strategy="epoch",  # Save at end of each epoch
    save_total_limit=3,  # Keep all 3 epoch checkpoints for comparison
    logging_steps=10,  # Log metrics every 10 steps
    logging_strategy="steps",  # Log during training for TensorBoard

    # Evaluation (optional)
    eval_strategy="no",  # No validation set

    # Reporting
    report_to=["tensorboard"],
    push_to_hub=False,
    hub_strategy="no",  # Disable model card creation to avoid wandb issues

    # Reproducibility
    seed=42,
)

# LoRA Configuration (for parameter-efficient fine-tuning)
LORA_CONFIG = {
    "r": 16,  # LoRA rank (larger for better adaptation)
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
