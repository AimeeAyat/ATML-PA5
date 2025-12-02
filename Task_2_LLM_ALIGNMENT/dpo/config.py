"""
Configuration for DPO Training on ORCA Dataset
Model: SmolLM2-135M-SFT-Only (LLM to fine-tune)
Dataset: ORCA DPO Pairs (has chosen/rejected responses)
"""

from trl import DPOConfig
import torch

# Model and Dataset Configuration
LLM_MODEL_ID = "HuggingFaceTB/smollm2-135M-SFT-Only"
DATASET_NAME = "Intel/orca_dpo_pairs"
DATASET_SPLIT = "train"

# DPO Training Configuration (Optimized for RTX 5090)
DPO_CONFIG = DPOConfig(
    # Output and checkpointing
    output_dir="./dpo_training_output",
    logging_dir="./dpo_training_output/logs",

    # Training hyperparameters
    num_train_epochs=3,
    learning_rate=5e-6,  # Conservative learning rate for 135M model
    lr_scheduler_type="cosine",
    warmup_steps=100,

    # Batch settings
    per_device_train_batch_size=8,  # Can be larger than PPO/GRPO (no generation)
    per_device_eval_batch_size=8,

    # DPO-specific parameters
    beta=0.5,  # Temperature controlling preference optimization strength (0.5 is standard)
    label_smoothing=0.0,  # No label smoothing needed for DPO

    # Optimization
    optim="adamw_torch",
    weight_decay=0.01,
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,  # Memory optimization
    bf16=False,  # Use fp32 for compatibility
    fp16=False,  # Disable mixed precision

    # Checkpointing and logging
    save_strategy="no",  # Don't save checkpoints - only save at end
    logging_steps=10,  # Log metrics every 10 steps
    logging_strategy="steps",  # Log during training for TensorBoard

    # Evaluation
    eval_strategy="no",  # No validation set

    # Reporting
    report_to=["tensorboard"],
    push_to_hub=False,

    # Reproducibility
    seed=42,

    # Other important settings
    max_prompt_length=512,  # Max prompt tokens
    max_length=1024,  # Max total (prompt + completion) tokens
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
    "test_size": 0.1,  # 10% for evaluation (not used for training, but for splits)
    "random_seed": 42,
}
