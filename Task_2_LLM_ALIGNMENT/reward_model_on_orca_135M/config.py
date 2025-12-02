"""
Configuration for Reward Model Training on ORCA Dataset
Model: SmolLM2-135M-SFT-Only
Dataset: ORCA DPO Pairs
"""

from trl import RewardConfig
import torch

# Model and Dataset Configuration
MODEL_ID = "HuggingFaceTB/smollm2-135M-SFT-Only"
DATASET_NAME = "Intel/orca_dpo_pairs"
DATASET_SPLIT = "train"

# Training Configuration (Optimized for RTX 5090)
TRAINING_ARGS = RewardConfig(
    output_dir="./reward_model_orca_output",
    per_device_train_batch_size=8,  # 135M model, RTX 5090 can handle larger batches
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    learning_rate=1.41e-5,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    optim="adamw_torch",
    bf16=True,  # RTX 5090 supports bfloat16
    logging_steps=10,
    save_steps=1500,  # Save more frequently with larger dataset
    save_total_limit=1,  # Keep only the most recent checkpoint
    gradient_accumulation_steps=2,  # Larger batch size needs less accumulation
    gradient_checkpointing=True,  # Save memory during backprop
    remove_unused_columns=False,
    report_to=["tensorboard"],
    center_rewards_coefficient=1e-2,  # Encourage rewards centered around zero
)

# LoRA Configuration (for parameter-efficient fine-tuning)
LORA_CONFIG = {
    "r": 8,  # Larger rank for 135M model
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "SEQ_CLS",
}
