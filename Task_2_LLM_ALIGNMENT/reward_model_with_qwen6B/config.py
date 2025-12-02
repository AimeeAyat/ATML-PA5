"""
Configuration for Reward Model Training
"""

from trl import RewardConfig
import torch

# Model and Dataset Configuration
MODEL_ID = "Qwen/Qwen3-0.6B"
DATASET_NAME = "trl-lib/ultrafeedback_binarized"
DATASET_SPLIT = "train"

# Training Configuration (Optimized for RTX 5090)
TRAINING_ARGS = RewardConfig(
    output_dir="./reward_model_output",
    per_device_train_batch_size=4,  # Reduced due to model size
    per_device_eval_batch_size=4,
    num_train_epochs=2,
    learning_rate=1.41e-5,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    optim="adamw_torch",  # Use standard AdamW on RTX 5090
    bf16=True,  # RTX 5090 supports bfloat16
    logging_steps=10,
    save_steps=2912,  # Save every ~25% of training
    save_total_limit=1,  # Keep only the most recent checkpoint, auto-delete others
    gradient_accumulation_steps=4,  # Accumulate 4 steps to simulate batch_size=16
    gradient_checkpointing=True,  # Save memory during backprop
    remove_unused_columns=False,
    report_to=["tensorboard"],
    center_rewards_coefficient=1e-2,  # Encourage rewards centered around zero
)

# LoRA Configuration (for parameter-efficient fine-tuning)
LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "SEQ_CLS",  # Sequence Classification
}

# Dataset columns mapping
DATASET_COLUMNS = {
    "prompt": "prompt",
    "chosen": "chosen",
    "rejected": "rejected",
}
