# Reward Model Training Pipeline

This project implements a reward model training pipeline using HuggingFace's TRL library based on the official documentation.

## Overview

A reward model learns to score/rank responses based on human preferences. It's commonly used in RLHF (Reinforcement Learning from Human Feedback) pipelines.

**Model**: Qwen/Qwen3-0.6B
**Dataset**: trl-lib/ultrafeedback_binarized (contains preference data with chosen/rejected responses)
**Hardware**: RTX 5090 (32GB VRAM) - Windows 11

## Project Structure

```
reward_model/
├── requirements.txt          # Project dependencies
├── config.py                 # Training configuration and hyperparameters
├── inspect_dataset.py        # Script to inspect dataset structure
├── train_reward_model.py     # Basic training script (minimal example)
├── train_advanced.py         # Advanced training with configuration support
├── evaluate_rewards.py       # Script to evaluate/score responses
└── README.md                 # This file
```

## Installation

1. **Create a virtual environment** (optional but recommended):
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Inspect the Dataset

Before training, inspect the dataset structure:

```bash
python inspect_dataset.py
```

This shows:
- Number of samples
- Column names and types
- Example data samples
- Dataset statistics

### Step 2: Train the Reward Model

**Option A - Basic Training** (simplest):
```bash
python train_reward_model.py
```

**Option B - Advanced Training** (with configuration):
```bash
# Standard training (uses RTX 5090 optimized config)
python train_advanced.py

# With LoRA (parameter-efficient, not needed on RTX 5090)
python train_advanced.py --use-lora
```

Training output will be saved to `./reward_model_trained/`

### Step 3: Evaluate the Trained Model

Score responses using the trained reward model:

```bash
python evaluate_rewards.py
```

This will:
- Load the trained model
- Score example responses
- Compare two responses and determine which is better
- Show confidence margins

## RTX 5090 Optimization

The configuration is optimized for RTX 5090:
- **Batch Size**: 32 (leverages 32GB VRAM)
- **Optimizer**: adamw_torch (standard, no paging needed)
- **Gradient Checkpointing**: Disabled (not needed with large VRAM)
- **Gradient Accumulation**: 1 (large batch size already sufficient)
- **Training Epochs**: 3 (can afford more iterations)
- **Mixed Precision**: bfloat16 (RTX 5090 supports natively)

## How It Works

### Dataset Format

The ultrafeedback_binarized dataset contains:
- **prompt**: The input question/instruction
- **chosen**: The preferred/better response
- **rejected**: The less preferred response

### Training Process

1. RewardTrainer loads the model and dataset
2. For each (prompt, chosen, rejected) triple:
   - Encodes all three into token sequences
   - Computes reward scores for chosen and rejected
   - Uses Bradley-Terry loss to maximize score difference
3. Model learns to assign higher scores to preferred responses

### Training Hyperparameters (in config.py)

- **learning_rate**: 1.41e-5
- **batch_size**: 32 (RTX 5090)
- **epochs**: 3
- **warmup_steps**: 100
- **optimizer**: adamw_torch
- **precision**: bfloat16

### Output

The trained model outputs a single scalar value (reward score) for each input, where:
- Higher scores = better responses
- Can be used to compare and rank responses

## Configuration

Edit `config.py` to customize:
- Model: Change `MODEL_ID` to use a different base model
- Dataset: Change `DATASET_NAME` to use different preference data
- Training: Adjust `TRAINING_ARGS` for different hyperparameters
- LoRA: Modify `LORA_CONFIG` for parameter-efficient training

## Next Steps

After training a reward model, you can:

1. **Use it in RLHF**: Train an LLM with this reward signal using PPO
2. **Evaluate Models**: Score outputs from different models
3. **Fine-tune Further**: Continue training on your own preference data
4. **Integrate**: Use as a quality metric in production pipelines

## Troubleshooting

- **CUDA not available**: Make sure NVIDIA drivers are installed
- **Dataset loading fails**: Check internet connection for HuggingFace Hub
- **Memory issues**: Reduce batch_size in config.py (though unlikely on RTX 5090)

## References

- [TRL RewardTrainer Docs](https://huggingface.co/docs/trl/main/en/reward_trainer)
- [UltraFeedback Dataset](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized)
- [Qwen3 Model](https://huggingface.co/Qwen/Qwen3-0.6B)
