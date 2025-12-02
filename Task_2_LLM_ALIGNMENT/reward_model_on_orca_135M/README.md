# Reward Model Training on ORCA Dataset

This project implements reward model training using HuggingFace's TRL library with the ORCA DPO Pairs dataset.

## Overview

Train a reward model to score/rank responses based on ORCA instruction pairs.

**Model**: SmolLM2-135M-SFT (HuggingFaceTB/SmolLM2-135M-SFT)
**Dataset**: ORCA DPO Pairs (argilla/orca-dpo-pairs)
**Hardware**: RTX 5090 (32GB VRAM) - Windows 11

## Project Structure

```
reward_model_on_orca_dataset/
├── requirements.txt          # Project dependencies
├── config.py                 # Training configuration
├── train_orca.py            # Training script
├── evaluate_rewards.py       # Evaluation script
└── README.md                 # This file
```

## Installation

1. **Create virtual environment**:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Train the Model

**Standard training** (full model fine-tuning):
```bash
python train_orca.py
```

**With LoRA** (parameter-efficient fine-tuning):
```bash
python train_orca.py --use-lora
```

Training output saved to `./reward_model_orca_output/`

### Step 2: Evaluate the Trained Model

Score responses using the trained reward model:

```bash
python evaluate_rewards.py
```

## Model & Dataset

### Model: SmolLM2-135M-SFT
- Lightweight but capable model (~135M parameters)
- Good for efficient reward modeling
- Better reasoning capabilities than 0.6B models

### Dataset: ORCA DPO Pairs
- High-quality instruction-following pairs
- Based on ORCA dataset with preference annotations
- Optimized for DPO (Direct Preference Optimization)
- ~80k+ instruction pairs

## Configuration

Edit `config.py` to customize:

**Batch Size Tuning**:
- Current: `per_device_train_batch_size=8`
- Increase if VRAM allows, decrease if OOM

**Learning Rate**:
- Current: `1.41e-5` (TRL default)
- Decrease for more stable training

**Training Duration**:
- `num_train_epochs=3`
- `save_steps=1500` (save checkpoints)
- `save_total_limit=1` (keep only latest)

**LoRA Settings** (if using --use-lora):
- `r=16` (rank)
- `lora_alpha=32`
- More aggressive LoRA than Qwen config

## Training Statistics

- **Dataset size**: ~80k samples
- **Train samples** (90%): ~72k
- **Eval samples** (10%): ~8k
- **Batch size**: 8 effective (8 × 1 accumulation steps)
- **Total steps**: ~27k per epoch
- **Checkpoints**: Save every 1500 steps (~1-2 per epoch)
- **Expected time**: 20-40 minutes (3 epochs)

## Key Differences from Qwen config

| Aspect | Qwen3-0.6B | SmolLM2-135M |
|--------|-----------|------------|
| Model size | 0.6B | 135M |
| Batch size | 4 | 8 |
| LoRA rank | 8 | 16 |
| Accumulation steps | 4 | 2 |
| Dataset | UltraFeedback | ORCA DPO |

## Next Steps

After training:
1. Use trained model to score ORCA responses
2. Compare with baseline reward models
3. Fine-tune LLMs using this reward signal in RLHF
4. Evaluate on downstream tasks

## References

- [TRL RewardTrainer](https://huggingface.co/docs/trl/main/en/reward_trainer)
- [ORCA Dataset](https://huggingface.co/datasets/argilla/orca-dpo-pairs)
- [SmolLM2 Model](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-SFT)
