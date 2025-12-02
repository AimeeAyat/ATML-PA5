# GRPO Training: SmolLM2-135M with Trained Reward Model

Fine-tune SmolLM2-135M-SFT LLM using **Group Relative Policy Optimization (GRPO)** with the trained reward model as the preference signal.

## Overview

**GRPO** is an efficient reinforcement learning method for aligning LLMs to human preferences. It generates multiple completions per prompt, scores them with a reward model, and optimizes the policy using group-relative advantages.

**Setup:**
- **LLM**: SmolLM2-135M-SFT-Only (the model to fine-tune)
- **Dataset**: Intel/orca_dpo_pairs (for prompts)
- **Reward Model**: SmolLM2-135M trained on ORCA DPO
- **Hardware**: RTX 5090 (32GB VRAM) - Windows 11

## Project Structure

```
grpo_training/
├── requirements.txt          # Project dependencies
├── config.py                 # GRPO configuration
├── train_grpo.py            # Main GRPO training script
└── README.md                 # This file
```

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Verify reward model path**: Make sure the trained reward model exists at:
```
../reward_model_on_orca_dataset/reward_model_orca_output
```

## Usage

### Basic Training

```bash
python train_grpo.py
```

This will:
1. Load SmolLM2-135M-SFT-Only LLM
2. Load trained reward model from ORCA training
3. Load ORCA DPO dataset (first 10k samples for faster iteration)
4. Fine-tune LLM using GRPO for 3 epochs
5. Save trained model to `./grpo_training_output`

### Training with LoRA (Parameter-Efficient)

```bash
python train_grpo.py --use-lora
```

LoRA reduces VRAM usage by ~40% - recommended for memory constraints.

## GRPO Training Pipeline

### 4-Phase Loop (per iteration):

**1. Generate Completions**
- Sample batch of prompts from dataset
- Generate G=4 completions per prompt (configurable)
- Example: 4 prompts × 4 completions = 16 sequences

**2. Score Completions**
- Pass each completion through trained reward model
- Get reward score (float) for each completion
- Example: 16 completions → 16 reward scores

**3. Calculate Group-Relative Advantages**
- For each prompt group (G completions):
  - mean_reward = average of G rewards
  - std_reward = standard deviation of G rewards
  - advantage = (reward - mean) / std
- Normalizes rewards within prompt group (not globally)

**4. Policy Update**
- Calculate probability ratio: P_new / P_reference
- Clip ratio to [1-ε, 1+ε] to prevent large updates
- Compute loss: -min(clipped_ratio × advantage)
- Update model via μ=1 gradient step

Repeat for N epochs.

## Configuration

Edit `config.py` to customize:

### Batch & Generation
- `per_device_train_batch_size`: Prompts per batch (4 = 16 total with 4 completions)
- `num_generations`: G (completions per prompt, 4-8 recommended)
- `num_iterations`: μ (gradient updates per generation, 1-4)

### Generation Quality
- `temperature`: Sampling temperature (0.7 = diverse)
- `top_p`: Nucleus sampling (0.9 = focus on top 90%)
- `max_completion_length`: Max tokens per completion (256-512)

### Optimization
- `learning_rate`: Policy learning rate (5e-6 for stability)
- `num_train_epochs`: Training epochs (3-5)
- `epsilon`: Clipping range (0.2 = ±20% probability shift)

### Memory
- `gradient_checkpointing`: Trade compute for VRAM (enabled)
- `bf16`: Use bfloat16 precision (RTX 5090 supports this)

### LoRA (if --use-lora)
- `r`: LoRA rank (16 for 135M model)
- `lora_alpha`: LoRA scaling (32)
- `target_modules`: Which attention layers to adapt

## Training Statistics

**Estimated Training Time**:
- Per epoch: ~2-3 hours (depends on generation speed)
- Total (3 epochs): ~6-9 hours
- With LoRA: ~20% faster

**VRAM Usage**:
- Standard: ~28-30GB
- With LoRA: ~18-20GB

**Dataset**:
- Full ORCA DPO: ~80k samples
- Script uses: 10k samples (first iteration, adjust in config)
- Batch size: 4 prompts → 16 completions (4×4)
- Effective training steps: ~750 per epoch

## Outputs

After training, you'll have:

1. **Trained Model**: `./grpo_training_output`
   - Full model weights OR LoRA adapters
   - Tokenizer configuration
   - Config.json

2. **Training Logs**: `./grpo_training_output/logs`
   - TensorBoard event files
   - Scalars: loss, rewards/mean, rewards/std, learning_rate

3. **Checkpoints**: `./grpo_training_output/checkpoint-*`
   - Intermediate model snapshots
   - Keep last 2 by default

## Key Metrics to Monitor

- **loss**: Policy optimization loss (lower is better)
- **rewards/mean**: Average reward per batch (higher = model prefers better outputs)
- **rewards/std**: Reward variance (stable training = moderate std)
- **clip_ratio**: Fraction of probability ratios hitting bounds (5-10% is good)
- **learning_rate**: Current LR (should decrease with cosine scheduler)

## Troubleshooting

### CUDA Out of Memory
- Reduce `per_device_train_batch_size` (try 2 or 1)
- Reduce `num_generations` (try 2)
- Enable `--use-lora`
- Reduce `max_completion_length`

### Slow Generation
- Reward model inference is bottleneck
- Install vLLM: `pip install vllm` for faster inference
- Config already includes vLLM settings

### Bad Rewards (all same value)
- Check reward model is loaded correctly
- Verify reward model outputs are not clipped
- Sample some completions manually to verify scoring

## Next Steps

After training:

1. **Evaluate**: Generate samples and check quality
2. **Compare**: Evaluate on benchmark tasks
3. **Analyze**: Check if model learned preferences (vs generic improvement)
4. **Report**: Document preference adherence vs capability trade-off

## References

- [TRL GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- [SmolLM2 Model](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-SFT)
- [ORCA Dataset](https://huggingface.co/datasets/Intel/orca_dpo_pairs)
