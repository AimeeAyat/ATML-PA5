# PPO Training: SmolLM2-135M with Trained Reward Model

Fine-tune SmolLM2-135M-SFT LLM using **Proximal Policy Optimization (PPO)** with the trained reward model as the preference signal.

## Overview

**PPO** is a reinforcement learning algorithm that trains LLMs to generate better completions while maintaining training stability through a clipping mechanism that prevents drastic policy changes.

**Setup:**
- **LLM**: SmolLM2-135M-SFT-Only (the model to fine-tune)
- **Dataset**: Intel/orca_dpo_pairs (for prompts)
- **Reward Model**: SmolLM2-135M trained on ORCA DPO
- **Hardware**: RTX 5090 (32GB VRAM) - Windows 11

## Project Structure

```
ppo_training/
├── requirements.txt          # Project dependencies
├── config.py                 # PPO configuration
├── train_ppo.py             # Main PPO training script
├── evaluate_generations.py  # Evaluation and scoring script
├── plot_training_progress.py # Training visualization
└── README.md                 # This file
```

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Verify reward model path**: Make sure the trained reward model exists at:
```
../reward_model_on_orca_dataset/reward_model_orca_output/checkpoint-2160
```

## Usage

### Basic Training

```bash
python train_ppo.py
```

This will:
1. Load SmolLM2-135M-SFT-Only LLM
2. Load trained reward model from ORCA training
3. Load ORCA DPO dataset (first 10k samples for faster iteration)
4. Fine-tune LLM using PPO for 3 epochs
5. Save trained model to `./ppo_training_output`

### Training with LoRA (Parameter-Efficient)

```bash
python train_ppo.py --use-lora
```

LoRA reduces VRAM usage by ~40% - recommended for memory constraints.

## PPO Training Pipeline

### PPO Difference from GRPO

**PPO uses 4 components:**
1. Policy Model - The LLM being trained
2. Reference Model - Frozen copy to prevent large policy shifts
3. Reward Model - Scores completions
4. Value Model - Predicts expected rewards (implicit in our setup)

**Key Stability Feature: Probability Ratio Clipping**
- Clips probability ratio P_new / P_reference to [1-epsilon, 1+epsilon]
- Prevents any single update from causing catastrophic forgetting
- Example: If probability would increase 3x, clip to 1.2x (20% increase)

### 5-Phase PPO Training Loop

**1. Generate Completions**
- Sample prompts from dataset
- Generate multiple completions using policy model

**2. Score with Reward Model**
- Pass completions through reward model
- Get reward scores

**3. Calculate KL Divergence**
- Compare policy vs reference model probability
- Use as stability penalty to keep updates bounded

**4. Calculate Advantages**
- Using reward model scores + value predictions
- Advantage = (reward + discounted future) - baseline

**5. Update Policy (with Clipping)**
- Calculate probability ratio: P_new / P_reference
- **Clip to [0.8, 1.2]** (epsilon=0.2)
- Compute loss and update model
- **Do NOT update reference model** (stays frozen)

Repeat for multiple epochs.

## Configuration

Edit `config.py` to customize:

### Batch & Generation
- `per_device_train_batch_size`: Prompts per batch (4 = fits RTX 5090)
- `mini_batch_size`: Gradient accumulation mini-batches
- `gradient_accumulation_steps`: How many steps to accumulate

### PPO-Specific
- `ppo_epochs`: Number of PPO update epochs per cycle (4)
- `init_kl_coef`: Initial KL penalty coefficient (0.05)
- `target`: Target KL divergence value (6.0)
- `gamma`: Discount factor (1.0)
- `lam`: GAE lambda for advantage estimation (0.95)

### Generation Quality
- `temperature`: Sampling temperature (0.7 = diverse)
- `top_p`: Nucleus sampling (0.9 = focus on top 90%)
- `max_new_tokens`: Max tokens per completion

### Memory
- `gradient_checkpointing`: Trade compute for VRAM (enabled)
- `bf16`: Use bfloat16 precision (disabled for compatibility)

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
- Standard: ~28-30GB (need both policy + reference model)
- With LoRA: ~18-20GB

**Dataset**:
- Full ORCA DPO: ~80k samples
- Script uses: 10k samples (adjust in config)
- Batch size: 4 prompts per batch
- Effective training steps: ~750 per epoch

## Outputs

After training, you'll have:

1. **Trained Model**: `./ppo_training_output`
   - Full model weights OR LoRA adapters
   - Tokenizer configuration
   - Config.json

2. **Training Logs**: `./ppo_training_output/logs`
   - TensorBoard event files
   - Scalars: loss, rewards/mean, kl_divergence, learning_rate

3. **Checkpoints**: `./ppo_training_output/checkpoint-*`
   - Intermediate model snapshots
   - Keep last 1 by default (due to save_total_limit=1)

## Key Metrics to Monitor

- **loss**: Policy optimization loss (lower is better)
- **learning_rate**: Current LR (should decrease with cosine scheduler)
- **rewards/mean**: Average reward per batch (higher = better completions)
- **kl_divergence**: KL penalty from reference policy (should stay near target)
- **advantage**: Difference between actual vs predicted value (balance is key)

## Evaluation

After training completes:

### Generate and Score Completions

```bash
python evaluate_generations.py --model-path ./ppo_training_output
```

This generates 20 random prompts with 4 completions each, scores them, and saves results to JSON showing:
- Generated completions
- Reward scores for each
- Rankings (BEST/MEDIUM/WORST)
- Statistical summaries

### Visualize Training Progress

```bash
python plot_training_progress.py
```

Creates 4-panel plot showing:
- Training loss over time
- Learning rate schedule
- Mean reward per batch
- KL divergence from reference policy

Exports metrics to CSV for further analysis.

## Troubleshooting

### CUDA Out of Memory
- Reduce `per_device_train_batch_size` (try 2 or 1)
- Reduce `mini_batch_size`
- Reduce `gradient_accumulation_steps`
- Enable `--use-lora`
- Reduce `max_new_tokens`

### KL Divergence Too High
- Increase `init_kl_coef` (higher penalty on divergence)
- Reduce learning rate
- Reduce `ppo_epochs` per update cycle

### Training Unstable
- Reduce learning rate (current: 5e-6)
- Increase `gradient_accumulation_steps`
- Disable gradient checkpointing if not needed

### Slow Generation
- Same as GRPO - generation is CPU-bound bottleneck
- Reduce batch size to let GPU work on reward scoring

## PPO vs GRPO

| Aspect | PPO | GRPO |
|--------|-----|------|
| Reference Model | Yes (frozen) | No |
| KL Divergence | Explicit penalty | Implicit via beta |
| Value Model | Yes (implicit) | No |
| Stability | Very stable | Efficient |
| Memory | Higher (2 models) | Lower (1 model) |
| Computation | Moderate | Lower |

Both achieve similar results; PPO is more stable, GRPO is more efficient.

## Next Steps

After training:

1. **Compare Results**: Evaluate both GRPO and PPO trained models
2. **Analyze Metrics**: Compare training trajectories
3. **Assess Quality**: Check generated completions from both
4. **Write Report**: Document preference adherence vs capability trade-off

## References

- [TRL PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer)
- [SmolLM2 Model](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-SFT)
- [ORCA Dataset](https://huggingface.co/datasets/Intel/orca_dpo_pairs)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
