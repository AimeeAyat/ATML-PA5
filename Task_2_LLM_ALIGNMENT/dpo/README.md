# DPO Training: SmolLM2-135M with ORCA Dataset

Fine-tune SmolLM2-135M-SFT LLM using **Direct Preference Optimization (DPO)** with the ORCA DPO Pairs dataset.

## Overview

**DPO** is a simpler, faster, and more stable alternative to PPO/GRPO. Instead of training a separate reward model, DPO directly learns from preference pairs (chosen vs rejected responses) using a single stage of training.

**Key Advantage**: No reward model training needed! DPO works directly with preference data.

**Setup:**
- **LLM**: SmolLM2-135M-SFT-Only (the model to fine-tune)
- **Dataset**: Intel/orca_dpo_pairs (has chosen/rejected response pairs)
- **Hardware**: RTX 5090 (32GB VRAM) - Windows 11
- **Training Time**: ~2-3 hours per epoch (faster than PPO/GRPO due to no generation)

## Project Structure

```
dpo_training/
├── requirements.txt          # Project dependencies
├── config.py                 # DPO configuration
├── train_dpo.py             # Main DPO training script
└── README.md                 # This file
```

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
python train_dpo.py
```

This will:
1. Load SmolLM2-135M-SFT-Only LLM
2. Load ORCA DPO dataset (first 10k samples for faster iteration)
3. Fine-tune LLM using DPO for 3 epochs
4. Save trained model to `./dpo_training_output`

### Training with LoRA (Parameter-Efficient)

```bash
python train_dpo.py --use-lora
```

LoRA reduces VRAM usage by ~40% - recommended for memory constraints.

## DPO Training Algorithm

### What is DPO?

DPO (Direct Preference Optimization) optimizes a language model to align with human preferences through a single loss function. Instead of the complex two-stage RLHF pipeline (reward model → RL), DPO directly uses preference pairs.

### Loss Function

```
DPO Loss = -log(sigmoid(β * (log_π(chosen) - log_π_ref(chosen) - log_π(rejected) + log_π_ref(rejected))))
```

Where:
- **β** = Temperature controlling preference strength (typically 0.5)
- **π** = Policy model (being trained)
- **π_ref** = Reference model (frozen, prevents divergence)
- **chosen** = Preferred response
- **rejected** = Non-preferred response

### Key Insight

DPO extracts an implicit reward function from preference data without explicitly training one. The model learns the optimal policy in a single stage.

## Configuration

Edit `config.py` to customize:

### Batch & Generation
- `per_device_train_batch_size`: Samples per batch (8 = can be larger than PPO/GRPO, no generation)
- `per_device_eval_batch_size`: Evaluation batch size

### DPO-Specific
- `beta`: Temperature for preference strength (0.5 is standard)
  - Lower beta = enforce preferences more strictly
  - Higher beta = allow more diversity
- `max_prompt_length`: Max prompt tokens (512)
- `max_length`: Max total tokens (1024)

### Learning & Optimization
- `learning_rate`: Currently 5e-6 (conservative)
- `lr_scheduler_type`: "cosine" (smooth decay)
- `num_train_epochs`: 3

### Memory
- `gradient_checkpointing`: Trade compute for VRAM (enabled)
- `bf16`: Use bfloat16 precision (disabled for compatibility)

### LoRA (if --use-lora)
- `r`: LoRA rank (16 for 135M model)
- `lora_alpha`: LoRA scaling (32)
- `target_modules`: Which attention layers to adapt

## Training Statistics

**Estimated Training Time:**
- Per epoch: ~40-60 minutes (much faster than PPO/GRPO!)
- Total (3 epochs): ~2-3 hours

**VRAM Usage:**
- Standard: ~24-26GB (policy + reference model)
- With LoRA: ~16-18GB

**Dataset:**
- Full ORCA DPO: ~80k samples with chosen/rejected pairs
- Script uses: 10k samples (adjust in train_dpo.py line 89)
- Batch size: 8 samples per batch
- Effective training steps: ~1250 per epoch

## Outputs

After training, you'll have:

1. **Trained Model**: `./dpo_training_output`
   - Full model weights OR LoRA adapters
   - Tokenizer configuration
   - Config.json

2. **Training Logs**: `./dpo_training_output/logs`
   - TensorBoard event files
   - Scalars: loss, learning_rate

3. **No Checkpoints**: Only the final model is saved (as requested)
   - `save_strategy="no"` prevents intermediate checkpoints
   - Saves final model after training completes

## Key Metrics to Monitor

- **loss**: DPO training loss (lower is better)
- **learning_rate**: Current LR (should decrease with cosine scheduler)

## DPO vs PPO vs GRPO Comparison

| Aspect | DPO | PPO | GRPO |
|--------|-----|-----|------|
| **Reward Model** | ❌ None | ✅ Required | ✅ Required |
| **RL Component** | ❌ No | ✅ Yes | ✅ Yes |
| **Stages** | 1 | 2 | 2 |
| **Training Time** | ⚡ ~2-3 hours | ~8-10 hours | ~6-8 hours |
| **VRAM** | ~24-26GB | ~28-30GB | ~25-28GB |
| **Stability** | Very stable | Good | Good |
| **Code Complexity** | Simple | Complex | Medium |
| **Data Format** | Chosen/Rejected pairs | Prompts only | Prompts only |

**DPO is optimal for:**
- Preference-aligned datasets (like ORCA DPO)
- Faster training iterations
- Resource-constrained scenarios
- Maximum stability

## Why DPO for ORCA Dataset?

The ORCA dataset is **perfect** for DPO because:

1. **Already has preference pairs**: Each sample includes both `chosen` and `rejected` responses
2. **No reward model needed**: Skip one entire training stage
3. **Faster training**: No generation sampling overhead (PPO/GRPO must generate completions)
4. **Better stability**: Direct classification loss is inherently stable
5. **Memory efficient**: Only 2 models (policy + reference), not 3+ like PPO

## Troubleshooting

### CUDA Out of Memory
- Reduce `per_device_train_batch_size` (try 4 or 2)
- Enable `--use-lora`
- Reduce `max_length`

### Training Unstable
- Reduce learning rate (current: 5e-6)
- Increase `gradient_accumulation_steps`
- Reduce `beta` (e.g., 0.3 instead of 0.5)

### Slow Training
- Increase `per_device_train_batch_size` (DPO doesn't need generation, so larger batches are OK)
- Reduce number of samples in dataset
- Use `--use-lora` for faster gradient updates

## Next Steps

After training:

1. **Compare Results**: Evaluate DPO model alongside GRPO and PPO
2. **Analyze Metrics**: Compare training trajectories across all three methods
3. **Assess Quality**: Check generated completions from all approaches
4. **Write Report**: Document preference adherence vs capability trade-off for all three methods

## References

- [TRL DPOTrainer](https://huggingface.co/docs/trl/main/en/dpo_trainer)
- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [SmolLM2 Model](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-SFT)
- [ORCA Dataset](https://huggingface.co/datasets/Intel/orca_dpo_pairs)
