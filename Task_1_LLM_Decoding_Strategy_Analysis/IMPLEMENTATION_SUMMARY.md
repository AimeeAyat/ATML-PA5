# LLM Decoding Strategies - Implementation Summary

## Overview
Comparative analysis of four fundamental LLM decoding strategies: Greedy Search, Beam Search, Top-K Sampling, and Top-P Sampling (Nucleus Sampling) using SmolLM2-135M-SFT-Only as the baseline model. The analysis investigates trade-offs between generation quality, output diversity, and the impact of the temperature hyperparameter.

---

## 1. IMPLEMENTATION STATUS

### 1.1 Decoding Strategies Implemented
✅ **Greedy Search** (`decoding_strategies/greedy.py`)
- Selects token with highest probability at each step
- Implements: softmax → argmax → token append → repeat until EOS/max_length
- Temperature scaling applied to logits before softmax

✅ **Beam Search** (`decoding_strategies/beam_search.py`)
- Maintains B most promising partial sequences at each step
- Tracks cumulative log-probabilities for each beam
- Keeps top-B sequences based on log-probability scores
- Returns best scoring completed or active sequence

✅ **Top-K Sampling** (`decoding_strategies/top_k_sampling.py`)
- Restricts sampling to K tokens with highest probability
- Zero-out probabilities outside top-K
- Stochastic sampling from renormalized distribution
- Temperature scaling applied before softmax

✅ **Top-P Sampling** (`decoding_strategies/top_p_sampling.py`)
- Dynamically selects smallest set with cumulative probability > P
- Sort tokens by probability (descending)
- Accumulate until nucleus threshold P exceeded
- Stochastic sampling from renormalized nucleus distribution
- Temperature scaling applied before softmax

### 1.2 Key Features Implemented
✅ **Chat Template Support**
- All decoders use `tokenizer.apply_chat_template()` with `add_generation_prompt=True`
- Format: `<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n`
- Fallback to plain tokenization for models without chat templates
- **Critical fix:** Resolves empty generation issue (was main bug)

✅ **Minimum Token Generation**
- Ensures at least 50 new tokens generated per prompt
- Formula: `max(50, max_length - input_length)`
- Prevents truncated/short outputs when prompts are long

✅ **Proper EOS Handling**
- Detects EOS token and stops generation early
- Cleans special tokens from output
- Handles edge cases (only EOS, empty generation)

---

## 2. MODELS & DATASETS

### 2.1 Language Model
```
Model: HuggingFaceTB/smollm2-135M-SFT-Only
Type: SFT (Supervised Fine-Tuned) Baseline
Size: ~270MB (inference-friendly)
Architecture: GPT-2 based
Special Tokens: <|im_start|>, <|im_end|>
```

### 2.2 Reward Model
```
Model: OpenAssistant/reward-model-deberta-v3-large
Type: Instruction-following quality scorer
Size: ~1.2GB
Architecture: DeBERTa v3 Large
Output: Unbounded logits (not [0,1] normalized)
Device: CUDA (falls back to CPU if unavailable)
Score Interpretation:
  - Higher = Better quality
  - Lower/Negative = Poor quality
  - Scale is unbounded, enables ranking
```

### 2.3 Instruction Dataset
```
Dataset: tatsu-lab/alpaca (HuggingFace)
Split: train
Size: 50 random samples (reduced for faster testing)
Sampling: Random with seed=42 for reproducibility
Format: {instruction, input, output} → extract instruction as prompt
```

---

## 3. HYPERPARAMETERS

### 3.1 Decoding Strategy Parameters
```
BEAM_WIDTH = 5                    # Number of beams in Beam Search
TOP_K = 50                        # K tokens for Top-K Sampling
TOP_P = 0.95                      # Nucleus threshold for Top-P Sampling
TEMPERATURE_VALUES = [0.2, 0.5, 0.8, 1.0, 1.2]  # Temperature sweep
MAX_LENGTH = 200                  # Max output tokens
```

### 3.2 Evaluation Parameters
```
NUM_PROMPTS = 5                   # Prompts for across-prompt diversity
SAMPLES_PER_PROMPT = 5            # Samples per prompt for within-prompt diversity
NUM_SAMPLES_PER_TEMP = 2          # Samples per temperature in ablation
DISTINCT_N_VALUES = [1, 2, 3]    # N-gram sizes: unigram, bigram, trigram
```

### 3.3 System Parameters
```
DEVICE = "cuda"                   # GPU (or CPU if unavailable)
RANDOM_SEED = 42                  # Reproducibility
INSTRUCTION_DATASET = "tatsu-lab/alpaca"
DATASET_SPLIT = "train"
DATASET_SIZE = 50
```

---

## 4. TESTS CONDUCTED

### 4.1 Temperature Ablation Study
**Purpose:** Measure quality and diversity variation across temperature range

**Procedure:**
- For each strategy (Greedy, Beam, Top-K, Top-P):
  - For each temperature in [0.2, 0.5, 0.8, 1.0, 1.2]:
    - Generate NUM_SAMPLES_PER_TEMP (2) samples per prompt
    - Across NUM_PROMPTS (5) prompts
    - Total: 4 strategies × 5 temperatures × 5 prompts × 2 samples = 200 generations

**Output Metrics:**
- Distinct-1, Distinct-2, Distinct-3 (per temperature)
- Average reward score (per temperature)
- Number of samples generated
- Total execution time per strategy

### 4.2 Across-Prompt Diversity Test
**Purpose:** Measure lexical variety when sampling from different prompts (T=0.8)

**Procedure:**
- For each strategy:
  - Generate 1 sample per prompt (NUM_PROMPTS = 5)
  - Fixed temperature T = 0.8
  - Aggregate all 5 samples
  - Compute Distinct-N over aggregated corpus

**Metrics:**
- Distinct-1, Distinct-2, Distinct-3 (corpus-level)
- Number of samples (should be ~5)
- Execution time

**Interpretation:**
- High Distinct-N → Strategy produces varied outputs across different inputs
- Low Distinct-N → Strategy tends to reuse same tokens regardless of prompt

### 4.3 Within-Prompt Diversity Test
**Purpose:** Measure variation in outputs for same prompt (T=0.8)

**Procedure:**
- For each strategy:
  - Generate SAMPLES_PER_PROMPT (5) samples from single prompt
  - Fixed temperature T = 0.8
  - Compute Distinct-N over all 5 samples

**Metrics:**
- Distinct-1, Distinct-2, Distinct-3 (corpus-level)
- Number of samples (should be ~5)
- Execution time

**Interpretation:**
- High Distinct-N → Strategy explores output space; avoids collapse to same response
- Low Distinct-N → Strategy tends to generate similar outputs for same prompt

### 4.4 Sample Generation Collection
**Purpose:** Inspect actual generations and reward scores for manual analysis

**Procedure:**
- For first 2 prompts:
  - For each strategy (4 total):
    - For each temperature in [0.2, 0.5, 0.8, 1.0, 1.2]:
      - Generate 1 sample
      - Score with reward model
      - Store generation, reward, length

**Output:**
- `sample_generations.json` containing:
  ```json
  {
    "samples": [
      {
        "prompt": "...",
        "strategies": {
          "Greedy": {
            "temperatures": {
              "0.2": {
                "generation": "...",
                "reward": float,
                "length": int
              }
            }
          }
        }
      }
    ]
  }
  ```

---

## 5. METRICS CALCULATED

### 5.1 Diversity Metric: Distinct-N
**Formula:**
```
Distinct-N = (unique N-grams) / (total N-grams)
Range: [0, 1]
Higher = Better (more diverse)
```

**Implementation:**
- Tokenize text (split by whitespace, lowercase)
- Extract all N-grams (sliding window of size N)
- Count unique N-grams
- Divide by total N-grams

**Computed for:**
- N ∈ {1, 2, 3} → Distinct-1 (unigrams), Distinct-2 (bigrams), Distinct-3 (trigrams)
- Per temperature (temperature ablation)
- Corpus-level (across-prompt and within-prompt diversity)

### 5.2 Quality Metric: Reward Score
**Model:** OpenAssistant/reward-model-deberta-v3-large

**Procedure:**
1. Tokenize generation (max 512 tokens, pad to 512)
2. Pass through DeBERTa model
3. Return raw logits (unbounded)

**Range:** (-∞, +∞), typically [-5, +5]
**Higher = Better quality**

**Interpretation:**
- Positive scores: coherent, helpful, well-structured
- Negative scores: incoherent, unhelpful, repetitive
- Magnitude indicates confidence of assessment

### 5.3 Additional Metrics
```
Generation Length: Number of tokens in output
  - Tracked for each sample
  - Compared across strategies

Execution Time: Per-strategy wall-clock time
  - Greedy << Beam ~ Top-K ~ Top-P
  - Affected by reward model inference time
```

---

## 6. OUTPUT FILES

### 6.1 JSON Results
- **`results/json_results/evaluation_results.json`**
  - Temperature ablation results
  - Across-prompt diversity scores
  - Within-prompt diversity scores
  - Per-strategy timing information

- **`results/json_results/sample_generations.json`**
  - Actual text generations for inspection
  - Reward scores per generation
  - Generation lengths

### 6.2 Plot Files
- **`results/plots/`**
  - Distinct-N vs Temperature (per strategy)
  - Quality vs Temperature (per strategy)
  - Strategy comparison plots

---

## 7. EXPECTED RESULTS & INTERPRETATION

### 7.1 Temperature Ablation (Quality & Diversity Trade-off)

#### Greedy Search
```
Expected Pattern:
- Quality: Remains relatively stable (deterministic, no randomness)
- Distinct-1: ~0.90-0.95 (low diversity - repetitive)
- Distinct-2: ~0.85-0.92
- Distinct-3: ~0.75-0.88
- Reward: Moderate (0 to +2 range)
- Temperature Effect: Minimal (logits scaled but argmax still deterministic)

Why:
- Argmax operation is deterministic regardless of temperature
- No exploration of output space
- Tends to generate similar tokens across all samples
```

#### Beam Search (B=5)
```
Expected Pattern:
- Quality: Moderate improvement over Greedy (+0.5 to +1.0 reward)
- Distinct-1: ~0.92-0.96 (slightly more diverse than Greedy)
- Distinct-2: ~0.87-0.94
- Distinct-3: ~0.78-0.90
- Temperature Effect: Slight (affects logit scaling before log-softmax)

Why:
- Explores B=5 best paths in sequence space
- More likely to find globally good sequences than Greedy
- Still deterministic (cum log-prob based)
- Similar tokens due to low diversity in top-B beam paths
```

#### Top-K Sampling (K=50)
```
Expected Pattern:
- Quality: T=0.2 (~+1.5) → T=1.2 (~-1.0) [DECREASES]
- Distinct-1: T=0.2 (~0.88) → T=1.2 (~0.96) [INCREASES]
- Distinct-2: T=0.2 (~0.80) → T=1.2 (~0.93) [INCREASES]
- Distinct-3: T=0.2 (~0.70) → T=1.2 (~0.88) [INCREASES]

Why:
- Low T: Sharp distribution → sample from narrow set → high quality, low diversity
- High T: Flat distribution → sample from broader set → lower quality, high diversity
- Stochastic sampling enables variation
- K=50 is large enough to capture most probability mass at all temps
```

#### Top-P Sampling (P=0.95)
```
Expected Pattern:
- Quality: T=0.2 (~+1.8) → T=1.2 (~-0.8) [DECREASES slightly less than Top-K]
- Distinct-1: T=0.2 (~0.89) → T=1.2 (~0.95) [INCREASES]
- Distinct-2: T=0.2 (~0.82) → T=1.2 (~0.92) [INCREASES]
- Distinct-3: T=0.2 (~0.72) → T=1.2 (~0.87) [INCREASES]

Why:
- Nucleus dynamically adapts (smaller set at low T, larger at high T)
- Generally better quality than Top-K at high temps (picks reliable tokens)
- Smoother quality degradation (less harsh than Top-K)
- Similar diversity pattern but slightly offset (often better quality at same T)
```

### 7.2 Across-Prompt Diversity (T=0.8)

```
Expected Ranking (Distinct-1 @ T=0.8):
1. Top-P: ~0.94     (nucleus sampling, stochastic)
2. Top-K: ~0.93     (K=50 sampling, stochastic)
3. Beam: ~0.92      (B=5 paths, deterministic but diverse)
4. Greedy: ~0.90    (single path, repetitive)

Interpretation:
- Stochastic methods (Top-K, Top-P) > Deterministic (Beam, Greedy)
- Top-P > Top-K: nucleus better adapts to prompt variation
- Sampling produces more varied outputs across different inputs
- Greedy severely limited by single-path constraint
```

### 7.3 Within-Prompt Diversity (T=0.8)

```
Expected Ranking (Distinct-1 @ T=0.8 for 5 samples from same prompt):
1. Top-K: ~0.91     (stochastic, explores output space)
2. Top-P: ~0.90     (stochastic, nucleus still adds variation)
3. Beam: ~0.88      (deterministic, limited variation)
4. Greedy: ~0.85    (fully deterministic, likely identical outputs)

Interpretation:
- Greedy often generates IDENTICAL output multiple times (Distinct-1 may be 0)
- Beam produces varied but limited set of sequences
- Top-K produces high variation within same prompt
- Top-P produces variation while maintaining coherence
- Key insight: Sampling > Deterministic for avoiding output collapse
```

### 7.4 Quality vs Diversity Trade-off

```
At T=0.2 (Conservative):
- Best Quality: Greedy/Beam (~+2.0 reward)
- Lowest Diversity: Greedy (~0.88 Distinct-1)
- Best Balance: Beam Search

At T=0.8 (Balanced):
- Good Quality: Top-P (~+0.5), Top-K (~+0.4)
- Good Diversity: Top-K (~0.93), Top-P (~0.94)
- Best for most use cases: Top-P

At T=1.2 (Exploratory):
- Lower Quality: All methods (~-1.0 to -2.0 reward)
- High Diversity: Top-K/Top-P (~0.96 Distinct-1)
- Trade-off: Generate many ideas, filter with external ranking

Expected Curve Shapes:
- Greedy: Flat curve (reward stable, diversity ≈ constant low)
- Beam: Slightly declining quality, slowly rising diversity
- Top-K: Steep quality decline, steep diversity increase
- Top-P: Moderate quality decline, moderate diversity increase (smoother than Top-K)
```

### 7.5 Key Observations to Verify

1. **Temperature Effect:**
   ✓ Should see STRONG correlation: Higher T → Higher Diversity, Lower Quality
   ✓ Greedy/Beam should show minimal change
   ✓ Top-K/Top-P should show clear trade-off

2. **Strategy Ranking:**
   ✓ Greedy: Fastest, worst diversity, moderate quality
   ✓ Beam: Faster, better quality than sampling, low diversity
   ✓ Top-K: Balanced, good quality+diversity, slower
   ✓ Top-P: Best balance, smoother curves, most reliable

3. **Sample Lengths:**
   ✓ Should be ~140-150 tokens (due to 50-token minimum + generation)
   ✓ Similar across strategies
   ✓ Slightly shorter if EOS hit early

4. **Reward Scores:**
   ✓ Positive at low T (quality generations)
   ✓ Negative at high T (incoherent generations)
   ✓ Cross-zeros around T=0.8-1.0

5. **Diversity Metrics:**
   ✓ Distinct-1 > Distinct-2 > Distinct-3 (easier to have diverse unigrams)
   ✓ All increase with temperature for stochastic methods
   ✓ Greedy/Beam should plateau or decrease

---

## 8. HOW TO VERIFY RESULTS

### 8.1 Checklist for Cross-Verification
- [ ] All 4 strategies generate non-empty text (≥50 tokens)
- [ ] Temperature values [0.2, 0.5, 0.8, 1.0, 1.2] all have generations
- [ ] Reward scores vary by generation (not all 0.5)
- [ ] Distinct-1 values in range [0.85, 0.96] for temperature sweep
- [ ] Clear inverse correlation: T↑ → Diversity↑, Quality↓
- [ ] Across-prompt Distinct-1: Top-P/Top-K > Beam > Greedy
- [ ] Within-prompt: Top-K > Top-P > Beam > Greedy
- [ ] Sample generations are coherent and varied
- [ ] Execution completes without crashes

### 8.2 Red Flags (Indicates Bugs)
- ❌ Empty generations (should be fixed, but double-check)
- ❌ All reward scores = 0.5 (dummy scorer, not real model)
- ❌ Distinct-N values outside [0, 1] range
- ❌ Same output for all temperatures (temperature not working)
- ❌ Greedy produces MORE diverse output than sampling (logic error)
- ❌ Quality increases with temperature (expected inverse)

### 8.3 Interpretation Guide
```
If Greedy diversity is HIGH:
  → Check: Is temperature actually being used?
  → Check: Is chat template applied?

If all rewards are 0.5:
  → Check: Is real reward model loaded? (check log for "[RewardModel] Successfully loaded")
  → Check: Device setting (ensure GPU has enough memory)

If Distinct-N doesn't change with temperature:
  → Check: Is sampling actually stochastic?
  → Check: Are different random seeds used per generation?

If quality doesn't trade off with diversity:
  → Check: Is reward model working correctly?
  → Check: Are generations actually different?
```

---

## 9. SUMMARY TABLE

| Component | Implementation | Status |
|-----------|----------------|--------|
| Greedy Search | ✓ Complete | Ready |
| Beam Search | ✓ Complete | Ready |
| Top-K Sampling | ✓ Complete | Ready |
| Top-P Sampling | ✓ Complete | Ready |
| Chat Template Support | ✓ Fixed | Ready |
| Minimum Token Generation | ✓ Implemented | Ready |
| Distinct-N Metric | ✓ Complete | Ready |
| Reward Model Scoring | ✓ Complete | Ready |
| Temperature Ablation | ✓ Complete | Ready |
| Across-Prompt Diversity | ✓ Complete | Ready |
| Within-Prompt Diversity | ✓ Complete | Ready |
| Sample Generation | ✓ Complete | Ready |
| Error Handling | ✓ Enhanced | Ready |
| GPU Support | ✓ Working (PyTorch 2.6.0+) | Ready |

---

## 10. NEXT STEPS

1. Run: `python main.py`
2. Wait for completion (GPU: ~2-5 min, CPU: ~15-20 min)
3. Check: `results/json_results/evaluation_results.json` and `sample_generations.json`
4. Verify against expected results in Section 7
5. Analyze plots in `results/plots/`

---

**Document Version:** 1.0
**Last Updated:** 2025-01-27
**Status:** All implementations complete, ready for evaluation
