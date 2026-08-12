# ATML-PA5 — LLM Decoding Strategies & Alignment Evaluation

This repository contains experiments and analysis for two related projects:
1) decoding strategy analysis for causal LLMs (beam, top-k, top-p, sampling, etc.), and
2) alignment evaluation experiments (reward-model based ranking, PPO/DPO/GRPO traces).

The work is organized as a sequence of task directories; each contains code, data scaffolding, and results for one subproject.

## Stack
- Language: Python 3.8+
- Main frameworks / libraries:
  - transformers (Hugging Face) — model loading & tokenization
  - torch (PyTorch) — model execution and evaluation
  - common data & utilities: numpy, pandas, matplotlib
- Notable patterns:
  - Each task has a main orchestration script (main.py)
  - Results and metrics are saved under results/ or metrics/ subfolders within tasks

## How it's organized

Top-level structure (annotated):

```
.gitignore
README.md
Task_1_LLM_Decoding_Strategy_Analysis/   Decoding strategy experiments & evaluation
  main.py                                Orchestration script (loads model, runs evaluations)
  config.py                              Hyperparameters and defaults (MODEL_NAME, DEVICE, TOP_K, etc.)
  requirements.txt                        Task-specific pip requirements
  IMPLEMENTATION_SUMMARY.md              Long implementation & methodology writeup
  data/                                   Dataset loading utilities
  decoding_strategies/                    Strategy implementations & helpers
  evaluation/                             Evaluators (DecodingStrategyEvaluator)
  utils/                                  plotting / results helpers
  metrics/                                output metrics
  results/                                evaluation outputs & plots
Task_2_LLM_ALIGNMENT/                    Alignment evaluation experiments & reward-model pipelines
  eval_alignment_pathologies.py           Alignment evaluation script (pathology checks)
  eval_run.log                            Example run log
  reward_model_on_orca_135M/              saved/structured artifacts (if provided)
  ppo/ grpo/ dpo/                         folders for different RL/optimization approaches
TASK_3_USAE/                              (other tasks / folders)
```

How it fits together:
- Task_1 runs decoding strategy comparisons for a chosen model (configurable via Task_1/config.py). It loads prompts, runs multiple decoding procedures, scores outputs, and produces comparative plots and JSON results.
- Task_2 contains alignment experiments that evaluate model behavior under reward models and RLHF-style training traces; it includes scripts to detect and log alignment pathologies.

## How to run

Suggested environment (example):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r Task_1_LLM_Decoding_Strategy_Analysis/requirements.txt
# plus: pip install torch transformers numpy pandas matplotlib
```

Run the decoding strategy analysis (Task 1):

```bash
# From repo root
python Task_1_LLM_Decoding_Strategy_Analysis/main.py --device cuda
# Or run on CPU:
python Task_1_LLM_Decoding_Strategy_Analysis/main.py --device cpu
```

Notes:
- Task_1/main.py reads MODEL_NAME, DEVICE, and decoding hyperparameters from Task_1_LLM_Decoding_Strategy_Analysis/config.py. Adjust that file or pass overrides as CLI args where supported.
- Task_1 includes a requirements.txt listing the main dependencies for that task — install it before running.
- Long-running evaluations (large models, many prompts) will require GPU and may use substantial memory. Consider using smaller models for quick smoke tests.

Run the alignment evaluation (Task 2):

```bash
python Task_2_LLM_ALIGNMENT/eval_alignment_pathologies.py
```

Task 2 may expect saved reward models or other artifacts; check the subfolders (reward_model_on_orca_135M/, reward_model_with_qwen6B/) and the script headers for usage details.

## Where results go
- Task_1: results/json_results/ and results/plots/
- Task_2: outputs and logs under Task_2_LLM_ALIGNMENT/ (eval_run.log, subfolders)
- Many tasks keep a metrics/ or results/ directory where evaluation JSON and plotted figures are stored.

## Reproducibility tips
- Use the config.py in each task to pin MODEL_NAME, RANDOM_SEED, DEVICE, and strategy hyperparameters.
- For heavy experiments, run a small sample first (Task_1 supports sampling and has sampling-generation helpers).
- Save intermediate JSON results and plots; re-running can be expensive for large model experiments.

## Contributing / Extending
- To add a new decoding strategy: add code under Task_1_LLM_Decoding_Strategy_Analysis/decoding_strategies/, add an entry in the evaluator / main orchestration, and update IMPLEMENTATION_SUMMARY.md with methodology.
- To add new reward-model experiments: add a new subfolder under Task_2_LLM_ALIGNMENT/ and a wrapper script that logs outputs to results/.

## Quick pointers (files to look at)
- Task_1_LLM_Decoding_Strategy_Analysis/main.py — main entrypoint for decoding comparisons
- Task_1_LLM_Decoding_Strategy_Analysis/config.py — defaults & hyperparameters
- Task_1_LLM_Decoding_Strategy_Analysis/IMPLEMENTATION_SUMMARY.md — long-form implementation notes and analysis
- Task_2_LLM_ALIGNMENT/eval_alignment_pathologies.py — alignment checks & logging


## Contact / Author
Repository owner: rabiaaslam92
