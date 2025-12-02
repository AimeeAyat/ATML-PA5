"""
Configuration and hyperparameters for LLM decoding strategies analysis.
"""

# Model Configuration
# OPTIONS (uncomment one - tested and working):
# MODEL_NAME = "gpt2"  # Small, fast, always available
# MODEL_NAME = "distilgpt2"  # Tiny version (even faster)
MODEL_NAME = "HuggingFaceTB/smollm2-135M-SFT-Only"  # ✓ RECOMMENDED: Instruction-tuned ~270MB
# MODEL_NAME = "microsoft/phi-1.5"  # Small instruction model (if available)
DEVICE = "cpu"  # CPU mode (CUDA sm_120 not fully supported yet)

# Decoding Hyperparameters
BEAM_WIDTH = 5
TOP_K = 50
TOP_P = 0.95
MAX_LENGTH = 200  # Reduced for faster inference
TEMPERATURE_VALUES = [0.2, 0.5, 0.8, 1.0, 1.2]  # Reduced set for faster evaluation

# Evaluation Configuration
NUM_PROMPTS = 5  # For across-prompt diversity (reduced for faster testing)
SAMPLES_PER_PROMPT = 5  # For within-prompt diversity (reduced for faster testing)
NUM_SAMPLES_PER_TEMP = 2  # Samples to generate per temperature value (reduced for faster testing)

# Reward Model Configuration
# Using OpenAssistant DeBERTa v3 Large - proven quality reward model
# First time download will be ~1.2GB, then cached locally
REWARD_MODEL_NAME = "OpenAssistant/reward-model-deberta-v3-large"

# Dataset Configuration
INSTRUCTION_DATASET = "tatsu-lab/alpaca"  # Using Alpaca dataset as reference
DATASET_SPLIT = "train"
DATASET_SIZE = 50  # Reduced dataset size for faster loading

# Paths
RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"
JSON_RESULTS_DIR = "results/json_results"
MODELS_CHECKPOINTS_DIR = "results/model_checkpoints"

# Metrics
DISTINCT_N_VALUES = [1, 2, 3]  # For Distinct-1, Distinct-2, Distinct-3

# Random Seed
RANDOM_SEED = 42
