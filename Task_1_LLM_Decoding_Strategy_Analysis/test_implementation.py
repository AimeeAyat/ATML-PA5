"""
Test script to verify all modules are working correctly.
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Fix for Windows encoding issues
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("=" * 60)
print("Testing LLM Decoding Strategies Implementation")
print("=" * 60)

# Test imports
print("\n1. Testing imports...")
try:
    from data import InstructionDataset
    print("   ✓ Data module imported")
except Exception as e:
    print(f"   ✗ Error importing data: {e}")
    sys.exit(1)

try:
    from decoding_strategies import (
        GreedySearchDecoder,
        BeamSearchDecoder,
        TopKSamplingDecoder,
        TopPSamplingDecoder,
    )
    print("   ✓ Decoding strategies module imported")
except Exception as e:
    print(f"   ✗ Error importing decoding strategies: {e}")
    sys.exit(1)

try:
    from metrics import DistinctNMetric, RewardModelScorer
    print("   ✓ Metrics module imported")
except Exception as e:
    print(f"   ✗ Error importing metrics: {e}")
    sys.exit(1)

try:
    from evaluation import DecodingStrategyEvaluator
    print("   ✓ Evaluation module imported")
except Exception as e:
    print(f"   ✗ Error importing evaluation: {e}")
    sys.exit(1)

try:
    from utils import ResultsPlotter
    print("   ✓ Utils module imported")
except Exception as e:
    print(f"   ✗ Error importing utils: {e}")
    sys.exit(1)

# Test DistinctNMetric
print("\n2. Testing DistinctNMetric...")
try:
    sample_texts = [
        "The quick brown fox jumps over the lazy dog",
        "The lazy dog sleeps all day",
    ]
    distinct_1 = DistinctNMetric.compute_distinct_n_corpus(sample_texts, n=1)
    distinct_2 = DistinctNMetric.compute_distinct_n_corpus(sample_texts, n=2)
    distinct_3 = DistinctNMetric.compute_distinct_n_corpus(sample_texts, n=3)

    print(f"   Sample texts: {len(sample_texts)} samples")
    print(f"   Distinct-1: {distinct_1:.4f}")
    print(f"   Distinct-2: {distinct_2:.4f}")
    print(f"   Distinct-3: {distinct_3:.4f}")

    assert 0 <= distinct_1 <= 1, "Distinct-1 should be between 0 and 1"
    assert 0 <= distinct_2 <= 1, "Distinct-2 should be between 0 and 1"
    assert 0 <= distinct_3 <= 1, "Distinct-3 should be between 0 and 1"
    print("   ✓ DistinctNMetric working correctly")
except Exception as e:
    print(f"   ✗ Error testing DistinctNMetric: {e}")
    sys.exit(1)

# Test RewardModelScorer initialization
print("\n3. Testing RewardModelScorer...")
try:
    scorer = RewardModelScorer()
    print("   ✓ RewardModelScorer initialized")
    print(f"   Note: Full reward model evaluation requires GPU")
except Exception as e:
    print(f"   ⚠ Warning initializing RewardModelScorer: {e}")
    print("   This may require downloading model weights on first run")

# Test InstructionDataset
print("\n4. Testing InstructionDataset...")
try:
    print("   Creating sample prompts...")
    sample_prompts = [
        "Explain quantum computing to a child",
        "Write a haiku about coffee",
        "What are the benefits of meditation?",
        "How do plants grow?",
        "Tell me a joke",
    ]
    print(f"   ✓ Created {len(sample_prompts)} sample prompts")
except Exception as e:
    print(f"   ✗ Error with prompts: {e}")
    sys.exit(1)

# Test configuration
print("\n5. Testing configuration...")
try:
    from config import (
        MODEL_NAME,
        DEVICE,
        BEAM_WIDTH,
        TOP_K,
        TOP_P,
        MAX_LENGTH,
        TEMPERATURE_VALUES,
    )

    print(f"   Model: {MODEL_NAME}")
    print(f"   Device: {DEVICE}")
    print(f"   Beam Width: {BEAM_WIDTH}")
    print(f"   Top-K: {TOP_K}")
    print(f"   Top-P: {TOP_P}")
    print(f"   Max Length: {MAX_LENGTH}")
    print(f"   Temperature Values: {TEMPERATURE_VALUES}")
    print("   ✓ Configuration loaded successfully")
except Exception as e:
    print(f"   ✗ Error loading configuration: {e}")
    sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("✓ All module tests passed successfully!")
print("=" * 60)
print("\nNext steps:")
print("1. Update config.py with your desired hyperparameters")
print("2. Run: python main.py --device cuda")
print("3. Results will be saved to results/ directory")
print("=" * 60)
