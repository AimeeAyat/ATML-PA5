#!/usr/bin/env python
"""Test reward model scoring."""
from metrics import RewardModelScorer
from config import REWARD_MODEL_NAME

print("=" * 70)
print("REWARD MODEL TEST")
print("=" * 70)
print(f"\nReward Model: {REWARD_MODEL_NAME}\n")

# Initialize reward model
scorer = RewardModelScorer(REWARD_MODEL_NAME)

# Test responses
test_responses = [
    "This is a very good and helpful response with clear explanations.",
    "Bad response. Short.",
    "A comprehensive answer that addresses all aspects of the question with detailed examples and explanations.",
    "",
]

print(f"Using Dummy Scorer: {scorer.is_dummy_scorer()}\n")
print("Sample Scores:")
print("-" * 70)

for i, response in enumerate(test_responses, 1):
    score = scorer.score_response_only(response)
    preview = response[:50] + "..." if len(response) > 50 else response if response else "[EMPTY]"
    print(f"{i}. Score: {score:7.4f} | {preview}")

print("\n" + "=" * 70)
if scorer.is_dummy_scorer():
    print("WARNING: Dummy scorer is being used!")
    print("Real reward model not loaded. All scores are constant 0.5")
else:
    print("SUCCESS: Real reward model is being used!")
    print("Scores vary based on response quality")
print("=" * 70)
