#!/usr/bin/env python
"""Quick greedy decoder test."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from decoding_strategies import GreedySearchDecoder
from config import MODEL_NAME, DEVICE

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto" if DEVICE=="cuda" else None, trust_remote_code=True)
if DEVICE=="cpu": model = model.to(DEVICE)
model.eval()

decoder = GreedySearchDecoder(model, tokenizer)
prompts = ["What are the advantages of Scrum?", "Design an advertisement for a home security product."]

print("GREEDY DECODER TEST\n" + "="*60)
for p in prompts:
    for t in [0.5, 1.0]:
        text, _ = decoder.decode(p, max_length=200, temperature=t)
        status = "OK" if text.strip() else "EMPTY"
        print(f"T={t} | {status} | Tokens: {len(text.split())}")
        print(f"  Generation: {text[:100]}...\n")
