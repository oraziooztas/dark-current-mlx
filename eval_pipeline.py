"""
eval_pipeline.py: MLX-based psychometric "Dark Current" evaluator for LLM Judges.
Author: Orazio
Based on: "LLM Judges Have Dark Current" (Usami et al., 2026)
"""

import argparse
import json
import logging
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm

from mlx_lm import load, generate

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Dummy dataset simulating ambiguous open-ended evaluations.
# In a real scenario, this would load a benchmark like MT-Bench or AlpacaEval.
EVAL_DATASET = [
    {
        "id": "q1",
        "instruction": "Explain quantum computing to a 5-year-old.",
        "model_a_response": "Quantum computers use qubits to do math really fast like magic.",
        "model_b_response": "Imagine a coin spinning. It's heads and tails at the same time until it stops.",
    },
    {
        "id": "q2",
        "instruction": "Write a python function for fibonacci.",
        "model_a_response": "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
        "model_b_response": "def fib(n): \n  a, b = 0, 1\n  for _ in range(n): a, b = b, a+b\n  return a",
    }
]

JUDGE_PROMPT_TEMPLATE = """
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question below. 
You should choose the response that follows the user's instructions better and provides more helpful, harmless, and honest information.

[User Question]
{instruction}

[The Start of Assistant A's Answer]
{model_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{model_b}
[The End of Assistant B's Answer]

Provide your evaluation reasoning first, and then output your final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" if assistant B is better, or "[[Tie]]" for a tie.
"""

def compute_dark_current(scores_ab: List[str], scores_ba: List[str]) -> float:
    """
    Computes a simplified 'Dark Current' score (systematic bias hidden under aggregate ties).
    If the judge is perfectly invariant to position, scores_ab should mirror scores_ba.
    """
    inconsistent_judgments = 0
    total = len(scores_ab)
    
    for score_ab, score_ba in zip(scores_ab, scores_ba):
        # A consistent judge should swap A and B when the order is swapped.
        if score_ab == "[[A]]" and score_ba != "[[B]]":
            inconsistent_judgments += 1
        elif score_ab == "[[B]]" and score_ba != "[[A]]":
            inconsistent_judgments += 1
        elif score_ab == "[[Tie]]" and score_ba != "[[Tie]]":
            inconsistent_judgments += 1
            
    # Normalize to 0.0 (no dark current bias) to 1.0 (extreme hidden bias)
    return inconsistent_judgments / max(total, 1)

def extract_verdict(response: str) -> str:
    if "[[A]]" in response: return "[[A]]"
    if "[[B]]" in response: return "[[B]]"
    if "[[Tie]]" in response: return "[[Tie]]"
    return "[[Unknown]]"

def run_evaluation(model_id: str, max_tokens: int = 256):
    logging.info(f"Loading local model {model_id} via MLX (Apple Silicon Native)...")
    model, tokenizer = load(model_id)
    
    scores_ab = []
    scores_ba = []
    
    logging.info("Running evaluation passes to measure Position Bias / Dark Current...")
    for item in tqdm(EVAL_DATASET):
        # Forward Pass (A vs B)
        prompt_ab = JUDGE_PROMPT_TEMPLATE.format(
            instruction=item["instruction"], 
            model_a=item["model_a_response"], 
            model_b=item["model_b_response"]
        )
        # Apply chat template
        messages_ab = [{"role": "user", "content": prompt_ab}]
        formatted_prompt_ab = tokenizer.apply_chat_template(messages_ab, add_generation_prompt=True)
        
        output_ab = generate(model, tokenizer, prompt=formatted_prompt_ab, max_tokens=max_tokens, verbose=False)
        scores_ab.append(extract_verdict(output_ab))
        
        # Reverse Pass (B vs A)
        prompt_ba = JUDGE_PROMPT_TEMPLATE.format(
            instruction=item["instruction"], 
            model_a=item["model_b_response"],  # Swapped
            model_b=item["model_a_response"]   # Swapped
        )
        messages_ba = [{"role": "user", "content": prompt_ba}]
        formatted_prompt_ba = tokenizer.apply_chat_template(messages_ba, add_generation_prompt=True)
        
        output_ba = generate(model, tokenizer, prompt=formatted_prompt_ba, max_tokens=max_tokens, verbose=False)
        scores_ba.append(extract_verdict(output_ba))
        
    dark_current_score = compute_dark_current(scores_ab, scores_ba)
    
    logging.info(f"Evaluation Complete!")
    logging.info(f"Scores (A vs B): {scores_ab}")
    logging.info(f"Scores (B vs A): {scores_ba}")
    logging.info(f"Psychometric 'Dark Current' Bias Score: {dark_current_score:.2f} (0.0 is perfect)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dark Current Bias Eval on MLX.")
    parser.add_argument("--model", type=str, default="mlx-community/Meta-Llama-3-8B-Instruct-4bit", 
                        help="HuggingFace model ID compatible with mlx-lm")
    args = parser.parse_args()
    
    run_evaluation(args.model)
