# Dark Current MLX Evaluator

This repository is a lightweight, local test suite designed to diagnose the "Dark Current" (systematic psychometric biases hidden in aggregate metrics) of LLM-as-a-judge models.

It is heavily inspired by the June 2026 paper: *"LLM Judges Have Dark Current: A Psychometric Datasheet for LLM-as-a-Judge Evaluation" (Usami et al.)*.

## Why this exists (The Bottleneck)
Evaluating open-source LLMs as judges currently requires either expensive API calls to proprietary models (for baseline anchoring) or heavy GPU infrastructure to run evaluations at scale. This project leverages **Apple Silicon (MLX)** to run fast, local, zero-cost psychometric evaluation passes on M-series Macs. 

This enables researchers to iterate on prompt bias mitigation without cloud compute constraints.

## Features
- **Apple Silicon Native**: Powered by `mlx-lm` for optimal speed and memory footprint on M2/M3 chips.
- **Position Bias Reversal**: Automatically runs A-vs-B and B-vs-A queries to compute the strict inconsistency metric.
- **Fast Prototyping**: Managed via `uv`, incredibly fast setup.

## Quickstart

1. Clone the repo and sync dependencies using `uv`:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e .
   ```

2. Run the evaluation pipeline using a quantized Llama-3 model (runs comfortably on 18GB Unified Memory):
   ```bash
   python eval_pipeline.py --model mlx-community/Meta-Llama-3-8B-Instruct-4bit
   ```

## Next Steps / Contribution
I developed this locally to test if we can extract Item Response Theory (IRT) parameters on the edge. If the NLP lab finds this helpful for automating the psychometric datasheet generation for future NeurIPS submissions, I am open to contributing part-time to expand the supported benchmarks (e.g., MT-Bench, AlpacaEval).
