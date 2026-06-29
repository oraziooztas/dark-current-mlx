"""Command-line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset import iter_builtin_sample, load_jsonl
from .models import GenerationConfig
from .pipeline import run_items, write_json, write_jsonl
from .runners import MlxRunner, MockRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local position-swap checks for LLM-as-a-judge models."
    )
    parser.add_argument(
        "--input",
        default="data/sample_pairwise.jsonl",
        help="Pairwise JSONL dataset. Use 'builtin' for the embedded sample.",
    )
    parser.add_argument(
        "--output",
        default="results/run.jsonl",
        help="Output JSONL file with one record per judge call.",
    )
    parser.add_argument(
        "--report",
        default="results/summary.json",
        help="Output JSON file with aggregate consistency metrics.",
    )
    parser.add_argument(
        "--runner",
        choices=["mlx", "mock"],
        default="mlx",
        help="Use 'mock' for offline validation without loading a model.",
    )
    parser.add_argument(
        "--mock-strategy",
        choices=["position", "length", "tie"],
        default="position",
        help="Deterministic mock behavior for dry runs.",
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Meta-Llama-3-8B-Instruct-4bit",
        help="Hugging Face model id for mlx-lm.",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.input == "builtin":
        items = list(iter_builtin_sample())
        if args.limit is not None:
            items = items[: args.limit]
    else:
        items = load_jsonl(args.input, limit=args.limit)

    config = GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )
    runner = MockRunner(args.mock_strategy) if args.runner == "mock" else MlxRunner(args.model)
    records, summary = run_items(
        items=items,
        runner=runner,
        judge_model=args.model if args.runner == "mlx" else f"mock:{args.mock_strategy}",
        generation_config=config,
    )

    write_jsonl(records, args.output)
    write_json(summary, args.report)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote records to {Path(args.output)}")
    print(f"Wrote report to {Path(args.report)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
