"""Evaluation pipeline and JSONL/report writers."""

from __future__ import annotations

import json
import platform
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import GenerationConfig, PairwiseItem, Verdict
from .prompts import PAIRWISE_JUDGE_TEMPLATE_ID, render_pairwise_prompt
from .runners import JudgeRunner
from .verdicts import extract_verdict, is_swap_consistent, original_winner


@dataclass(frozen=True)
class RunContext:
    """Metadata shared by all records from a run."""

    eval_id: str
    judge_model: str
    generation_config: GenerationConfig
    hardware_env: dict[str, Any]
    prompt_template_id: str = PAIRWISE_JUDGE_TEMPLATE_ID


def hardware_env() -> dict[str, str]:
    """Capture lightweight reproducibility metadata."""

    mac_ver = platform.mac_ver()
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "mac_version": mac_ver[0] if mac_ver else "",
        "python_version": platform.python_version(),
    }


def run_items(
    items: list[PairwiseItem],
    runner: JudgeRunner,
    judge_model: str,
    generation_config: GenerationConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run normal and swapped pairwise judge calls for every item."""

    context = RunContext(
        eval_id=str(uuid.uuid4()),
        judge_model=judge_model,
        generation_config=generation_config,
        hardware_env=hardware_env(),
    )
    records: list[dict[str, Any]] = []

    for item in items:
        for order in ("ab", "ba"):
            call = render_pairwise_prompt(item, order)
            started = time.perf_counter()
            raw_generation = runner.generate(call.prompt, generation_config)
            latency_s = time.perf_counter() - started
            verdict = extract_verdict(raw_generation)

            records.append(
                {
                    "eval_id": context.eval_id,
                    "item_id": item.item_id,
                    "judge_model": context.judge_model,
                    "hardware_env": context.hardware_env,
                    "generation_config": context.generation_config.to_dict(),
                    "prompt_template_id": context.prompt_template_id,
                    "order": order,
                    "inputs": {
                        "model_a_id": call.model_a_id,
                        "model_b_id": call.model_b_id,
                        "instruction": item.instruction,
                        "response_a": call.response_a,
                        "response_b": call.response_b,
                        "metadata": item.metadata,
                    },
                    "outputs": {
                        "raw_generation": raw_generation,
                        "parsed_position_winner": verdict.value,
                        "parsed_original_winner": original_winner(verdict, order),
                    },
                    "timing": {
                        "latency_s": round(latency_s, 6),
                    },
                }
            )

    return records, summarize_records(records)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute strict position-swap consistency metrics."""

    by_item: dict[str, dict[str, Verdict]] = {}
    unknown_calls = 0
    for record in records:
        verdict = Verdict(record["outputs"]["parsed_position_winner"])
        if verdict == Verdict.UNKNOWN:
            unknown_calls += 1
        by_item.setdefault(record["item_id"], {})[record["order"]] = verdict

    evaluated_pairs = 0
    inconsistent_pairs = 0
    pair_details: list[dict[str, Any]] = []
    for item_id, verdicts in sorted(by_item.items()):
        if "ab" not in verdicts or "ba" not in verdicts:
            continue
        evaluated_pairs += 1
        consistent = is_swap_consistent(verdicts["ab"], verdicts["ba"])
        if not consistent:
            inconsistent_pairs += 1
        pair_details.append(
            {
                "item_id": item_id,
                "ab": verdicts["ab"].value,
                "ba": verdicts["ba"].value,
                "swap_consistent": consistent,
            }
        )

    score = inconsistent_pairs / evaluated_pairs if evaluated_pairs else 0.0
    return {
        "total_calls": len(records),
        "total_items": evaluated_pairs,
        "unknown_calls": unknown_calls,
        "swap_consistent_items": evaluated_pairs - inconsistent_pairs,
        "swap_inconsistent_items": inconsistent_pairs,
        "dark_current_score": round(score, 6),
        "pair_details": pair_details,
    }


def write_jsonl(records: list[dict[str, Any]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(payload: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
