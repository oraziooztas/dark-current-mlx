"""Dataset loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .models import PairwiseItem


def load_jsonl(path: str | Path, limit: int | None = None) -> list[PairwiseItem]:
    """Load pairwise items from a JSONL file."""

    items: list[PairwiseItem] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                payload = json.loads(stripped)
                items.append(PairwiseItem.from_dict(payload))
            except Exception as exc:  # pragma: no cover - keeps line context for users
                raise ValueError(f"Invalid JSONL item at {path}:{line_number}: {exc}") from exc
            if limit is not None and len(items) >= limit:
                break
    return items


def iter_builtin_sample() -> Iterable[PairwiseItem]:
    """Return a tiny in-memory smoke-test dataset."""

    yield PairwiseItem(
        item_id="sample_quantum_child",
        instruction="Explain quantum computing to a 5-year-old.",
        response_a="Quantum computers use qubits to do math really fast like magic.",
        response_b=(
            "Imagine a coin spinning. It is not just heads or tails until it stops. "
            "A quantum computer uses pieces a bit like that spinning coin."
        ),
        model_a_id="toy_short",
        model_b_id="toy_analogy",
        metadata={"source": "builtin_sample"},
    )
    yield PairwiseItem(
        item_id="sample_fibonacci",
        instruction="Write a Python function for Fibonacci numbers.",
        response_a="def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)",
        response_b="def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
        model_a_id="toy_recursive",
        model_b_id="toy_iterative",
        metadata={"source": "builtin_sample"},
    )
