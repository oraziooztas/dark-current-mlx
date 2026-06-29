"""Judge model runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .models import GenerationConfig


class JudgeRunner(Protocol):
    """Minimal interface used by the evaluation pipeline."""

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        """Return raw judge text for a rendered prompt."""


@dataclass
class MockRunner:
    """Deterministic runner for offline tests and JSONL plumbing checks."""

    strategy: str = "position"

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        if self.strategy == "tie":
            return "Mock rationale: both answers are acceptable.\n[[Tie]]"
        if self.strategy == "length":
            a_start = prompt.find("[The Start of Assistant A's Answer]")
            b_start = prompt.find("[The Start of Assistant B's Answer]")
            if a_start != -1 and b_start != -1:
                a_text = prompt[a_start:b_start]
                b_text = prompt[b_start:]
                return "[[A]]" if len(a_text) >= len(b_text) else "[[B]]"
        return "Mock rationale: fixed position-biased baseline for smoke tests.\n[[A]]"


class MlxRunner:
    """Lazy `mlx-lm` runner so tests do not import or download MLX models."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import mlx.core as mx
            from mlx_lm import load
        except ImportError as exc:  # pragma: no cover - depends on optional MLX install
            raise RuntimeError(
                "MLX dependencies are not installed. Run `uv pip install -e '.[mlx]'` first."
            ) from exc

        self._model, self._tokenizer = load(self.model_id)
        self._mx = mx

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        self._load()
        from mlx_lm import generate

        try:
            self._mx.random.seed(config.seed)
        except Exception:
            pass

        messages = [{"role": "user", "content": prompt}]
        try:
            formatted_prompt = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except TypeError:
            formatted_prompt = self._tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )

        if not isinstance(formatted_prompt, str) and hasattr(self._tokenizer, "decode"):
            formatted_prompt = self._tokenizer.decode(formatted_prompt)

        return generate(
            self._model,
            self._tokenizer,
            prompt=formatted_prompt,
            max_tokens=config.max_tokens,
            temp=config.temperature,
            verbose=False,
        )
