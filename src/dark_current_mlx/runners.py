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
    """Lazy `mlx-lm` runner so tests do not import or download MLX models.

    Targets the `mlx-lm>=0.31` API: sampling is configured through
    `make_sampler` (the old `temp=` kwarg of `generate()` is gone) and the
    generation is consumed via `stream_generate`, whose final response carries
    the budget-aware usage stats (`prompt_tokens`, `generation_tokens`, tps,
    peak memory, finish reason) exposed as `last_usage`.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.last_usage: dict[str, object] | None = None
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
        self._align_declared_eos()

    def _align_declared_eos(self) -> None:
        """Add the tokenizer-declared eos token to the wrapper's stop set.

        Some checkpoints (e.g. Llama-3: config eos `<|end_of_text|>` vs chat
        template eos `<|eot_id|>`) declare a chat eos token that is missing
        from the wrapper's `eos_token_ids`, so generation never stops and
        every call burns the full `max_tokens` budget.
        """

        declared_eos = getattr(self._tokenizer, "eos_token", None)
        if not declared_eos:
            return
        try:
            token_id = self._tokenizer.convert_tokens_to_ids(declared_eos)
        except Exception:
            return
        if token_id is not None and token_id not in self._tokenizer.eos_token_ids:
            self._tokenizer.add_eos_token(declared_eos)

    def generate(self, prompt: str, config: GenerationConfig) -> str:
        self._load()
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

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

        sampler = make_sampler(temp=config.temperature)
        segments: list[str] = []
        final_response = None
        for response in stream_generate(
            self._model,
            self._tokenizer,
            prompt=formatted_prompt,
            max_tokens=config.max_tokens,
            sampler=sampler,
        ):
            segments.append(response.text)
            final_response = response

        self.last_usage = None
        if final_response is not None:
            self.last_usage = {
                "prompt_tokens": final_response.prompt_tokens,
                "generation_tokens": final_response.generation_tokens,
                "prompt_tps": round(final_response.prompt_tps, 3),
                "generation_tps": round(final_response.generation_tps, 3),
                "peak_memory_gb": round(final_response.peak_memory, 3),
                "finish_reason": final_response.finish_reason,
            }

        return "".join(segments)
