"""Core data models for pairwise judge evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    """Strict parser output for a pairwise judge verdict."""

    A = "A"
    B = "B"
    TIE = "tie"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GenerationConfig:
    """Deterministic generation settings for a judge run."""

    max_tokens: int = 256
    temperature: float = 0.0
    seed: int = 42

    def to_dict(self) -> dict[str, int | float]:
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class PairwiseItem:
    """One original A/B comparison item."""

    item_id: str
    instruction: str
    response_a: str
    response_b: str
    model_a_id: str = "model_a"
    model_b_id: str = "model_b"
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PairwiseItem":
        """Accept both the current schema and the first prototype's field names."""

        item_id = payload.get("item_id") or payload.get("id")
        response_a = payload.get("response_a") or payload.get("model_a_response")
        response_b = payload.get("response_b") or payload.get("model_b_response")

        missing = [
            name
            for name, value in {
                "item_id/id": item_id,
                "instruction": payload.get("instruction"),
                "response_a/model_a_response": response_a,
                "response_b/model_b_response": response_b,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(f"Missing required pairwise item fields: {', '.join(missing)}")

        return cls(
            item_id=str(item_id),
            instruction=str(payload["instruction"]),
            response_a=str(response_a),
            response_b=str(response_b),
            model_a_id=str(payload.get("model_a_id", "model_a")),
            model_b_id=str(payload.get("model_b_id", "model_b")),
            metadata=dict(payload.get("metadata") or {}),
        )


@dataclass(frozen=True)
class JudgeCall:
    """A rendered prompt for one order of a pairwise item."""

    item: PairwiseItem
    order: str
    prompt: str

    @property
    def response_a(self) -> str:
        return self.item.response_a if self.order == "ab" else self.item.response_b

    @property
    def response_b(self) -> str:
        return self.item.response_b if self.order == "ab" else self.item.response_a

    @property
    def model_a_id(self) -> str:
        return self.item.model_a_id if self.order == "ab" else self.item.model_b_id

    @property
    def model_b_id(self) -> str:
        return self.item.model_b_id if self.order == "ab" else self.item.model_a_id
