"""Verdict parsing and position-swap consistency checks."""

from __future__ import annotations

import re

from .models import Verdict

_VERDICT_RE = re.compile(r"\[\[\s*(A|B|Tie)\s*\]\]", re.IGNORECASE)


def extract_verdict(text: str) -> Verdict:
    """Parse the last strict verdict marker in a judge response."""

    matches = _VERDICT_RE.findall(text or "")
    if not matches:
        return Verdict.UNKNOWN

    last = matches[-1].lower()
    if last == "a":
        return Verdict.A
    if last == "b":
        return Verdict.B
    if last == "tie":
        return Verdict.TIE
    return Verdict.UNKNOWN


def expected_swapped_verdict(verdict: Verdict) -> Verdict:
    """Return what a swapped-order call should emit if the judge is invariant."""

    if verdict == Verdict.A:
        return Verdict.B
    if verdict == Verdict.B:
        return Verdict.A
    if verdict == Verdict.TIE:
        return Verdict.TIE
    return Verdict.UNKNOWN


def is_swap_consistent(verdict_ab: Verdict, verdict_ba: Verdict) -> bool:
    """Strict position-swap consistency check."""

    if Verdict.UNKNOWN in {verdict_ab, verdict_ba}:
        return False
    return expected_swapped_verdict(verdict_ab) == verdict_ba


def original_winner(position_verdict: Verdict, order: str) -> str:
    """Map A/B position verdict to original model ids for summary output."""

    if position_verdict == Verdict.TIE:
        return "tie"
    if position_verdict == Verdict.UNKNOWN:
        return "unknown"
    if order == "ab":
        return "model_a" if position_verdict == Verdict.A else "model_b"
    if order == "ba":
        return "model_b" if position_verdict == Verdict.A else "model_a"
    raise ValueError("order must be 'ab' or 'ba'")
